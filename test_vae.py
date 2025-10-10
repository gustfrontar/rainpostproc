import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional

class SkipConnectionVAE(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        latent_dim: int = 64,
        input_height: int = 100,
        input_width: int = 120,
        hidden_dims: List[int] = None,
        use_bilinear: bool = True
    ):
        super(SkipConnectionVAE, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.input_width = input_width
        self.use_bilinear = use_bilinear
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
        
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        
        # Build encoder and calculate output dimensions
        self.encoder, self.encoder_features, self.encoded_height, self.encoded_width = self._build_encoder()
        
        # Latent space
        encoded_size = hidden_dims[-1] * self.encoded_height * self.encoded_width
        self.fc_mu = nn.Linear(encoded_size, latent_dim)
        self.fc_logvar = nn.Linear(encoded_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, hidden_dims[-1] * self.encoded_height * self.encoded_width)
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        print(f"Model initialized:")
        print(f"  Input: {input_channels}x{input_height}x{input_width}")
        print(f"  Encoded: {hidden_dims[-1]}x{self.encoded_height}x{self.encoded_width}")
        print(f"  Latent: {latent_dim}")
        print(f"  Output: {output_channels}x{input_height}x{input_width}")
    
    def _build_encoder(self) -> Tuple[nn.Module, List[Tuple[int, int, int]], int, int]:
        """Build encoder and calculate feature dimensions at each layer"""
        layers = nn.ModuleList()
        feature_dims = []  # Store (channels, height, width) at each layer
        
        in_channels = self.input_channels
        current_height, current_width = self.input_height, self.input_width
        
        # Store input dimensions
        feature_dims.append((in_channels, current_height, current_width))
        
        for i, h_dim in enumerate(self.hidden_dims):
            conv_layer = nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1)
            bn_layer = nn.BatchNorm2d(h_dim)
            relu_layer = nn.ReLU(inplace=True)
            
            layers.extend([conv_layer, bn_layer, relu_layer])
            
            # Calculate output dimensions after conv
            current_height = (current_height + 2 * 1 - 3) // 2 + 1
            current_width = (current_width + 2 * 1 - 3) // 2 + 1
            
            # Store feature dimensions after this block
            feature_dims.append((h_dim, current_height, current_width))
            in_channels = h_dim
        
        # Verify dimensions are valid
        assert current_height > 0 and current_width > 0, "Input dimensions too small for encoder"
        
        return nn.Sequential(*layers), feature_dims, current_height, current_width
    
    def _build_decoder(self) -> nn.ModuleList:
        """Build decoder with proper skip connection channel handling"""
        layers = nn.ModuleList()
        hidden_dims_rev = self.hidden_dims[::-1]
        
        # Reverse the feature dimensions for decoder
        decoder_feature_dims = self.encoder_features[::-1]
        
        # Build decoder blocks
        for i in range(len(hidden_dims_rev)):
            # Current decoder input channels
            in_ch = hidden_dims_rev[i]
            
            # Skip connection channels from corresponding encoder layer
            skip_ch = decoder_feature_dims[i+1][0] if i < len(hidden_dims_rev) - 1 else self.input_channels
            
            # Output channels for this block
            out_ch = hidden_dims_rev[i+1] if i < len(hidden_dims_rev) - 1 else hidden_dims_rev[-1] // 2
            
            # Target dimensions for this block
            target_height = decoder_feature_dims[i+1][1] if i < len(hidden_dims_rev) - 1 else self.input_height
            target_width = decoder_feature_dims[i+1][2] if i < len(hidden_dims_rev) - 1 else self.input_width
            
            if i == len(hidden_dims_rev) - 1:
                # Final layer
                layers.append(
                    FinalDecoderBlock(
                        in_channels=in_ch,
                        out_channels=self.output_channels,
                        skip_channels=skip_ch,
                        target_height=target_height,
                        target_width=target_width,
                        use_bilinear=self.use_bilinear
                    )
                )
            else:
                # Intermediate layers
                layers.append(
                    DecoderBlockWithSkip(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        skip_channels=skip_ch,
                        target_height=target_height,
                        target_width=target_width,
                        use_bilinear=self.use_bilinear
                    )
                )
        
        return layers
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Encode input and return features for skip connections"""
        features = [x]  # Store input as first feature
        
        current = x
        for layer in self.encoder:
            current = layer(current)
            # Store output after each convolutional layer
            if isinstance(layer, nn.Conv2d):
                features.append(current)
        
        encoded_flat = current.view(current.size(0), -1)
        mu = self.fc_mu(encoded_flat)
        logvar = self.fc_logvar(encoded_flat)
        
        return mu, logvar, features
    
    def decode(self, z: torch.Tensor, features: List[torch.Tensor]) -> torch.Tensor:
        """Decode with skip connections"""
        # Initial projection
        h = self.fc_decode(z)
        h = h.view(-1, self.hidden_dims[-1], self.encoded_height, self.encoded_width)
        
        # Reverse features for decoder (skip the input feature for now)
        skip_features = features[1:][::-1]  # Skip input, reverse order
        input_feature = features[0]  # Keep input for final layer
        
        # Forward through decoder blocks
        current = h
        for i, block in enumerate(self.decoder):
            # Get appropriate skip connection
            if i < len(skip_features):
                skip = skip_features[i]
            else:
                # For final layer, use original input
                skip = input_feature
            
            current = block(current, skip)
        
        return current
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode and get features for skip connections
        mu, logvar, features = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Decode with skip connections
        reconstruction = self.decode(z, features)
        
        return reconstruction, mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    #def predict(self, x: torch.Tensor) -> torch.Tensor:
    #    """Deterministic prediction (mean of the distribution)"""
    #    with torch.no_grad():
    #        reconstruction, _, _ = self.forward(x)
    #        return reconstruction

    def predict(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Predict precipitation from temperature.
        
        Args:
            x: Input temperature images
            num_samples: Number of samples to draw for probabilistic prediction
                    (if >1, returns multiple predictions)
        
        Returns:
            If num_samples=1: mean prediction (torch.Tensor)
            If num_samples>1: tuple of (samples, mean, std)
        """
        with torch.no_grad():
            if num_samples == 1:
                # Deterministic prediction: use mean of the distribution
                mu, logvar, features = self.encode(x)
                # Use mu directly (no sampling) for mean prediction
                reconstruction = self.decode(mu, features)
                return reconstruction
            else:
                # Probabilistic prediction: sample multiple times
                mu, logvar, features = self.encode(x)
                samples = []
                
                for _ in range(num_samples):
                    z = self.reparameterize(mu, logvar)
                    sample_recon = self.decode(z, features)
                    samples.append(sample_recon)
                
                samples = torch.stack(samples)  # [num_samples, batch, channels, H, W]
                
                # Calculate mean and std across samples
                mean_pred = samples.mean(dim=0)
                std_pred = samples.std(dim=0)
                
                return samples, mean_pred, std_pred

    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Explicit method for mean prediction"""
        with torch.no_grad():
            mu, logvar, features = self.encode(x)
            # For mean prediction, we use mu directly (no random sampling)
            reconstruction = self.decode(mu, features)
            return reconstruction

    def sample_from_prior(self, condition: torch.Tensor = None, 
                         noise_scale: float = 1.0) -> torch.Tensor:
        """
        Sample precipitation patterns.
        
        Args:
            num_samples: Number of samples to generate
            condition: Iinput for conditional generation
            use_posterior: If True, sample from posterior p(z|temperature). 
                        If False, sample from prior N(0,I) (less meaningful)
            noise_scale: Scale for additional noise when sampling from posterior
        """
        with torch.no_grad():
            if condition is not None:
                # Encode the temperature condition to get the posterior distribution
                mu, logvar, features = self.encode(condition)
                
                # Sample from the posterior distribution: z ~ N(mu, sigma)
                # This gives samples conditioned on the temperature
                z = self.reparameterize(mu, logvar)
                    
                # Optionally add extra noise for more diversity
                if noise_scale > 1.0:
                    extra_noise = torch.randn_like(z) * (noise_scale - 1.0)
                    z = z + extra_noise
                
                reconstruction = self.decode(z, features)
                
            
            return reconstruction


class DecoderBlockWithSkip(nn.Module):
    """Decoder block with skip connection"""
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int, 
                 target_height: int, target_width: int, use_bilinear: bool = True):
        super(DecoderBlockWithSkip, self).__init__()
        
        self.target_height = target_height
        self.target_width = target_width
        self.use_bilinear = use_bilinear
        
        if use_bilinear:
            self.upsample = nn.Upsample(size=(target_height, target_width), mode='bilinear', align_corners=False)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            # Calculate stride and padding for transpose conv to hit target size
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels + skip_channels, out_channels, 3, 
                                  stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.use_bilinear:
            # Upsample to target size
            x = self.upsample(x)
        else:
            x = self.conv[0](x)  # Only the transpose conv
            x = self.conv[1](x)  # BN
            x = self.conv[2](x)  # ReLU
        
        # Ensure skip connection matches current spatial dimensions
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        if self.use_bilinear:
            x = self.conv(x)
        
        return x


class FinalDecoderBlock(nn.Module):
    """Final decoder block to produce output"""
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int, 
                 target_height: int, target_width: int, use_bilinear: bool = True):
        super(FinalDecoderBlock, self).__init__()
        
        self.target_height = target_height
        self.target_width = target_width
        self.use_bilinear = use_bilinear
        
        if use_bilinear:
            self.upsample = nn.Upsample(size=(target_height, target_width), mode='bilinear', align_corners=False)
            self.conv = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        else:
            self.conv = nn.ConvTranspose2d(in_channels + skip_channels, out_channels, 3, 
                                          stride=2, padding=1, output_padding=1)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.use_bilinear:
            # Upsample to target size
            x = self.upsample(x)
        
        if skip is not None:
            # Ensure skip connection matches current spatial dimensions
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv(x)
        
        return x


def vae_regression_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
                       logvar: torch.Tensor, beta: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined loss for regression VAE"""
    # Reconstruction loss (MSE for regression)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


# Test the fixed implementation
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test configuration (similar to your setup)
    config = {
        'input_channels': 1,
        'output_channels': 1, 
        'latent_dim': 128,
        'input_height': 96,   # Non-square
        'input_width': 128,   # Non-square
        'hidden_dims': [64, 128, 256],  # Fewer layers for testing
        'use_bilinear': True,
    }
    
    # Create model
    model = SkipConnectionVAE(**config)
    model.to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    batch_size = 2
    dummy_temp = torch.randn(batch_size, config['input_channels'], 
                           config['input_height'], config['input_width']).to(device)
    dummy_precip = torch.randn(batch_size, config['output_channels'], 
                             config['input_height'], config['input_width']).to(device)
    
    try:
        # Forward pass test
        recon, mu, logvar = model(dummy_temp)
        
        print(f"✅ Forward pass successful!")
        print(f"Input shape: {dummy_temp.shape}")
        print(f"Output shape: {recon.shape}")
        print(f"Latent mean shape: {mu.shape}")
        
        # Loss calculation test
        loss, recon_loss, kl_loss = vae_regression_loss(recon, dummy_precip, mu, logvar)
        print(f"✅ Loss calculation successful!")
        print(f"Total loss: {loss.item():.4f}")
        print(f"Recon loss: {recon_loss.item():.4f}")
        print(f"KL loss: {kl_loss.item():.4f}")
        
        # Backward pass test
        loss.backward()
        print(f"✅ Backward pass successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Debugging info:")
        
        # Debug encoder
        print("\nEncoder features:")
        for i, (ch, h, w) in enumerate(model.encoder_features):
            print(f"  Layer {i}: channels={ch}, size={h}x{w}")
        
        # Debug decoder structure
        print("\nDecoder blocks:")
        for i, block in enumerate(model.decoder):
            if hasattr(block, 'conv'):
                if isinstance(block.conv, nn.Sequential):
                    conv_layer = block.conv[0] if isinstance(block.conv[0], (nn.Conv2d, nn.ConvTranspose2d)) else block.conv
                else:
                    conv_layer = block.conv
                print(f"  Block {i}: in_channels={conv_layer.in_channels}, out_channels={conv_layer.out_channels}")