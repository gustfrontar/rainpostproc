import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import os

class SkipConnectionVAE(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        latent_dim: int = 64,
        input_height: int = 100,
        input_width: int = 120,
        hidden_dims: List[int] = None,
        use_bilinear: bool = True,
        use_batchnorm: bool = True
    ):
        super(SkipConnectionVAE, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.input_width = input_width
        self.use_bilinear = use_bilinear
        self.use_batchnorm = use_batchnorm
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        
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
        
        # Print encoder feature dimensions for debugging
        print("Encoder feature dimensions:")
        for i, (ch, h, w) in enumerate(self.encoder_features):
            print(f"    Layer {i}: {ch} channels, {h}x{w}")
    
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
            if self.use_batchnorm:
                layers.extend([conv_layer, bn_layer, relu_layer])
            else:   
                layers.extend([conv_layer, relu_layer])
            
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
        
        # Reverse the feature dimensions for decoder (skip the input)
        decoder_feature_dims = self.encoder_features[1:][::-1]  # Skip input, reverse
        decoder_feature_dims.append(self.encoder_features[0])   # Add input at end for final layer
        
        print("Decoder feature dimensions:")
        for i, (ch, h, w) in enumerate(decoder_feature_dims):
            print(f"    Layer {i}: {ch} channels, {h}x{w}")
        
        # Build decoder blocks
        for i in range(len(hidden_dims_rev)):
            # Current decoder input channels
            in_ch = hidden_dims_rev[i]
            
            # Skip connection channels from corresponding encoder layer
            skip_ch = decoder_feature_dims[i][0]
            
            # Output channels for this block
            if i < len(hidden_dims_rev) - 1:
                out_ch = hidden_dims_rev[i + 1]
            else:
                out_ch = self.output_channels
            
            # Target dimensions for this block
            target_height = decoder_feature_dims[i][1]
            target_width = decoder_feature_dims[i][2]
            
            print(f"Decoder block {i}: in_ch={in_ch}, skip_ch={skip_ch}, out_ch={out_ch}, target={target_height}x{target_width}")
            
            if i == len(hidden_dims_rev) - 1:
                # Final layer - always upsample to input size
                layers.append(
                    FinalDecoderBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        skip_channels=skip_ch,
                        target_height=self.input_height,  # Always target full input size
                        target_width=self.input_width,    # Always target full input size
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
        # We use features[1:] to skip the input, then reverse
        skip_features = features[1:][::-1]  # Skip input, reverse order
        # Add input as the final skip connection
        skip_features.append(features[0])
        
        #print(f"Decoding: starting with {h.shape}")
        #print(f"Skip features: {[f.shape for f in skip_features]}")
        
        # Forward through decoder blocks
        current = h
        for i, block in enumerate(self.decoder):
            skip = skip_features[i]
            #print(f"Block {i}: input {current.shape}, skip {skip.shape}")
            current = block(current, skip)
            #print(f"Block {i}: output {current.shape}")
        
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
        self.skip_channels = skip_channels
        
        # First, process the input (upsample + initial conv)
        if use_bilinear:
            self.upsample = nn.Upsample(size=(target_height, target_width), mode='bilinear', align_corners=False)
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # Then process the concatenated features (skip + processed input)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Step 1: Process the input (upsample + conv)
        if self.use_bilinear:
            x = self.upsample(x)
        x = self.conv1(x)
        
        # Step 2: Concatenate with skip connection
        if skip is not None:
            # Ensure skip connection matches current spatial dimensions
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        # Step 3: Process the combined features
        x = self.conv2(x)
        
        return x


class FinalDecoderBlock(nn.Module):
    """Final decoder block to produce output - FIXED to always output input size"""
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int, 
                 target_height: int, target_width: int, use_bilinear: bool = True):
        super(FinalDecoderBlock, self).__init__()
        
        self.target_height = target_height
        self.target_width = target_width
        self.use_bilinear = use_bilinear
        self.skip_channels = skip_channels
        
        # First, upsample to target size (which should be input size)
        self.upsample = nn.Upsample(size=(target_height, target_width), mode='bilinear', align_corners=False)
        # Process the upsampled features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Then process with skip connection (original input)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            # No activation in final layer for regression
            nn.Softplus()
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Step 1: Upsample to target size and process
        if self.use_bilinear:
            x = self.upsample(x)
        x = self.conv1(x)
        
        # Step 2: Handle skip connection (original input)
        if skip is not None:
            # Skip should already be at target size (original input)
            if x.shape[-2:] != skip.shape[-2:]:
                # If not, interpolate skip to match x
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        # Step 3: Final processing
        x = self.conv2(x)
        
        return x


def vae_regression_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
                       logvar: torch.Tensor, beta: float = 0.1 , free_bits: float = 0.01 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined loss for regression VAE"""
    # Reconstruction loss (MSE for regression)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = free_bits_kl_loss(mu, logvar, free_bits=free_bits)
    #kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

def free_bits_kl_loss(mu, logvar, free_bits=2.0):
    # KL per dimension: -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # Free bits: each dimension must have KL > free_bits
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    
    # Sum over dimensions, average over batch
    kl_loss = torch.mean(torch.sum(kl_per_dim, dim=1))
    return kl_loss

def beta_kl_scheduler(epoch , max_beta_kl=0.1 , free_bits=1.0):
    if epoch < 10:
        return {'beta_kl': 0.0, 'free_bits': 0.0}           # Warmup - learn to reconstruct
    elif epoch < 20:
        return {'beta_kl': max_beta_kl/10.0 , 'free_bits': free_bits}         # Gentle push
    elif epoch < 30:
        return {'beta_kl': max_beta_kl/2.0, 'free_bits': free_bits * 2.0 }         # Stronger push  
    else:
        return {'beta_kl': max_beta_kl , 'free_bits': free_bits * 3.0 }          # Full regularization
        
def init_weights(m):
    """Initialize weights for different layer types"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # He initialization for ReLU networks
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, nn.BatchNorm2d):
        # Standard BN initialization
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, nn.Linear):
        # Xavier initialization for linear layers
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)        
        
def train_vae_model(
    model,
    train_loader,
    val_loader=None,
    epochs=100,
    learning_rate=1e-4,
    device='cuda',
    max_beta_kl=0.1,
    free_bits=0.01,
    use_lr_scheduler=True,
    early_stopping_patience=20,
    grad_clip=1.0,
    save_best=True,
    checkpoint_dir='checkpoints'
):
    """
    Train the VAE model for temperature to precipitation regression.
    
    Args:
        model: The VAE model instance
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        device: Training device ('cuda' or 'cpu')
        beta_kl: Weight for KL divergence loss
        use_lr_scheduler: Whether to use learning rate scheduling
        early_stopping_patience: Early stopping patience
        grad_clip: Gradient clipping value
        save_best: Whether to save best model
        checkpoint_dir: Directory to save checkpoints
    """

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0e-5)
    
    # Learning rate scheduler
    if use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5, verbose=True
        )
    
    # Training history
    history = {
        'train_total_loss': [], 'train_recon_loss': [], 'train_kl_loss': [],
        'val_total_loss': [], 'val_recon_loss': [], 'val_kl_loss': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # Create checkpoint directory
    if save_best and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Device: {device}, KL weight: {max_beta_kl}")
    
    for epoch in range(epochs):
        # === Training Phase ===
        model.train()
        train_total, train_recon, train_kl = 0.0, 0.0, 0.0
        num_train_batches = 0
        
        for batch_idx, (temperature, precipitation) in enumerate(train_loader):
            temperature = temperature.to(device)
            precipitation = precipitation.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_precip, mu, logvar = model(temperature)
            
            
            betasc = beta_kl_scheduler(epoch , max_beta_kl , free_bits)

            total_loss, recon_loss, kl_loss = vae_regression_loss(
                recon_precip, precipitation, mu, logvar, betasc['beta_kl'] , free_bits=betasc['free_bits'] 
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            # Accumulate losses
            train_total += total_loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
            num_train_batches += 1
            
            # Print batch progress
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch:03d} | Batch {batch_idx:03d}/{len(train_loader)} | '
                      f'Loss: {total_loss.item()/temperature.size(0):.4f}')
        
        # Calculate average training losses
        avg_train_total = train_total / len(train_loader.dataset)
        avg_train_recon = train_recon / len(train_loader.dataset)
        avg_train_kl = train_kl / len(train_loader.dataset)
        
        history['train_total_loss'].append(avg_train_total)
        history['train_recon_loss'].append(avg_train_recon)
        history['train_kl_loss'].append(avg_train_kl)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # === Validation Phase ===
        if val_loader is not None:
            model.eval()
            val_total, val_recon, val_kl = 0.0, 0.0, 0.0
            
            with torch.no_grad():
                for temperature, precipitation in val_loader:
                    temperature = temperature.to(device)
                    precipitation = precipitation.to(device)
                    
                    recon_precip, mu, logvar = model(temperature)
                    total_loss, recon_loss, kl_loss = vae_regression_loss(
                        recon_precip, precipitation, mu, logvar, betasc['beta_kl'] , free_bits=betasc['free_bits']
                    )
                    
                    val_total += total_loss.item()
                    val_recon += recon_loss.item()
                    val_kl += kl_loss.item()
            
            avg_val_total = val_total / len(val_loader.dataset)
            avg_val_recon = val_recon / len(val_loader.dataset)
            avg_val_kl = val_kl / len(val_loader.dataset)
            
            history['val_total_loss'].append(avg_val_total)
            history['val_recon_loss'].append(avg_val_recon)
            history['val_kl_loss'].append(avg_val_kl)
            
            # Learning rate scheduling
            if use_lr_scheduler:
                scheduler.step(avg_val_total)
            
            # Early stopping and model checkpointing
            if avg_val_total < best_val_loss:
                best_val_loss = avg_val_total
                early_stopping_counter = 0
                
                if save_best:
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_total,
                        'val_loss': avg_val_total,
                        'history': history,
                        'config': {
                            'beta_kl': betasc['beta_kl'],
                            'learning_rate': learning_rate,
                            'hidden_dims': model.hidden_dims,
                            'latent_dim': model.latent_dim
                        }
                    }, checkpoint_path)
                    print(f'  ✅ Saved best model (val_loss: {avg_val_total:.4f})')
            else:
                early_stopping_counter += 1
        
        # === Print Epoch Summary ===
        print(f'Epoch {epoch:03d}/{epochs} Summary:')
        print(f'  Train - Total: {avg_train_total:.4f}, Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}')
        
        if val_loader is not None:
            print(f'  Val   - Total: {avg_val_total:.4f}, Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f}')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}, Early stopping: {early_stopping_counter}/{early_stopping_patience}')
        
        # Check for KL collapse or explosion
        if avg_train_kl < 1.0:
            print('  ⚠️  Warning: KL loss very low - possible posterior collapse')
        elif avg_train_kl > 1000:
            print('  ⚠️  Warning: KL loss very high - check beta_kl value')
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch} epochs')
            break
    
    print(f'Training completed! Best val loss: {best_val_loss:.4f}')
    
    # Save final model
    if save_best:
        final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pth')
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'config': {
                'beta_kl': betasc['beta_kl'],
                'learning_rate': learning_rate,
                'hidden_dims': model.hidden_dims,
                'latent_dim': model.latent_dim
            }
        }, final_checkpoint_path)
    
    return history, best_val_loss

def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    #import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(history['train_total_loss'], label='Train Total Loss')
    if history['val_total_loss']:
        axes[0, 0].plot(history['val_total_loss'], label='Val Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].set_ylabel('Loss')
    
    # Reconstruction loss
    axes[0, 1].plot(history['train_recon_loss'], label='Train Recon Loss')
    if history['val_recon_loss']:
        axes[0, 1].plot(history['val_recon_loss'], label='Val Recon Loss')
    axes[0, 1].set_title('Reconstruction Loss (MSE)')
    axes[0, 1].legend()
    
    # KL loss
    axes[1, 0].plot(history['train_kl_loss'], label='Train KL Loss')
    if history['val_kl_loss']:
        axes[1, 0].plot(history['val_kl_loss'], label='Val KL Loss')
    axes[1, 0].set_title('KL Divergence Loss')
    axes[1, 0].legend()
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_xlabel('Epoch')
    
    # Learning rate
    axes[1, 1].plot(history['learning_rates'])
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def test_random_cases(model, test_loader, num_cases=10, device='cuda',outpath='./'):
    """
    Test the model on randomly selected cases and visualize results
    """
    model.eval()
    model.to(device)
    
    # Get random batch from test loader
    input_batch, output_batch = next(iter(test_loader))
    
    # Select random indices
    indices = torch.randperm(len(input_batch))[:num_cases]
    input_samples = output_batch[indices].to(device)
    output_samples = input_batch[indices].to(device)
    
    print(f"Testing on {num_cases} random cases...")
    
    with torch.no_grad():
        # Get predictions
        predictions = model.predict_mean(input_samples)
        
        # Generate multiple samples for uncertainty (optional)
        samples, mean_samples, std_samples = model.predict(input_samples, num_samples=50)
    
    # Convert to numpy for plotting
    input_np = input_samples.cpu().numpy()
    output_np = output_samples.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    std_np = std_samples.cpu().numpy() if 'std_samples' in locals() else None
    
    # Plot results
    plot_comparisons(input_np, output_np, predictions_np, std_np , outpath)
    
    return input_np, output_np, predictions_np


def plot_comparisons(input, target, predictions, uncertainty=None,outpath='./'):
    """
    Plot input precipitation, actual precipitation, and predictions side by side
    """
    num_cases = len(input)
    
    fig, axes = plt.subplots(num_cases, 4 if uncertainty is not None else 3, 
                            figsize=(15, 3*num_cases))
    
    if num_cases == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_cases):
        # Input precipitation
        axes[i, 0].imshow(input[i, 0], cmap='Blues', aspect='auto')
        axes[i, 0].set_title(f'Case {i+1}: Input')
        axes[i, 0].axis('off')
        
        # Actual Precipitation
        axes[i, 1].imshow(target[i, 0], cmap='Blues', aspect='auto')
        axes[i, 1].set_title('Actual Precipitation')
        axes[i, 1].axis('off')
        
        # Predicted Precipitation
        im = axes[i, 2].imshow(predictions[i, 0], cmap='Blues', aspect='auto')
        axes[i, 2].set_title('Predicted Precipitation')
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046)
        
        # Uncertainty (if available)
        if uncertainty is not None:
            im_unc = axes[i, 3].imshow(uncertainty[i, 0], cmap='Reds', aspect='auto')
            axes[i, 3].set_title('Uncertainty (Std)')
            axes[i, 3].axis('off')
            plt.colorbar(im_unc, ax=axes[i, 3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(outpath + f'comparison_case_{i+1}.png', dpi=300, bbox_inches='tight')
    plt.show()
    


