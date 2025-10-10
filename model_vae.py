import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class ConvVAE_NonSquare(nn.Module):
    def __init__(self, input_channels=1, latent_dim=20, hidden_dims=[32, 64, 128, 256], 
                 input_height=28, input_width=28):
        super(ConvVAE_NonSquare, self).__init__()
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.input_width = input_width
        
        # Encoder
        encoder_layers = []
        in_channels = input_channels
        
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU()
            ])
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate the size after convolutions
        self.encoded_height, self.encoded_width, self.encoder_shapes = self._get_encoder_info(
            input_channels, input_height, input_width
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.encoded_height * self.encoded_width, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * self.encoded_height * self.encoded_width, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(
            latent_dim, hidden_dims[-1] * self.encoded_height * self.encoded_width
        )
        
        # Decoder with bilinear upsampling for non-square images
        self.decoder = self._build_decoder(hidden_dims, input_channels)
    
    def _get_encoder_info(self, input_channels, input_height, input_width):
        """Calculate the size after each encoder layer for both dimensions"""
        with torch.no_grad():
            x = torch.zeros(1, input_channels, input_height, input_width)
            heights = [input_height]
            widths = [input_width]
            shapes = [(input_height, input_width)]
            
            for layer in self.encoder:
                if isinstance(layer, nn.Conv2d):
                    x = layer(x)
                    h, w = x.shape[-2], x.shape[-1]
                    heights.append(h)
                    widths.append(w)
                    shapes.append((h, w))
                else:
                    x = layer(x)
            
            return x.shape[-2], x.shape[-1], shapes
    
    def _build_decoder(self, hidden_dims, input_channels):
        """Build decoder with proper size matching for non-square images"""
        decoder_layers = nn.Sequential()
        hidden_dims_rev = hidden_dims[::-1]
        
        # Reverse encoder shapes for decoder
        target_shapes = self.encoder_shapes[::-1]
        
        for i in range(len(hidden_dims_rev)):
            in_ch = hidden_dims_rev[i]
            
            # Determine output channels
            if i < len(hidden_dims_rev) - 1:
                out_ch = hidden_dims_rev[i + 1]
            else:
                out_ch = input_channels  # Final output to match input channels
            
            # Get target height and width for this layer
            target_height, target_width = target_shapes[i + 1]
            
            if i == len(hidden_dims_rev) - 1:
                # Final layer
                decoder_layers.add_module(f'final_block_{i}',
                    FinalUpsampleBlock(in_ch, out_ch, target_height, target_width)
                )
            else:
                # Intermediate layers
                decoder_layers.add_module(f'decoder_block_{i}',
                    DecoderBlock_NonSquare(in_ch, out_ch, target_height, target_width)
                )
        
        return decoder_layers
    
    def encode(self, x):
        """Encode input to latent parameters"""
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        
        return mu, logvar
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.encoded_height, self.encoded_width)
        result = self.decoder(result)
        return result
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class DecoderBlock_NonSquare(nn.Module):
    """Decoder block for non-square images with explicit height/width control"""
    def __init__(self, in_channels, out_channels, target_height, target_width):
        super(DecoderBlock_NonSquare, self).__init__()
        self.target_height = target_height
        self.target_width = target_width
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Use exact target dimensions for interpolation
        x = F.interpolate(x, size=(self.target_height, self.target_width), 
                         mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class FinalUpsampleBlock(nn.Module):
    """Final upsample block with output activation for non-square images"""
    def __init__(self, in_channels, out_channels, target_height, target_width):
        super(FinalUpsampleBlock, self).__init__()
        self.target_height = target_height
        self.target_width = target_width
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Final activation for pixel values
        )
    
    def forward(self, x):
        x = F.interpolate(x, size=(self.target_height, self.target_width), 
                         mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function"""
    # Reconstruction loss (binary cross entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss



def train_model(model, train_loader, epochs=50, learning_rate=1e-3, device='cuda'):
    """Train the VAE model"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch:03d} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {loss.item() / len(data):.6f}')
        
        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon = total_recon_loss / len(train_loader.dataset)
        avg_kl = total_kl_loss / len(train_loader.dataset)
        
        train_losses.append(avg_loss)
        
        print(f'====> Epoch {epoch:03d} Average loss: {avg_loss:.4f}, '
              f'Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}')
        
        # Test reconstruction size
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_batch = next(iter(train_loader))[0][:1].to(device)
                recon_test, _, _ = model(test_batch)
                print(f"Input size: {test_batch.shape}, Output size: {recon_test.shape}")
    
    return train_losses