import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import os
import loss_functions as lf



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
        use_batchnorm: bool = True,
        skip_connections = None,
    ):
        super(SkipConnectionVAE, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.input_width = input_width
        self.use_bilinear = use_bilinear
        self.use_batchnorm = use_batchnorm
        self.latent_dropout = nn.Dropout(p=0.5)
        self.feature_dropout = nn.Dropout(p=0.5)
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        if skip_connections is None :
            skip_connections = [1] * len( hidden_dims)
        
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.skip_connections = skip_connections
        
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
        decoder_feature_dims = self.encoder_features[::-1]  # Skip input, reverse
        #decoder_feature_dims.append(self.encoder_features[0])   # Add input at end for final layer
        print("Decoder feature dimensions:")
        for i, (ch, h, w) in enumerate(decoder_feature_dims):
            print(f"    Layer {i}: {ch} channels, {h}x{w}")
        
        # Build decoder blocks
        for i in range(len(hidden_dims_rev)):
            # Current decoder input channels
            in_ch = hidden_dims_rev[i]
            
            # Skip connection channels from corresponding encoder layer
            skip_ch = decoder_feature_dims[i+1][0]
            
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
        for ii , layer in enumerate( self.encoder ):
            current = layer(current)
            # Store output after each convolutional layer (but do not store the last output)
            if isinstance(layer, nn.Conv2d) :
                features.append(current)
        del features[-1]
        
        encoded_flat = current.view(current.size(0), -1)
        encoded_flat_dropped = self.feature_dropout(encoded_flat)
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
        skip_features = features[::-1]  # Skip input, reverse order
       
        #print(f"Decoding: starting with {h.shape}")
        #print(f"Skip features: {[f.shape for f in skip_features]}")
        
        # Forward through decoder blocks
        current = h
        for i, block in enumerate(self.decoder):
            skip = skip_features[i]
            #print(f"Block {i}: input {current.shape}, skip {skip.shape}")
            if self.skip_connections[i] == 1 :
               current = block(current, skip)
            else :
               current = block(current, None)
            #print(f"Block {i}: output {current.shape}")
        
        return current
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode and get features for skip connections
        mu, logvar, features = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # 2. Apply dropout to the sampled latent code (z) if in training mode
        if self.training:
           # Applies dropout directly to the sampled z vector
           z_decoded = self.latent_dropout(z)
        else:
           # No dropout during inference
           z_decoded = z
        
        # Decode with skip connections
        reconstruction = self.decode(z_decoded, features)
        
        return reconstruction, mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        device = mu.device
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        return mu + eps * std

    def sampler(self, num_samples , condition ) :
        """
        Sample precipitation patterns.
            
        Args:
            num_samples: Number of samples to generate
            condition: Iinput for conditional generation
        """
        # Encode the temperature condition to get the posterior distribution
        mu, logvar, features = self.encode(condition)

        # Sample from the posterior distribution: z ~ N(mu, sigma)
        # This gives samples conditioned on the temperature

        reconstruction = []
        for my_sample in range( num_samples ) :
            z = self.reparameterize(mu, logvar)
            # Apply dropout to z, if implemented, while in training mode
            if hasattr(self, 'latent_dropout') and self.training:
               z = self.latent_dropout(z)
            reconstruction.append( self.decode( z , features ) )

        return torch.stack( reconstruction )
    
    
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
        self.conv2_no_skip = nn.Sequential(
            nn.Conv2d(out_channels , out_channels, 3, padding=1),
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
            x = self.conv2(x)
        else :
           #Skip concatenation and use the small convolution.
           x = self.conv2_no_skip(x) # Input channel is just 'out_channels'
        
        
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
        # Then process without skip connection (original input)
        self.conv2_no_skip = nn.Sequential(
            nn.Conv2d(out_channels , out_channels, 3, padding=1),
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
        else  :
            x = self.conv2_no_skip( x )
        
        return x

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
    free_bits=0.01,
    use_lr_scheduler=True,
    early_stopping_patience=20,
    grad_clip=1.0,
    save_best=True,
    checkpoint_dir='checkpoints',
    weighted_loss_flag = False,
    LambdaProb = None ,
    LambdaProbMax = None ,
    LambdaProbMin = None , 
    LambdaProbTrend = None ,
    LambdaVal = None ,
    LambdaMinEpoch = None ,
    LambdaMaxEpoch = None ,
    LossName = None , 
    StochAnn = False ,
    CRPSNumSamples = 0 
):
    """
    Train the VAE model for temperature to precipitation regression.
    
    """

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0e-5)

    NLoss = len( LossName )
    #Get a list with the available loss functions.
    loss_list = lf.get_loss_list( LossName , device )

    # Loss weight and probability scheduler.
    if StochAnn : 
       loss_weight_scheduler = lf.StochasticAnnealingScheduler( LambdaProb , LambdaProbMax = LambdaProbMax , LambdaProbMin = LambdaProbMin , LambdaProbTrend = LambdaProbTrend , LambdaVal = LambdaVal , LambdaMinEpoch = LambdaMinEpoch )

    #Create objects to monitor loss values
    HistTrainLoss = lf.LossHistory( LossName )
    HistValLoss   = lf.LossHistory( LossName )
    
    # Learning rate scheduler
    if use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=20, factor=0.5)
    
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # Create checkpoint directory
    if save_best and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # === Training Phase ===
        model.train()
        
        for batch_idx, (gfs_precipitation, precipitation) in enumerate(train_loader):
            gfs_precipitation = gfs_precipitation.to(device)
            precipitation = precipitation.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_precip, mu, logvar = model( gfs_precipitation )

            if StochAnn :
               loss_weight =loss_weight_scheduler.get_weights( epoch ).to(device)
            else        :
               loss_weight = LambdaVal.to(device)

            # Forward pass
            recon_precip, mu, logvar = model( gfs_precipitation )
            # If CRPS loss is active, then generate a sample of outputs.
            for ii , my_loss in enumerate( LossName ) :
                if my_loss == 'CRPS' and loss_weight[ii] > 0.0 :
                   recon_precip_samples = model.sampler( CRPSNumSamples , gfs_precipitation )
                else :
                   recon_precip_samples = None

            #Compute loss
            loss_vec = torch.ones( NLoss )
            loss_vec = lf.compute_loss( loss_list , recon_precip , precipitation , LossName , loss_weight , mu=mu , logvar=logvar , output_samples = recon_precip_samples , free_bits = free_bits )
            total_loss = loss_vec.sum()

            #Compute gradient
            total_loss.backward()
            
            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_( model.parameters() , grad_clip )
            optimizer.step()
            
            # Accumulate losses
            HistTrainLoss.loss_add( loss_vec , loss_weight )
            
            # Print batch progress
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch:03d} | Batch {batch_idx:03d}/{len(train_loader)} | '
                      f'Loss: {total_loss.item():.4f}')
        
        # Store the average loss for this epoch
        HistTrainLoss.loss_epoch()
        
        # === Validation Phase ===
        if val_loader is not None:
            model.eval()
            
            with torch.no_grad():
                for gfs_precipitation , precipitation in val_loader:
                    gfs_precipitation = gfs_precipitation.to(device)
                    precipitation = precipitation.to(device)
                    
                    recon_precip, mu, logvar = model(gfs_precipitation)

                    #We do not use stochastic annealing during validation
                    loss_weight = LambdaVal

                    # Forward pass
                    recon_precip, mu, logvar = model( gfs_precipitation )
                    # If CRPS loss is active, then generate a sample of outputs.
                    for ii , my_loss in enumerate( LossName ) :
                       if my_loss == 'CRPS' and loss_weight[ii] > 0.0 :
                          recon_precip_samples = model.sampler( CRPSNumSamples , gfs_precipitation )
                       else :
                          recon_precip_samples = None

                    #Compute loss
                    loss_vec = lf.compute_loss( loss_list , recon_precip , precipitation , LossName , loss_weight , mu=mu , output_samples = recon_precip_samples , logvar=logvar )
                    # Accumulate losses
                    HistValLoss.loss_add( loss_vec , loss_weight )

            # Store the average loss for this epoch
            HistValLoss.loss_epoch()
            
            # Learning rate scheduling
            if use_lr_scheduler:
                scheduler.step( HistValLoss.history['TotalLoss'][-1] )
            
            # Early stopping and model checkpointing
            if HistValLoss.history['TotalLoss'][-1] < best_val_loss:
                best_val_loss = HistValLoss.history['TotalLoss'][-1]
                early_stopping_counter = 0
                
                if save_best:
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': HistTrainLoss.history['TotalLoss'][-1],
                        'val_loss': HistValLoss.history['TotalLoss'][-1],
                        'historyval': HistValLoss.history,
                        'historytrain': HistTrainLoss.history,
                        'config': {
                            'learning_rate': learning_rate,
                            'hidden_dims': model.hidden_dims,
                            'latent_dim': model.latent_dim
                        }
                    }, checkpoint_path)
                    print(f'✅Saved best model')
            else:
                early_stopping_counter += 1
        
        # === Print Epoch Summary ===
        print(f'Epoch {epoch:03d}/{epochs} Summary:')
        print('Training loss')
        HistTrainLoss.print_loss()
        if val_loader is not None:
            print('Validation Loss')
            HistValLoss.print_loss()
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}, Early stopping: {early_stopping_counter}/{early_stopping_patience}')
        
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
            'train_loss': HistTrainLoss.history['TotalLoss'][-1],
            'val_loss': HistValLoss.history['TotalLoss'][-1],
            'historyval': HistValLoss.history,
            'historytrain': HistTrainLoss.history,
            'config': {
                'learning_rate': learning_rate,
                'hidden_dims': model.hidden_dims,
                'latent_dim': model.latent_dim
            }
        }, final_checkpoint_path)
        print(f'  ✅ Saved final model')
    
    return #history, best_val_loss

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
    


