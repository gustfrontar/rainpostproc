import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.fft as fft
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


def vae_regression_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
                       logvar: torch.Tensor, free_bits: float = 0.01 ,
                       lambda_mse: float = 0.0 , lambda_kl: float = 0.0 , lambda_var: float = 0.0 , lambda_skew: float = 0.0 , lambda_kurt: float = 0.0 ,
                       weigthed_loss_flag: float = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined loss for regression VAE"""
    # Reconstruction loss (MSE for regression)
    recon_x = recon_x.squeeze()
    x = x.squeeze()
    
    
    mse_loss = loss_mse(recon_x, x, weigthed_loss_flag = weigthed_loss_flag)
    
    # KL divergence
    kl_loss = free_bits_kl_loss(mu, logvar, free_bits=free_bits)
    #kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    var_loss = loss_var(recon_x, x)
    skew_loss = loss_skew(recon_x, x)
    kurt_loss = loss_kurt(recon_x, x)
    # Combine losses
    
    total_loss = lambda_mse * mse_loss + lambda_kl * kl_loss + lambda_var * var_loss + lambda_skew * skew_loss + lambda_kurt * kurt_loss
    
    return total_loss, lambda_mse * mse_loss, lambda_kl * kl_loss , var_loss * lambda_var , skew_loss * lambda_skew , kurt_loss * lambda_kurt

def free_bits_kl_loss(mu, logvar, free_bits=2.0):
    # KL per dimension: -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # Free bits: each dimension must have KL > free_bits
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    
    # Sum over dimensions, average over batch
    kl_loss = torch.mean(torch.sum(kl_per_dim, dim=1))
    return kl_loss

def loss_mse(output, target , weigthed_loss_flag = False) :
    output = output.squeeze()
    target = target.squeeze()   
    
    if weigthed_loss_flag:
        eps = 1e-6
        #Weighted MSE loss giving more weight to high precipitation values.
        return torch.sum( (target+eps) * (output - target)**2)
    else:
        #Standard MSE loss  
        return torch.sum((output - target)**2)

def loss_var(output, target) :
    output = output.squeeze()
    target = target.squeeze()
    #Compute the variance of the output and target, then return the MSE between them. (image-wise)
    #nb = output.shape[0]
    varo = torch.var(output, dim=(1,2))
    vart = torch.var(target, dim=(1,2))
    return torch.mean((varo - vart)**2)

def loss_skew(output, target) :
    output = output.squeeze()
    target = target.squeeze()    
    #Compute the skewness of the output and target, then return the MSE between them. (image-wise)
    #nb = output.shape[0]
    eps = 1e-6
    meano = torch.mean(output, dim=(1,2))
    meant = torch.mean(target, dim=(1,2))
    stdo = torch.std(output, dim=(1,2)) + eps
    stdt = torch.std(target, dim=(1,2)) + eps
    
    skwo = torch.mean( ((output - meano[:,None,None])/stdo[:,None,None] )**3 , dim=(1,2) )
    skwt = torch.mean( ((target - meant[:,None,None])/stdt[:,None,None] )**3 , dim=(1,2) )
    return torch.mean((skwo - skwt)**2)

def loss_kurt(output, target) :
    output = output.squeeze()
    target = target.squeeze()    
    #Compute the kurtosis of the output and target, then return the MSE between them. (image-wise)
    #nb = output.shape[0]
    eps = 1e-6
    meano = torch.mean(output, dim=(1,2))
    meant = torch.mean(target, dim=(1,2))
    stdo = torch.std(output, dim=(1,2)) + eps
    stdt = torch.std(target, dim=(1,2)) + eps
    kurto = torch.mean( ((output - meano[:,None,None])/stdo[:,None,None] )**4 , dim=(1,2) )
    kurt = torch.mean( ((target - meant[:,None,None])/stdt[:,None,None] )**4 , dim=(1,2) )
    return torch.mean((kurto - kurt)**2)


class RandomFourierLoss(nn.Module):
    "Fourier Amplitude and Correlation Loss: Beyond Using L2 Loss for Skillful Precipitation Nowcasting by Yan et al. 2024"  
    "https://arxiv.org/abs/2410.23159"
    def __init__(self, min_prob = 0.0 , max_prob = 1.0 , prob_slope = 1.0e-4 ):
        super(RandomFourierLoss, self).__init__()
        self.step = 0
        self.out = 0
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.prob_slope = prob_slope


    def fcl(self, fft_pred, fft_truth):
        
        # In general, FFTs here must be shifted to the center; but here we use the whole fourier space, so it is okay to no need have fourier shift operation
        conj_pred = torch.conj(fft_pred)
        numerator = (conj_pred*fft_truth).sum().real
        denominator = torch.sqrt(((fft_truth).abs()**2).sum()*((fft_pred).abs()**2).sum())
        return 1. - numerator/denominator

    def fal(self, fft_pred, fft_truth):

        #return nn.MSELoss()(fft_pred.abs(), fft_truth.abs())
        return (( fft_pred.abs() - fft_truth.abs() )**(2)).mean()
    def forward(self, pred, gt , mode='train' ):
        pred=pred.squeeze()
        gt=gt.squeeze()
        fft_pred = torch.fft.fftn(pred, dim=[-1,-2] , norm='ortho')
        fft_gt = torch.fft.fftn(gt, dim=[-1,-2]    , norm='ortho')
        #print(pred.shape,fft_pred.shape,fft_gt.shape   )
        

        prob_t = ( self.step * self.prob_slope ) * ( self.max_prob - self.min_prob ) + self.min_prob
        if prob_t > self.max_prob :
            prob_t = self.max_prob 
        prob = 1.0 if np.random.rand() < prob_t else 0.0
        if mode == 'val' : prob = 0.5  #For validation use equal weights
        #print(prob_t)
        H, W = pred.shape[-2:]
        weight = np.sqrt(H*W)
        loss = (prob)*self.fal(fft_pred, fft_gt) + (1-prob) * self.fcl(fft_pred, fft_gt)
        #print( prob , self.fal(fft_pred, fft_gt).item() , self.fcl(fft_pred, fft_gt).item()    )
        loss = loss*weight
        if mode == 'train' : self.step += 1
        return loss


class FourierLoss(nn.Module):
    """
    Computes the Mean Squared Error (MSE) in the Fourier domain (Amplitude Spectrum).
    Optionally applies a frequency-based weight mask.
    """
    def __init__(self, mode: str = 'MSE', weight_mode: Optional[str] = 'log_distance', alpha: float = 1.0):
        """
        Args:
            mode: The distance metric to use ('MSE' or 'L1').
            weight_mode: 'none', 'distance', or 'log_distance'. 
                         'distance' weights higher frequencies more heavily.
            alpha: Weighting factor for the frequency mask.
        """
        super(FourierLoss, self).__init__()
        self.mode = mode
        self.weight_mode = weight_mode
        self.alpha = alpha
        # Buffer to store the pre-calculated weight mask
        self.register_buffer('weight_mask', None)

    def _get_weight_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        Generates a 2D mask based on the distance from the DC component (center/low frequency).
        High frequencies get larger weights.
        """
        if self.weight_mask is not None and self.weight_mask.shape[-2:] == (h, w):
            return self.weight_mask.to(device)

        # Create coordinate grids
        center_h, center_w = h // 2, w // 2
        
        # Frequency indices range from 0 to N-1
        y_indices = torch.arange(h, device=device) - center_h
        x_indices = torch.arange(w, device=device) - center_w

        # Use meshgrid for 2D indices
        Y, X = torch.meshgrid(y_indices, x_indices, indexing='ij')

        # Calculate Euclidean distance from the origin (DC component)
        # Note: fft.fftshift is used implicitly later, so the distance from the origin (0,0) is correct
        distance = torch.sqrt(X**2 + Y**2)
        
        # Shift the distance map so the zero-frequency component is at the corner (0,0)
        # matching the output format of standard torch.fft (no shift applied)
        distance = fft.ifftshift(distance) 

        if self.weight_mode == 'distance':
            # Linear weighting by distance (weights high frequencies)
            mask = 1 + self.alpha * distance
        elif self.weight_mode == 'log_distance':
            # Logarithmic weighting (smoother transition)
            mask = 1 + self.alpha * torch.log1p(distance)
        else: # 'none'
            mask = torch.ones((h, w), device=device)
        
        # Store mask as a buffer
        self.weight_mask = mask.detach()
        return self.weight_mask

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted image tensor [B, C, H, W].
            target: Target image tensor [B, C, H, W].
        
        Returns:
            Weighted Fourier Loss (scalar).
        """
        pred = pred.squeeze()
        target = target.squeeze()
        
        b, c, h, w = pred.shape
        device = pred.device

        # 1. Compute 2D FFT (operates independently on HxW planes)
        # Output is complex: [B, C, H, W, 2] or [B, C, H, W] (complex dtype)
        f_pred = fft.fft2(pred, dim=(-2, -1))
        f_target = fft.fft2(target, dim=(-2, -1))

        # 2. Extract Amplitude Spectrum (|A| = sqrt(real^2 + imag^2))
        amp_pred = torch.abs(f_pred)
        amp_target = torch.abs(f_target)
        
        # 3. Calculate Loss in Frequency Domain
        if self.mode == 'MSE':
            loss = (amp_pred - amp_target).pow(2)
        elif self.mode == 'L1':
            loss = torch.abs(amp_pred - amp_target)
        else:
            raise ValueError(f"Mode '{self.mode}' not supported. Use 'MSE' or 'L1'.")

        # 4. Apply Weighting Mask
        if self.weight_mode != 'none':
            # Get and expand mask to match [B, C, H, W] shape
            mask = self._get_weight_mask(h, w, device).unsqueeze(0).unsqueeze(0).expand(b, c, h, w)
            loss = loss * mask
        
        # 5. Return mean loss over all batch elements, channels, and frequencies
        return torch.mean(loss)


def lambda_kl_scheduler(epoch , minibatch , max_lambda_kl=0.1 , free_bits=1.0):
    if epoch > 10 and ( epoch % 5 )  == 0  and (minibatch % 50 ) == 0 :
       return max_lambda_kl , free_bits 
    else :
       return 0.0 , 0.0

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
    max_lambda_kl = 0.0 ,
    lambda_mse = 0.0 ,
    lambda_var = 0.0 ,
    lambda_skew = 0.0 ,
    lambda_kurt = 0.0 ,
    lambda_fourier = 0.0 ,
    lambda_random_fourier = 0.0 ,
    weigthed_loss_flag = False,
    fourier_loss_flag = False ,
    random_fourier_loss_flag = False ,
    random_fourier_loss_min_prob = 0.4 ,
    random_fourier_loss_max_prob = 0.4 , 
    random_fourier_loss_prob_slope = 1.0e-4,
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
        lambda_kl: Weight for KL divergence loss
        use_lr_scheduler: Whether to use learning rate scheduling
        early_stopping_patience: Early stopping patience
        grad_clip: Gradient clipping value
        save_best: Whether to save best model
        checkpoint_dir: Directory to save checkpoints
    """

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0e-5)
    
    if fourier_loss_flag:
        print("Using Fourier Loss for reconstruction")
        fourier_loss_fn = FourierLoss(mode='MSE', weight_mode='log_distance', alpha=1.0).to(device)

    if random_fourier_loss_flag :
        num_mini_batch = len( train_loader )
        print(num_mini_batch)
        total_step = num_mini_batch * epochs  #Maximum number of weigth updates
        print("Using Random Fourier Loss for reconstruction")
        random_fourier_loss_fn = RandomFourierLoss( min_prob = random_fourier_loss_min_prob , max_prob = random_fourier_loss_max_prob , prob_slope = random_fourier_loss_prob_slope ).to(device)
    
    
    # Learning rate scheduler
    if use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=20, factor=0.5)
    
    # Training history
    history = {
        'train_total_loss': [], 'train_recon_loss': [], 'train_kl_loss': [],
        'train_var_loss': [], 'train_skew_loss': [], 'train_kurt_loss': [],
        'train_fourier_loss': [], 'train_rdn_fourier_loss' : [] ,
        'val_total_loss': [], 'val_recon_loss': [], 'val_kl_loss': [],
        'val_var_loss': [], 'val_skew_loss': [], 'val_kurt_loss': [],
        'val_fourier_loss': [], 'val_rdn_fourier_loss' : [] ,
        'learning_rates': [] 
    }
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # Create checkpoint directory
    if save_best and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Device: {device}, KL weight: {max_lambda_kl}")
    
    for epoch in range(epochs):
        # === Training Phase ===
        model.train()
        train_total, train_recon, train_kl , train_var , train_skew , train_kurt , train_fourier , train_rdn_fourier = 0.0, 0.0, 0.0 , 0.0 , 0.0 , 0.0, 0.0 , 0.0
        num_train_batches = 0
        
        for batch_idx, (temperature, precipitation) in enumerate(train_loader):
            temperature = temperature.to(device)
            precipitation = precipitation.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_precip, mu, logvar = model(temperature)
            
            
            lambda_kl , free_bits = lambda_kl_scheduler(epoch , batch_idx , max_lambda_kl , free_bits)

            total_loss, recon_loss, kl_loss , var_loss , skew_loss , kurt_loss = vae_regression_loss(
                recon_precip, precipitation, mu, logvar, lambda_kl = lambda_kl , free_bits=free_bits ,
                lambda_mse = lambda_mse , lambda_var=lambda_var , lambda_skew=lambda_skew , lambda_kurt=lambda_kurt ,
                weigthed_loss_flag=weigthed_loss_flag
            )
            # Add Fourier loss if specified
            if fourier_loss_flag:
                fourier_loss = fourier_loss_fn(recon_precip, precipitation)
                total_loss += fourier_loss * lambda_fourier
            if random_fourier_loss_flag :
                rdn_fourier_loss = random_fourier_loss_fn(recon_precip , precipitation)
                total_loss += rdn_fourier_loss * lambda_random_fourier
            
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
            train_var += var_loss.item()
            train_skew += skew_loss.item()
            train_kurt += kurt_loss.item()
            train_fourier += fourier_loss.item() if fourier_loss_flag else 0.0
            train_rdn_fourier += rdn_fourier_loss.item() if random_fourier_loss_flag else 0.0
            num_train_batches += 1
            
            # Print batch progress
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch:03d} | Batch {batch_idx:03d}/{len(train_loader)} | '
                      f'Loss: {total_loss.item()/temperature.size(0):.4f}')
        
        # Calculate average training losses
        avg_train_total = train_total / len(train_loader.dataset)
        avg_train_recon = train_recon / len(train_loader.dataset)
        avg_train_kl = train_kl / len(train_loader.dataset)
        avg_train_var = train_var / len(train_loader.dataset)
        avg_train_skew = train_skew / len(train_loader.dataset)
        avg_train_kurt = train_kurt / len(train_loader.dataset)
        avg_fourier_loss = train_fourier / len(train_loader.dataset) if fourier_loss_flag else 0.0
        avg_rdn_fourier_loss = train_rdn_fourier / len( train_loader.dataset) if random_fourier_loss_flag else 0.0
        
        
        history['train_total_loss'].append(avg_train_total)
        history['train_recon_loss'].append(avg_train_recon)
        history['train_kl_loss'].append(avg_train_kl)
        history['train_var_loss'].append(avg_train_var)
        history['train_skew_loss'].append(avg_train_skew)
        history['train_kurt_loss'].append(avg_train_kurt)
        history['train_fourier_loss'].append(avg_fourier_loss)
        history['train_rdn_fourier_loss'].append(avg_rdn_fourier_loss)

        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # === Validation Phase ===
        if val_loader is not None:
            model.eval()
            val_total, val_recon, val_kl , val_var , val_skew , val_kurt , val_fourier , val_rdn_fourier = 0.0, 0.0, 0.0 , 0.0 , 0.0 , 0.0, 0.0 , 0.0
            
            with torch.no_grad():
                for temperature, precipitation in val_loader:
                    temperature = temperature.to(device)
                    precipitation = precipitation.to(device)
                    
                    recon_precip, mu, logvar = model(temperature)
                    total_loss, recon_loss, kl_loss , var_loss , skew_loss , kurt_loss = vae_regression_loss(
                        recon_precip, precipitation, mu, logvar, lambda_kl = lambda_kl , free_bits=free_bits ,
                        lambda_mse = lambda_mse , lambda_var=lambda_var , lambda_skew=lambda_skew , lambda_kurt=lambda_kurt ,
                        weigthed_loss_flag=weigthed_loss_flag
                    )
                    if fourier_loss_flag :
                        fourier_loss = fourier_loss_fn(recon_precip, precipitation)
                        total_loss += fourier_loss * lambda_fourier

                    if random_fourier_loss_flag :
                        rdn_fourier_loss = random_fourier_loss_fn(recon_precip,precipitation,mode='val')
                        total_loss += rdn_fourier_loss * lambda_random_fourier
                    
                    
                    val_total += total_loss.item()
                    val_recon += recon_loss.item()
                    val_kl += kl_loss.item()
                    val_var += var_loss.item()
                    val_skew += skew_loss.item()
                    val_kurt += kurt_loss.item()
                    val_fourier += fourier_loss.item() if fourier_loss_flag else 0.0
                    val_rdn_fourier += rdn_fourier_loss.item() if random_fourier_loss_flag else 0.0 
            
            avg_val_total = val_total / len(val_loader.dataset)
            avg_val_recon = val_recon / len(val_loader.dataset)
            avg_val_kl = val_kl / len(val_loader.dataset)
            avg_val_var = val_var / len(val_loader.dataset)
            avg_val_skew = val_skew / len(val_loader.dataset)
            avg_val_kurt = val_kurt / len(val_loader.dataset)
            avg_val_fourier = val_fourier / len(val_loader.dataset) if fourier_loss_flag else 0.0
            avg_val_rdn_fourier = val_rdn_fourier / len(val_loader.dataset) if random_fourier_loss_flag else 0.0 
            
            
            history['val_total_loss'].append(avg_val_total)
            history['val_recon_loss'].append(avg_val_recon)
            history['val_kl_loss'].append(avg_val_kl)
            history['val_var_loss'].append(avg_val_var)
            history['val_skew_loss'].append(avg_val_skew)
            history['val_kurt_loss'].append(avg_val_kurt)
            history['val_fourier_loss'].append(avg_val_fourier)
            history['val_rdn_fourier_loss'].append(avg_val_rdn_fourier)
            
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
                            'lambda_kl': lambda_kl,
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
        print(f'  Train - Total: {avg_train_total:.4f}, Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f} , Var: {avg_train_var:.4f} , Skew: {avg_train_skew:.4f} , Kurt: {avg_train_kurt:.4f} , Fourier: {avg_fourier_loss:.4f} , RdnFourier: {avg_rdn_fourier_loss:.4f}')
        
        if val_loader is not None:
            print(f'  Val   - Total: {avg_val_total:.4f}, Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f} , Var: {avg_val_var:.4f} , Skew: {avg_val_skew:.4f} , Kurt: {avg_val_kurt:.4f} , Fourier: {avg_val_fourier:.4f}, RdnFourier: {avg_val_rdn_fourier:.4f} ')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}, Early stopping: {early_stopping_counter}/{early_stopping_patience}')
        
        # Check for KL collapse or explosion
        if avg_train_kl < 1.0:
            print('  ⚠️  Warning: KL loss very low - possible posterior collapse')
        elif avg_train_kl > 1000:
            print('  ⚠️  Warning: KL loss very high - check lambda_kl value')
        
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
                'lambda_kl': lambda_kl,
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
        min_val = np.min([np.min(input[i, 0]),np.min(target[i, 0]),np.min(predictions[i, 0])])
        max_val = np.max([np.max(input[i, 0]),np.max(target[i, 0]),np.max(predictions[i, 0])])
        axes[i, 0].imshow(input[i, 0], cmap='Blues', aspect='auto',vmin=min_val ,vmax=max_val)
        axes[i, 0].set_title(f'Case {i+1}: Input')
        axes[i, 0].axis('off')
        
        # Actual Precipitation
        axes[i, 1].imshow(target[i, 0], cmap='Blues', aspect='auto',vmin=min_val,vmax=max_val)
        axes[i, 1].set_title('Actual Precipitation')
        axes[i, 1].axis('off')
        
        # Predicted Precipitation
        im = axes[i, 2].imshow(predictions[i, 0], cmap='Blues', aspect='auto',vmin=min_val,vmax=max_val)
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
    


