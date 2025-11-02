import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.fft as fft
import numpy as np

def loss_kl(mu, logvar, free_bits=2.0):
    # KL per dimension: -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Free bits: each dimension must have KL > free_bits
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    # Sum over dimensions, average over batch
    loss = torch.mean(torch.sum(kl_per_dim, dim=1))
    return loss

def loss_mse(output, target , weighted_loss_flag = False) :
    output = output.squeeze()
    target = target.squeeze()

    if weighted_loss_flag:
        eps = 1e-6
        #Weighted MSE loss giving more weight to high precipitation values.
        return torch.mean( (target+eps) * (output - target)**2)
    else:
        #Standard MSE loss
        return torch.mean((output - target)**2)

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

class loss_FAL(nn.Module):
    "Fourier Amplitude Loss: Beyond Using L2 Loss for Skillful Precipitation Nowcasting by Yan et al. 2024"
    "https://arxiv.org/abs/2410.23159"
    def __init__( self ):
        super( loss_FAL , self).__init__()

    def fal(self, fft_pred, fft_truth):

        #return nn.MSELoss()(fft_pred.abs(), fft_truth.abs())
        return (( fft_pred.abs() - fft_truth.abs() )**(2)).mean()
    def forward(self , pred, gt ):
        pred=pred.squeeze()
        gt=gt.squeeze()
        fft_pred = torch.fft.fftn(pred, dim=[-1,-2] , norm='ortho')
        fft_gt = torch.fft.fftn(gt, dim=[-1,-2]    , norm='ortho')
        #print(pred.shape,fft_pred.shape,fft_gt.shape   )
        H, W = pred.shape[-2:]
        weight = np.sqrt(H*W)
        loss = self.fal(fft_pred, fft_gt)
        return loss


class loss_FCM(nn.Module):
    "Fourier Correlation Loss: Beyond Using L2 Loss for Skillful Precipitation Nowcasting by Yan et al. 2024"
    "https://arxiv.org/abs/2410.23159"
    def __init__( self ):
        super( loss_FCM , self ).__init__()

    def fcl(self, fft_pred, fft_truth):

        # In general, FFTs here must be shifted to the center; but here we use the whole fourier space, so it is okay to no need have fourier shift operation
        conj_pred = torch.conj(fft_pred)
        numerator = (conj_pred*fft_truth).sum().real
        denominator = torch.sqrt(((fft_truth).abs()**2).sum()*((fft_pred).abs()**2).sum())
        return 1. - numerator/denominator

    def forward(self, pred, gt ):
        pred=pred.squeeze()
        gt=gt.squeeze()
        fft_pred = torch.fft.fftn(pred, dim=[-1,-2] , norm='ortho')
        fft_gt = torch.fft.fftn(gt, dim=[-1,-2]    , norm='ortho')
        #print(prob_t)
        H, W = pred.shape[-2:]
        weight = np.sqrt(H*W)
        loss = self.fcl(fft_pred, fft_gt)
        return loss


class loss_Fourier(nn.Module):
    """
    Computes the Mean Squared Error (MSE) in the Fourier domain (Amplitude Spectrum).
    Optionally applies a frequency-based weight mask.
    """
    def __init__(self, mode = 'MSE', weight_mode = 'log_distance', alpha = 1.0):
        """
        Args:
            mode: The distance metric to use ('MSE' or 'L1').
            weight_mode: 'none', 'distance', or 'log_distance'. 
                         'distance' weights higher frequencies more heavily.
            alpha: Weighting factor for the frequency mask.
        """
        super( loss_Fourier , self).__init__()
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

class loss_CRPS(nn.Module):
    """
    Calculates the averaged Empirical Continuous Ranked Probability Score (CRPS)
    over all pixels, based on a set of generated samples.
    """
    def __init__(self):
        super( loss_CRPS , self).__init__()

    def forward(self, samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            samples: Tensor of generated predictions. Shape: [N_samples, B, C, H, W]
            target: Ground truth tensor. Shape: [B, C, H, W]

        Returns:
            Scalar CRPS loss value.
        """

        # 1. Expand target to match the sample count for element-wise calculations
        # Target shape: [1, B, C, H, W]
        target_expanded = target.unsqueeze(0)

        # Target samples are identical across the sample dimension (N_samples)
        # Target shape now: [N_samples, B, C, H, W]
        N = samples.size(0)
        target_expanded = target_expanded.expand_as(samples)

        # --- CRPS Terms ---

        # 2. Term 1: Accuracy and Sharpness (Mean Absolute Error, MAE)
        # Calculates |y_hat_i - y| for all i and j
        # Shape: [N_samples, B, C, H, W]
        term1_abs_diff = torch.abs(samples - target_expanded)
        # Average over the sample dimension (N_samples)
        term1 = torch.mean(term1_abs_diff, dim=0)
       # 3. Term 2: Spread (Diversity)
        # Calculates |y_hat_i - y_hat_j| for all i and j pairs
        # We need to broadcast samples [N, B, ...] to [N, N, B, ...] for subtraction

        # samples_i shape: [N, 1, B, C, H, W]
        samples_i = samples.unsqueeze(1)
        # samples_j shape: [1, N, B, C, H, W]
        samples_j = samples.unsqueeze(0)

        # Calculates |y_hat_i - y_hat_j| for all pairs (i, j)
        # Shape: [N, N, B, C, H, W]
        term2_abs_diff = torch.abs(samples_i - samples_j)

        # Sum/Mean over i and j, then divide by 2 (as per the empirical formula's 1/(2N^2) summation)
        # torch.mean over dim 0 and 1 performs the summation and division by N^2 implicitly.
        term2 = torch.mean(term2_abs_diff, dim=(0, 1)) / 2.0

        # 4. Final CRPS (pixel-wise)
        # Shape: [B, C, H, W]
        crps_map = term1 - term2

        # 5. Return the average CRPS over all pixels, channels, and batches
        return torch.mean(crps_map)

#def vae_regression_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
#                       logvar: torch.Tensor, free_bits: float = 0.01 ,
#                       lambda_mse: float = 0.0 , lambda_kl: float = 0.0 , lambda_var: float = 0.0 , lambda_skew: float = 0.0 , lambda_kurt: float = 0.0 ,
#                       weighted_loss_flag: float = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#    """Combined loss for regression VAE"""
#    # Reconstruction loss (MSE for regression)
#    recon_x = recon_x.squeeze()
#    x = x.squeeze()


#    mse_loss = loss_mse(recon_x, x, weighted_loss_flag = weighted_loss_flag)

#    # KL divergence
#    kl_loss = free_bits_kl_loss(mu, logvar, free_bits=free_bits)
#    #kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#    var_loss = loss_var(recon_x, x)
#    skew_loss = loss_skew(recon_x, x)
#    kurt_loss = loss_kurt(recon_x, x)
    # Combine losses
#    total_loss = lambda_mse * mse_loss + lambda_kl * kl_loss + lambda_var * var_loss + lambda_skew * skew_loss + lambda_kurt * kurt_loss

#    return total_loss, lambda_mse * mse_loss, lambda_kl * kl_loss , var_loss * lambda_var , skew_loss * lambda_skew , kurt_loss * lambda_kurt



#class RandomFourierLoss(nn.Module):
#    "Fourier Amplitude and Correlation Loss: Beyond Using L2 Loss for Skillful Precipitation Nowcasting by Yan et al. 2024"
#    "https://arxiv.org/abs/2410.23159"
#    def __init__(self, min_prob = 0.0 , max_prob = 1.0 , prob_slope = 1.0e-4 ):
#        super(RandomFourierLoss, self).__init__()
#        self.step = 0
#        self.out = 0
#        self.min_prob = min_prob
#        self.max_prob = max_prob
#        self.prob_slope = prob_slope


#    def fcl(self, fft_pred, fft_truth):

#        # In general, FFTs here must be shifted to the center; but here we use the whole fourier space, so it is okay to no need have fourier shift operation
#        conj_pred = torch.conj(fft_pred)
#        numerator = (conj_pred*fft_truth).sum().real
#        denominator = torch.sqrt(((fft_truth).abs()**2).sum()*((fft_pred).abs()**2).sum())
#        return 1. - numerator/denominator

#    def fal(self, fft_pred, fft_truth):

#        #return nn.MSELoss()(fft_pred.abs(), fft_truth.abs())
#        return (( fft_pred.abs() - fft_truth.abs() )**(2)).mean()
#    def forward(self, pred, gt , mode='train' ):
#        pred=pred.squeeze()
#        gt=gt.squeeze()
#        fft_pred = torch.fft.fftn(pred, dim=[-1,-2] , norm='ortho')
#        fft_gt = torch.fft.fftn(gt, dim=[-1,-2]    , norm='ortho')
#        #print(pred.shape,fft_pred.shape,fft_gt.shape   )


#        prob_t = ( self.step * self.prob_slope ) * ( self.max_prob - self.min_prob ) + self.min_prob
#        if prob_t > self.max_prob :
#            prob_t = self.max_prob
#        prob = 1.0 if np.random.rand() < prob_t else 0.0
#        if mode == 'val' : prob = 0.5  #For validation use equal weights
#        #print(prob_t)
#        H, W = pred.shape[-2:]
#        weight = np.sqrt(H*W)
#        loss = (prob)*self.fal(fft_pred, fft_gt) + (1-prob) * self.fcl(fft_pred, fft_gt)
#        #print( prob , self.fal(fft_pred, fft_gt).item() , self.fcl(fft_pred, fft_gt).item()    )
#        loss = loss*weight
#        if mode == 'train' : self.step += 1
#        return loss

class LossHistory() :
    def __init__( self , LossName ) :
        super( LossHistory , self).__init__()
        self.LossName = LossName
        self.LossValue = np.zeros( len( self.LossName ) )
        self.LossCount = np.ones( len( self.LossName ) )
        self.history = dict()
        self.history['LossName'] = LossName
        self.history['Loss'] = []
        self.history['TotalLoss'] = []

    def loss_add( self , loss_vec , loss_w ) :
        for ii in range( len(loss_vec) ) :
           if loss_w[ii] > 0.0 :
              self.LossValue[ii] += loss_vec[ii].item()
              self.LossCount[ii] += 1
    def loss_epoch( self )  :
        self.history['Loss'].append( np.copy( self.LossValue / self.LossCount ) )
        self.history['TotalLoss'].append( np.copy( self.history['Loss'][-1].sum() ) )
        self.loss_reset()
    def loss_reset( self )  :
        self.LossValue = np.zeros( len( self.LossName ) )
        self.LossCount = np.ones( len( self.LossName ) )
    def print_loss( self ) :
        for ii , my_loss in enumerate( self.history['LossName'] ) :
            print( my_loss + ': ' + str( self.history['Loss'][-1][ii] ) )
        print('Total Loss : ' + str( self.history['TotalLoss'][-1] ) )


class StochasticAnnealingScheduler() :
    def __init__(self, LambdaProb , LambdaProbMax = None , LambdaProbMin = None , LambdaProbTrend = None , LambdaVal = None , LambdaMinEpoch = None , LambdaMaxEpoch = None ) :
        super( StochasticAnnealingScheduler , self).__init__()

        self.LambdaProb = np.array( LambdaProb )
        if LambdaProbMax is not None :
           self.LambdaProbMax = np.array( LambdaProbMax )
        else :
           self.LambdaProbMax = np.ones( LambdaProb.shape )
        if LambdaProbMin is not None :
           self.LambdaProbMin = np.array( LambdaProbMin )
        else :
           self.LambdaProbMin = np.zeros( LambdaProb.shape )
        if LambdaProbTrend is not None :
           self.LambdaProbTrend =  np.array( LambdaProbTrend )
        else :
           self.LambdaProbTrend = np.zeros( LambdaProb.shape )
        if LambdaVal is not None :
           self.LambdaVal = np.array( LambdaVal )
        else :
           LambdaVal = np.ones( LambdaProb.shape )
        if LambdaMinEpoch is not None :
           self.LambdaMinEpoch = np.array( LambdaMinEpoch )
        else  :
           self.LambdaMinEpoch = np.zeros( LambdaProb.shape )
        if LambdaMaxEpoch is not None :
           self.LambdaMaxEpoch = np.array( LambdaMaxEpoch )
        else  :
           self.LambdaMaxEpoch = 1e10 * np.ones( LambdaProb.shape )


        self.step = 0

    def get_weights( self , epoch )  :
        self.step += 1
        
        #Update the probabilities according to the trend.
        lambda_prob = self.LambdaProb + self.step * self.LambdaProbTrend
        #Check that the probabilities remain within the min/max range
        mask_max = lambda_prob > self.LambdaProbMax
        lambda_prob[ mask_max ] = self.LambdaProbMax[ mask_max ]
        mask_min = lambda_prob < self.LambdaProbMin
        lambda_prob[ mask_min ] = self.LambdaProbMin[ mask_min ]

        #Remove those losses that should be started after a certain epoch.
        mask_epoch = self.LambdaMinEpoch > epoch
        lambda_prob[ mask_epoch ] = 0.0
        #Remove those losses that should be turn off after a certain epoch.
        mask_epoch = self.LambdaMaxEpoch < epoch
        lambda_prob[ mask_epoch ] = 0.0
        #Renormalize the probabilities.
        lambda_prob = lambda_prob / lambda_prob.sum()

        # Use numpy.random.choice to select an index based on the probabilities
        # The 'a' parameter is the range of indices: [0, 1, ..., len(probs) - 1]
        # The 'p' parameter is the probabilities for each index
        selected_index = np.random.choice( a=len( lambda_prob ), p=lambda_prob )
        current_lambda = np.zeros( np.shape( lambda_prob ) )
        current_lambda[ selected_index ] = self.LambdaVal[ selected_index ]

        return torch.tensor( current_lambda )



def get_loss_list( LossName , device  ) :

    loss_list = []
    for my_loss in LossName :
        if my_loss == 'MSE' :
            loss_list.append( loss_mse )
        if my_loss == 'KL'  :
            loss_list.append( loss_kl  )
        if my_loss == 'VAR' :
            loss_list.append( loss_var )
        if my_loss == 'SKE' :
            loss_list.append( loss_skew )
        if my_loss == 'KUR' :
            loss_list.append( loss_kurt )
        if my_loss == 'FOU' :
            loss_list.append( loss_Fourier().to(device) )
        if my_loss == 'FAL' :
            loss_list.append( loss_FAL().to(device) )
        if my_loss == 'FCL' :
            loss_list.append( loss_FCM().to(device) )
        if my_loss == 'CRPS' :
            loss_list.append( loss_CRPS().to(device) )

    return loss_list


def compute_loss( loss_list , output , target , LossName , loss_weight , output_samples = None , mu=None , logvar=None , free_bits = 0.0 ) :
    #Compute the loss function
    loss_vec = torch.zeros( len(LossName) )
    for ii , my_loss in enumerate( LossName ) :
        if my_loss == 'MSE' and loss_weight[ii] > 0 :
           loss_vec[ii] = loss_list[ii]( output , target )* loss_weight[ii]
        if my_loss == 'KL'  and loss_weight[ii] > 0 :
           loss_vec[ii] = loss_list[ii]( mu , logvar , free_bits = free_bits )* loss_weight[ii]
        if my_loss == 'VAR' and loss_weight[ii] > 0 :
           loss_vec[ii] = loss_list[ii]( output , target )* loss_weight[ii]
        if my_loss == 'SKE' and loss_weight[ii] > 0 :
           loss_vec[ii] = loss_list[ii]( output , target )* loss_weight[ii]
        if my_loss == 'KUR' and loss_weight[ii] > 0 :
           loss_vec[ii] = loss_list[ii]( output , target )* loss_weight[ii]
        if my_loss == 'FOU' and loss_weight[ii] > 0 :
           loss_vec[ii] = loss_list[ii]( output , target )* loss_weight[ii]
        if my_loss == 'FAL' and loss_weight[ii] > 0 :
           loss_vec[ii] = loss_list[ii]( output , target )* loss_weight[ii]
        if my_loss == 'FCL' and loss_weight[ii] > 0 :
           loss_vec[ii] = loss_list[ii]( output , target )* loss_weight[ii]
        if my_loss == 'CRPS' and loss_weight[ii] > 0 :
           loss_vec[ii] = loss_list[ii]( output_samples , target )* loss_weight[ii]


    return loss_vec




