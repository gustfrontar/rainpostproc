import random
import torch
import models
import numpy as np
import verificacion as ver
import set_dataset as ds
import pickle
import os
import plots
import gc
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu' #Forzamos cpu

# PyTorch deterministic settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    
# Set float32 precision
torch.set_float32_matmul_precision('high')

def define_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

#Early stopper class 
#https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        print('Early stop is enabled')

    def early_stop(self, TrainConf , model , validation_loss ):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            #Save the model that produces the minimum of the validation loss.
            print('The validation loss is the minimum reached so far.')
            print('Saving the current version of the model as the BestModel.')
            OutPath = TrainConf['OutPath'] + TrainConf['ExpName'] + "_" + str(TrainConf['ExpNumber']) + "/"
            models.save_model( model , OutPath , modelname='BestModel' )
        elif validation_loss > (self.min_validation_loss * (1.0 + self.min_delta / 100.0 ) ):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False    
    
def loss_var(output, target) :
    #Compute the variance of the output and target, then return the MSE between them. (image-wise)
    #nb = output.shape[0]
    varo = torch.var(output, dim=(1,2))
    vart = torch.var(target, dim=(1,2))
    return torch.mean((varo - vart)**2)

def loss_skew(output, target) :
    #Compute the skewness of the output and target, then return the MSE between them. (image-wise)
    #nb = output.shape[0]
    meano = torch.mean(output, dim=(1,2))
    meant = torch.mean(target, dim=(1,2))
    #stdo = torch.std(output, dim=(1,2))
    #stdt = torch.std(target, dim=(1,2))
    skwo = torch.mean( ((output - meano[:,None,None]))**3 , dim=(1,2) )
    skwt = torch.mean( ((target - meant[:,None,None]))**3 , dim=(1,2) )
    return torch.mean((skwo - skwt)**2)

def loss_kurt(output, target) :
    #Compute the kurtosis of the output and target, then return the MSE between them. (image-wise)
    #nb = output.shape[0]
    meano = torch.mean(output, dim=(1,2))
    meant = torch.mean(target, dim=(1,2))
    #stdo = torch.std(output, dim=(1,2))
    #stdt = torch.std(target, dim=(1,2))
    kurto = torch.mean( ((output - meano[:,None,None]))**4 , dim=(1,2) ) - 3.0
    kurt = torch.mean( ((target - meant[:,None,None]))**4 , dim=(1,2) ) - 3.0
    return torch.mean((kurto - kurt)**2)

def loss_mse(output, target):
    return torch.mean((output - target)**2)

def loss_dileoni( output , target , w ) :
    #Dileoni et al. 2024 loss function
    mse = loss_mse( output , target )
    var = loss_var( output , target )
    skew = loss_skew( output , target )
    kurt = loss_kurt( output , target )
    return w[0]*mse + w[1]*var + w[2]*skew + w[3]*kurt  

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


    def fcl(self, fft_pred, fft_truth ):
        # In general, FFTs here must be shifted to the center; but here we use the whole fourier space, so it is okay to no need have fourier shift operation
        conj_truth = torch.conj(fft_truth)
        numerator = (conj_truth*fft_pred).sum().real
        var_pred = (fft_pred.abs()**2).sum()
        var_truth = (fft_truth.abs()**2).sum()
        denominator = torch.sqrt(var_pred * var_truth)
        #print(fft_pred.shape)
        #import matplotlib.pyplot as plt
        #plt.imshow( fft_pred[0,:,:].cpu().detach().numpy() , cmap='jet' )
        #plt.colorbar()
        #plt.show( )
        
        #quit()
        #print(   numerator.item() , denominator.item() , 1. - numerator/denominator)
        #quit()
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
    def __init__(self, mode: str = 'MSE', weight_mode: str = 'log_distance', alpha: float = 1.0):
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
        distance = torch.fft.ifftshift(distance) 

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
        b, c, h, w = pred.shape
        device = pred.device

        # 1. Compute 2D FFT (operates independently on HxW planes)
        # Output is complex: [B, C, H, W, 2] or [B, C, H, W] (complex dtype)
        f_pred = torch.fft.fft2(pred, dim=(-2, -1))
        f_target = torch.fft.fft2(target, dim=(-2, -1))

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

def update_loss_wigths( loss , weigths ):
    #Update the weights of the different loss components according to their relative values.
    #The weigths are updated so that the contribution of each component to the total loss is similar.
    #This is done by dividing each weight by the corresponding loss value.
    #The weights are then normalized so that they sum to 1.
    ncomp = len(weigths)
    new_weigths = np.zeros(ncomp)
    sum_w = 0.0
    for i in range(ncomp):
        if loss[i] > 0.0:
            new_weigths[i] = weigths[i] / loss[i]
        else:
            new_weigths[i] = weigths[i]
        sum_w += new_weigths[i]
    for i in range(ncomp):
        new_weigths[i] /= sum_w
    print(loss,weigths,new_weigths)
    return new_weigths
 
#Funcion que entrena el modelo y hace una validacion basica de su desempenio.   
def trainer( TrainConf , Data ) : 
    
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    

    #Definimos el modelo en base a la clase seleccionada y la configuracion. 
    model = TrainConf['ModelClass']( TrainConf['ModelConf'] )

    model.to(device) #Cargamos en memoria, de haber disponible GPU se computa por ahí
    #Mi modelo ya tiene un valor para los pesos 
    models.initialize_weights(model) #Definimos los pesos iniciales con los cuales inicia el modelo antes de entrenarlo

    #Definimos el optimizador (descenso de gradiente)
    #Params conjunto de parametros que definen mi kernel
    #Para cada celdita del kernel tenes un 1 parametro X la cantidad de parametros 
    optimizer = torch.optim.Adam( model.parameters() , lr=TrainConf['LearningRate'] , weight_decay=TrainConf['WeightDecay'] ) 

    #Early stopper initialization
    if TrainConf['EarlyStop'] :
        early_stopper = EarlyStopper(patience=TrainConf['Patience'], min_delta=TrainConf['MinDelta'])


    #Definimos el Scheduler para el Learning Rate
    if TrainConf['LrDecay'] :
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = TrainConf['Milestones'] , gamma = TrainConf['Gamma'] , verbose = True)
        

    #Listas donde guardamos loss de entrenamiento, y para el de validación la loss y las métricas de evaluación.
    Stats=dict()
    Stats['RMSE']=[];Stats['BIAS']=[];Stats['CorrP']=[];Stats['CorrS']=[]
    Stats['LossTrain']=[];Stats['LossVal']=[]
    
    if TrainConf['LossType'] == "MSE" :
        loss_fn = loss_mse
    elif TrainConf['LossType'] == "RandomFourierLoss" :
        loss_fn = RandomFourierLoss( min_prob = TrainConf['RandomFourierLossMinProb'] ,
                                      max_prob = TrainConf['RandomFourierLossMaxProb'] ,
                                      prob_slope = TrainConf['RandomFourierLossProbSlope'] )
    else:
        print('Error: Loss type not recognized. Exiting.')
        exit(0) 
    loss_fn.to( device )    

    TrainDataLoader = torch.utils.data.DataLoader( Data['TrainDataSet'], batch_size=TrainConf['BatchSize'], shuffle=False)
    #Uno recorre varias veces los mismos datos para acercarse al minimo 
    #recordemos que 
    for epoch in range( TrainConf['MaxEpochs'] ):
        print('Epoca: '+ str(epoch+1) + ' de ' + str( TrainConf['MaxEpochs'] ) )

        #Entrenamiento del modelo        
        model.train()  #Esto le dice al modelo que se comporte en modo entrenamiento.

        sum_loss = 0.0
        batch_counter = 0

        # Iteramos sobre los minibatches. 
        for inputs, target in TrainDataLoader :

            #Enviamos los datos a la memoria.
            inputs, target = inputs.to(device), target.to(device)
            #-print( 'Batch ' + str(batch_counter) )

            optimizer.zero_grad()          
            #Aplicamos el modelo sobre el conjunto de datos que compone el mini-Batch
            outputs = model(inputs) 
             
            #Calculamos la funcion de costo entre la salida del modelo y el target esperado.
            loss = loss_fn( outputs , target , mode = 'train' ) 

            #Backpropagation
        
            #Calculamos el gradiente de la funcion de costo respecto de los pesos de la red.                
            loss.backward() 
            #Calculamos el valor actualizado de los pesos de la red, moviendolos en la direccion en 
            #la que el error desciende mas rapidamente
            optimizer.step()
                            
            batch_counter += 1
            sum_loss = sum_loss + loss.item()

        gc.collect()

        if TrainConf['LrDecay'] :
            scheduler.step()

        #Calculamos la loss media sobre todos los minibatches 
        Stats['LossTrain'].append( sum_loss / batch_counter )
    
        #Calculamos la loss sobre el conjunto de validacion
        _ , target_val , output_val = model_eval( model , Data['ValDataSet'] , numpy=False , denorm=False ) 
        with torch.no_grad():
            Stats['LossVal'].append( loss_fn( output_val , target_val , mode='val' ).item() )

        #Mostramos por pantalla la loss de esta epoca para training y validacion.
        print('Loss Train: ', str(Stats['LossTrain'][epoch]))
        print('Loss Val:   ', str(Stats['LossVal'][epoch]))

        if TrainConf['EarlyStop'] :
           if early_stopper.early_stop( TrainConf , model , Stats['LossVal'][epoch] ):  
              print('Warning: We have reached the patience of the early stop criteria')
              print('Stoping the training of the model') 
              print('Recovering the most successful version of the model')
              OutPath = TrainConf['OutPath'] + TrainConf['ExpName'] + "_" + str(TrainConf['ExpNumber']) + "/"
              model = models.load_model( OutPath , modelname = 'BestModel' )
              break

        #Calculamos metricas sobre el conjunto de testing con los datos desnormailzados.
        _ , target_test , output_test = model_eval( model , Data['TestDataSet'] , numpy=True , denorm = True )
        Stats['RMSE'].append(  ver.rmse(   output_test , target_test ) )
        Stats['BIAS'].append(  ver.bias(   output_test , target_test ) )
        Stats['CorrP'].append( ver.corr_P( output_test , target_test ) )
        Stats['CorrS'].append( ver.corr_S( output_test , target_test ) )  
    
    return model , Stats 

def model_eval( model, dataset, numpy=False , denorm = False):
    
    model.to( device )
    model.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)

    my_input_list = []
    my_target_list = []
    my_output_list = []

    with torch.no_grad():
        for batch in dataloader:
            my_input, my_target = batch
            my_input, my_target = my_input.to(device) , my_target.to(device)
            my_input_list.append(my_input)
            my_target_list.append(my_target)

            my_output = model(my_input)
            my_output_list.append(my_output)
            gc.collect()

    my_input = torch.cat(my_input_list, dim=0)
    my_target = torch.cat(my_target_list, dim=0)
    my_output = torch.cat(my_output_list, dim=0)
    
    if denorm :
       my_input  = dataset.denormx( my_input )
       my_target = dataset.denormy( my_target )
       my_output = dataset.denormy( my_output )

    if numpy:
        return my_input.cpu().numpy(), my_target.cpu().numpy(), my_output.cpu().detach().numpy()
    else:
        return my_input.cpu(), my_target.cpu(), my_output.cpu()

def meta_model_train( TrainConf ) : 

    #Input> TrainConf dictionary containing proper configuration for model training. 
    #Oputput> Figures and trained model in the OutPath folder. 
    OutPath = TrainConf['OutPath'] + TrainConf['ExpName'] + "_" + str(TrainConf['ExpNumber']) + "/"
    print( 'My outpath is : ' , OutPath )

    print(OutPath)
    if not os.path.exists( OutPath ):
        # Creo un nuevo directorio si no existe (para guardar las imagenes y datos)
        os.makedirs( OutPath )

    #Inicializamos los generadores de pseudo random numbers
    define_seed(seed=TrainConf['RandomSeed'])

    #Obtenemos el conjunto de datos (dataloaders)
    Data = ds.get_data( TrainConf )
    print(Data['Nx'],Data['Ny'])
    print("Muestras de Train / Valid / Test: ",(Data["TrainLen"],Data["ValLen"],Data["TestLen"]))

    TrainConf['ModelConf']['Nx']=Data['Nx']
    TrainConf['ModelConf']['Ny']=Data['Ny']

    #Entrenamos el modelo
    TrainedModel , ModelStats  = trainer( TrainConf , Data )

    #Save the model
    models.save_model( TrainedModel , OutPath )
    #Save model stats
    with open( OutPath + '/ModelStats.pkl', 'wb' ) as handle:
        pickle.dump( ModelStats , handle , protocol=pickle.HIGHEST_PROTOCOL )
    
    #Save configuration.
    with open( OutPath + '/TrainConf.pkl', 'wb' ) as handle:
        pickle.dump( TrainConf , handle , protocol=pickle.HIGHEST_PROTOCOL )

    #Plot basic training statistics
    #plots.PlotModelStats( ModelStats , OutPath )

    return 0 
