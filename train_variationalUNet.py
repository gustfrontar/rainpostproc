import numpy as np
import plots
import torch
import set_dataset as ds
import pickle
import os
import default_conf as dc
import model_variationalUNet as vunet

TrainConf = dc.TrainConf #Get default configuration.


#Exp type
TrainConf['ExpName']    = 'VARUNET'
TrainConf['ExpNumber']  = 0
TrainConf['TempFix'] = False                     #Temporal fix for data size difference among experiments.

#Hiperparametros
TrainConf['RandomSeed']=1029
TrainConf['BatchSize']= 20
TrainConf['MaxEpochs']= 100
TrainConf['LearningRate']=2.5e-4

#Paths
TrainConf['OutPath']  ="../experiments/"

#Data
TrainConf['DataPath'] ="../data/medios.npz"
TrainConf['InputName']      = 'pp_medios_gfs'
TrainConf['TargetName']     = 'pp_medios_gsmap'
TrainConf['TrainRatio']     =  0.8
TrainConf['ValRatio']       =  0.1
TrainConf['TestRatio']      =  0.1
TrainConf['DateFileName']   =  'files'
TrainConf['LambdaKL']       = 1.0  #Weight of the KL loss term.
TrainConf['LambdaMSE']      = 1.0  #Weigth of the MSE loss term.
TrainConf['FreeBits']       = 1.0  #Number of free bits for KL loss term.
TrainConf['LambdaVar']      = 0.0  #Weight for variance loss term.
TrainConf['LambdaSkew']     = 0.0  #Weight for skewness loss term.
TrainConf['LambdaKurt']     = 0.0  #Weight for kurtosis loss term.
TrainConf['LambdaFourier']  = 0.0  #Weight for fourier loss term.
TrainConf['LambdaRandomFourier'] = 0.0 #Weigth for the random fourier loss term.  
TrainConf['WeigthedLossFlag']   = False  #[bool] Wether to use a weighted MSE loss or not.
TrainConf['FourierLossFlag']    = False  #[bool] Wether to use a fourier loss term or not.
TrainConf['RandomFourierLossFlag'] = False #[bool] Wether to use a random fourier loss. 
TrainConf['RandomFourierLossMinProb'] = 0.0 #
TrainConf['RandomFourierLossMaxProb'] = 0.9 #
TrainConf['RandomFourierLossProbSlope'] = 0.25e-4


#Parametros de configuracion que dependen del modelo a utilizar.
TrainConf['ModelConf']=dict()
TrainConf['ModelConf']['LatentDim']      =  32           #Latent space dimension.
TrainConf['ModelConf']['HiddenDims']     =  [6, 12, 24 , 24]  #Number of channels for each convolutional layer.
TrainConf['ModelConf']['InputChannels']  = 1              #
TrainConf['ModelConf']['OutputChannels'] = 1              #
TrainConf['ModelConf']['BatchNorm']      = True           #
TrainConf['ModelConf']['SkipConnections'] = [None,None,None,None]       # 1 means activate skip connection, None means turn it off. Last element correspond to the highest level in the UNET

OutPath = "../experiments/"+ TrainConf['ExpName'] +"_"+str(TrainConf['ExpNumber'])+"/"
print(OutPath)
if not os.path.exists( OutPath ):
       # Creo un nuevo directorio si no existe (para guardar las imagenes y datos)
       os.makedirs( OutPath )

#Inicializamos los generadores de pseudo random numbers
torch.manual_seed(TrainConf['RandomSeed'])
np.random.seed(TrainConf['RandomSeed'])
np.random.seed(TrainConf['RandomSeed'])

#Obtenemos el conjunto de datos (dataloaders)
Data = ds.get_data( TrainConf ) 
print(Data['Nx'],Data['Ny'])                             
print("Muestras de Train / Valid / Test: ",(Data["TrainLen"],Data["ValLen"],Data["TestLen"]))

TrainConf['ModelConf']['Nx']=Data['Nx']
TrainConf['ModelConf']['Ny']=Data['Ny']

#Create train data loader.
TrainDataLoader = torch.utils.data.DataLoader( Data['TrainDataSet'], batch_size=TrainConf['BatchSize'], shuffle=True)
ValDataLoader   = torch.utils.data.DataLoader( Data['ValDataSet'], batch_size=TrainConf['BatchSize'], shuffle=False)

# Initialize model
#Define the model
my_model = vunet.SkipConnectionVAE(
        input_channels=TrainConf['ModelConf']['InputChannels'],
        output_channels=TrainConf['ModelConf']['OutputChannels'],
        latent_dim=TrainConf['ModelConf']['LatentDim'],
        input_height=TrainConf['ModelConf']['Nx'],
        input_width=TrainConf['ModelConf']['Ny'],
        hidden_dims=TrainConf['ModelConf']['HiddenDims'],
        use_batchnorm=TrainConf['ModelConf']['BatchNorm'],
        skip_connections=TrainConf['ModelConf']['SkipConnections']
    )  

#Initialize model weights
my_model.apply(vunet.init_weights)
  
print(f"Model initialized with latent dimension: {TrainConf['ModelConf']['LatentDim']}")
#print(f"Encoder output size: {my_model.encoded_width} x {my_model.encoded_height} x {my_model.encoded_channels}")
print(f"Total parameters: {sum(p.numel() for p in my_model.parameters()):,}")
    

# Train the model
thistory , best_loss = vunet.train_vae_model(
    my_model,
    TrainDataLoader,
    val_loader=ValDataLoader,
    epochs=TrainConf['MaxEpochs'],
    learning_rate=TrainConf['LearningRate'],
    device='cuda',
    max_lambda_kl=TrainConf['LambdaKL'],
    free_bits=TrainConf['FreeBits'],
    lambda_mse=TrainConf['LambdaMSE'],
    lambda_var=TrainConf['LambdaVar'],
    lambda_skew=TrainConf['LambdaSkew'],
    lambda_kurt=TrainConf['LambdaKurt'],
    lambda_fourier=TrainConf['LambdaFourier'],
    lambda_random_fourier=TrainConf['LambdaRandomFourier'],
    weigthed_loss_flag=TrainConf['WeigthedLossFlag'],
    fourier_loss_flag=TrainConf['FourierLossFlag'],
    random_fourier_loss_flag=TrainConf['RandomFourierLossFlag'],
    random_fourier_loss_max_prob = TrainConf['RandomFourierLossMaxProb'],
    random_fourier_loss_min_prob = TrainConf['RandomFourierLossMinProb'],
    random_fourier_loss_prob_slope = TrainConf['RandomFourierLossProbSlope'],
    use_lr_scheduler=True,
    early_stopping_patience=TrainConf['MaxEpochs'],
    grad_clip=1.0,
    save_best=True,
    checkpoint_dir = OutPath
)
# Save the final model
torch.save(my_model.state_dict(), OutPath + 'final_model.pth')
print("Final model saved.")
# Save training configuration
with open(OutPath + 'train_config.pkl', 'wb') as f:
    pickle.dump(TrainConf, f)
print("Training configuration saved.")

# Plot training history
vunet.plot_training_history(thistory, save_path=OutPath + '/training_history.png')

vunet.test_random_cases(my_model, TrainDataLoader, num_cases=10, device='cuda',outpath=OutPath)


