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
TrainConf['BatchSize']= 10
TrainConf['MaxEpochs']= 100
TrainConf['LearningRate']=1.0e-4

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
#Loss function scheduler
TrainConf['LambdaProb'] = np.array([0.98 , 0.01 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.01 ])    #Lambda prob for stochastic annealing.
#                                  MSE    KL   VAR   SKE   KUR   FOU   FAL   FCL  CRPS
TrainConf['LambdaProbTrend'] = np.array([0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) #Lambda prob trend for stochastic annealing
#                                       MSE    KL   VAR   SKE   KUR   FOU   FAL   FCL  CRPS
TrainConf['LambdaProbMax'] = np.array([1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 ])    #Maximum lambda prob for stochastic annealing
TrainConf['LambdaProbMin'] = np.array([0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ])    #Minimum lambda prob for stochastic annealing
#                                  MSE    KL   VAR   SKE   KUR   FOU    FAL   FCL   CRPS
TrainConf['LambdaVal'] = np.array([1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 0.0 , 1.0 , 1.0 , 1.0 ])
#                                  MSE    KL   VAR   SKE   KUR   FOU  FAL   FCL  CRPS
TrainConf['LambdaMinEpoch'] = np.array([ 0 , 20 , 0 , 0 , 0 , 0 , 0 , 0 , 20 ])
TrainConf['LambdaMaxEpoch'] = None
TrainConf['LossName']  = np.array(['MSE','KL','VAR','SKE','KUR','FOU','FAL','FCM','CRPS'])
TrainConf['StochAnn']  = True   #Weather we activate the stochastic annealing.

TrainConf['FreeBits']           = 0.5    #Number of free bits for KL loss term.
TrainConf['WeigthedLossFlag']   = False  #[bool] Wether to use a weighted MSE loss or not.
TrainConf['CRPSNumSamples']     = 10

#Parametros de configuracion que dependen del modelo a utilizar.
TrainConf['ModelConf']=dict()
TrainConf['ModelConf']['LatentDim']      =  16           #Latent space dimension.
TrainConf['ModelConf']['HiddenDims']     =  [12,24,48]  #Number of channels for each convolutional layer.
TrainConf['ModelConf']['InputChannels']  = 1              #
TrainConf['ModelConf']['OutputChannels'] = 1              #
TrainConf['ModelConf']['BatchNorm']      = True           #
TrainConf['ModelConf']['SkipConnections'] = [1.0,1.0,None]       # 1 means activate skip connection, None means turn it off. Last element correspond to the highest level in the UNET

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
vunet.train_vae_model(
    my_model,
    TrainDataLoader,
    val_loader=ValDataLoader,
    epochs=TrainConf['MaxEpochs'],
    learning_rate=TrainConf['LearningRate'],
    device='cuda',
    free_bits=TrainConf['FreeBits'],
    weighted_loss_flag=TrainConf['WeigthedLossFlag'],
    use_lr_scheduler=True,
    early_stopping_patience=TrainConf['MaxEpochs'],
    grad_clip=1.0,
    save_best=True,
    checkpoint_dir = OutPath ,
    LambdaProb = TrainConf['LambdaProb'] , 
    LambdaProbMax = TrainConf['LambdaProbMax'] ,
    LambdaProbMin = TrainConf['LambdaProbMin'] ,
    LambdaProbTrend = TrainConf['LambdaProbTrend'] ,
    LambdaVal       = TrainConf['LambdaVal'] ,
    LambdaMinEpoch  = TrainConf['LambdaMinEpoch'] ,
    LambdaMaxEpoch  = TrainConf['LambdaMaxEpoch'] ,
    StochAnn        = TrainConf['StochAnn'] ,
    LossName        = TrainConf['LossName'] ,
    CRPSNumSamples  = TrainConf['CRPSNumSamples']
)
# Save the final model
torch.save(my_model.state_dict(), OutPath + 'final_model.pth')
print("Final model saved.")
# Save training configuration
with open(OutPath + 'train_config.pkl', 'wb') as f:
    pickle.dump(TrainConf, f)
print("Training configuration saved.")

# Plot training history
#vunet.plot_training_history(thistory, save_path=OutPath + '/training_history.png')

vunet.test_random_cases(my_model, TrainDataLoader, num_cases=10, device='cuda',outpath=OutPath)


