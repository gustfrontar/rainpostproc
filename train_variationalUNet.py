import numpy as np
import plots
import torch
import set_dataset as ds
import pickle
import train_utils_hsinn as tu
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
TrainConf['MaxEpochs']= 50
TrainConf['LearningRate']=1.0e-4
#TrainConf['LrDecay']=True
#TrainConf['Milestones']=[10]
#TrainConf['Gamma']=0.08
#TrainConf['WeightDecay']=0.0e-6

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
TrainConf['Beta']           = 1.0  #Weight of the KL loss term.
TrainConf['FreeBits']       = 0.5   #Number of free bits for KL loss term.

#Parametros de configuracion que dependen del modelo a utilizar.
TrainConf['ModelConf']=dict()
TrainConf['ModelConf']['LatentDim']      =  32           #Latent space dimension.
TrainConf['ModelConf']['HiddenDims']     = [8, 32, 64, 128]  #Number of channels for each convolutional layer.
TrainConf['ModelConf']['InputChannels']  = 1              #
TrainConf['ModelConf']['OutputChannels'] = 1              #
TrainConf['ModelConf']['BatchNorm']      = True          #

OutPath = "../experiments/"+ TrainConf['ExpName'] +"_"+str(TrainConf['ExpNumber'])+"/"
print(OutPath)
if not os.path.exists( OutPath ):
       # Creo un nuevo directorio si no existe (para guardar las imagenes y datos)
       os.makedirs( OutPath )

#Inicializamos los generadores de pseudo random numbers
tu.define_seed(seed=TrainConf['RandomSeed'])

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
        use_batchnorm=TrainConf['ModelConf']['BatchNorm']
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
    max_beta_kl=TrainConf['Beta'],
    free_bits=TrainConf['FreeBits'],
    use_lr_scheduler=True,
    early_stopping_patience=20,
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

vunet.test_random_cases(my_model, ValDataLoader, num_cases=10, device='cuda',outpath=OutPath)


