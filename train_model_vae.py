import numpy as np
import plots
import set_dataset as ds
import pickle
import train_utils_hsinn as tu
import os
import default_conf as dc
import model_vae as vae

TrainConf = dc.TrainConf #Get default configuration.


#Exp type
TrainConf['ModelClass'] = models.EncoderDecoder  #Determino el modelo a utilizar.
TrainConf['ExpName']    = 'ENCODECO-TEST'
TrainConf['ExpNumber']  = 0
TrainConf['TempFix'] = False                     #Temporal fix for data size difference among experiments.

#Hiperparametros
TrainConf['RandomSeed']=1029
TrainConf['BatchSize']= 20
TrainConf['MaxEpochs']= 60
TrainConf['LearningRate']= 1.0e-4
TrainConf['LrDecay']=True
TrainConf['Milestones']=[10]
TrainConf['Gamma']=0.08
TrainConf['LossType']="MSE"
TrainConf['WeightDecay']=0.0e-6

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

#Parametros de configuracion que dependen del modelo a utilizar.
#TrainConf['ModelConf']=dict()
#TrainConf['ModelConf']['KernelSize'] =  5           #Convolutional kernel size.
#TrainConf['ModelConf']['Pool']       =  2           #Pooling kernel size 
#TrainConf['ModelConf']['DecoType'] = 'bilinear'     #conv, bilinear, bicubic, nearest  (up sample operation type)
#TrainConf['ModelConf']['Channels'] = [1,16,16,32,64,128,64,32,16,1]  #Number of channels for each convolutional layer.
#TrainConf['ModelConf']['InActivation'] = 'ReLU'             #Activation function of the hidden layers.
#TrainConf['ModelConf']['OutActivation'] = 'SiLU'            #Activation function of the output layer.
#TrainConf['ModelConf']['Bias']=[True]                       #Bias parameter. 
#TrainConf['ModelConf']['EnableHSiNNEpoch']=[20]             #At which epoch will High Order Statistics-informed Neural Networks be enabled.
TrainConf['ModelConf']=dict()
TrainConf['ModelConf']['LatentDim'] =  20           #Latent space dimension.
TrainConf['ModelConf']['HiddenDims'] = [32, 64, 128, 256, 512]  #Number of channels for each convolutional layer.


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

# Initialize model

model = vae.ConvVAE(input_channels=1, latent_dim=TrainConf['ModelConf']['LatentDim'], hidden_dims=TrainConf['ModelConf']['HiddenDims'], input_size=Data['Nx'])
    
print(f"Model initialized with latent dimension: {latent_dim}")
print(f"Encoder output size: {model.encoded_size}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
# Train the model
train_losses = vae.train_model(model, train_loader, epochs=epochs, 
                            learning_rate=learning_rate, device=device)
    
# Generate and display samples
samples = vae.generate_samples(model, num_samples=16, device=device)
    
# Plot generated samples
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i].squeeze(), cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()
    
# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()



#Entrenamos el modelo
#TrainedModel , ModelStats  = tu.trainer( TrainConf , Data )
       
#Save the model
#models.save_model( TrainedModel , OutPath )
#Save training stats
#with open( OutPath + '/ModelStats.pkl', 'wb' ) as handle:
#    pickle.dump( ModelStats , handle , protocol=pickle.HIGHEST_PROTOCOL )   
#Save Configuration
#with open( OutPath + '/TrainConf.pkl', 'wb' ) as handle:
#    pickle.dump( TrainConf , handle , protocol=pickle.HIGHEST_PROTOCOL )   

#Plot basic training statistics
#plots.PlotModelStats( ModelStats , OutPath )


   



