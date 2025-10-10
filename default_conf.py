import models

#Una forma de identificar la configuración que utilizamos
TrainConf=dict()  #Guardamos todos los hyperparametros en un diccionario.
#Exp type
TrainConf['ModelClass'] = models.unet  #Determino el modelo a utilizar.
TrainConf['ExpName']    = 'TEST'
TrainConf['ExpNumber']  = 0
#Hiperparametros
TrainConf['RandomSeed']=1029
TrainConf['BatchSize']= 100
TrainConf['MaxEpochs']= 40
TrainConf['LearningRate']= 1.0e-4
TrainConf['LrDecay']=True
TrainConf['Milestones']=[20]
TrainConf['Gamma']=0.1
TrainConf['LossType']="MSE"
TrainConf['WeightDecay']=1.0e-6
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
TrainConf['ModelConf']=dict()
TrainConf['ModelConf']['KernelSize'] =  5           #Convolutional kernel size.
TrainConf['ModelConf']['Pool']       =  2           #Pooling kernel size 
TrainConf['ModelConf']['DecoType'] = 'bilinear'     #conv, bilinear, bicubic, nearest  (up sample operation type)
TrainConf['ModelConf']['Channels'] = [1,32,32,64,128,64,1]  #Number of channels for each convolutional layer.
TrainConf['ModelConf']['InActivation'] = 'ReLU'             #Activation function of the hidden layers.
TrainConf['ModelConf']['OutActivation'] = 'SiLU'            #Activation function of the output layer.
TrainConf['ModelConf']['Bias']=[True]                       #Bias parameter. 

TrainConf['ModelConf']['BatchNorm'] = False

TrainConf['EarlyStop'] = False                  #If early stop is activated or not
TrainConf['Patience'] = 5                       #How many increasing loss epochs we will wait until interrupting the training. 
TrainConf['MinDelta'] = 0.5                     #How big an improvement in the loss will be tolerated before considering it a degradation of the loss [%]
TrainConf['TempFix'] = False                     #Temporal fix for data size difference among experiments.
   



