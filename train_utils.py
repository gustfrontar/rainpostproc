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

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' #Forzamos cpu

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
    

    
#Funcion que entrena el modelo y hace una validacion basica de su desempenio.   
def trainer( TrainConf , Data ) : 

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


    if ( TrainConf['LossType'] is None or TrainConf['LossType'] == "MSE" ) :
        Loss = torch.nn.MSELoss(reduction='mean')


    #Definimos el Scheduler para el Learning Rate
    if TrainConf['LrDecay'] :
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = TrainConf['Milestones'] , gamma = TrainConf['Gamma'] , verbose = True)
        

    #Listas donde guardamos loss de entrenamiento, y para el de validación la loss y las métricas de evaluación.
    Stats=dict()
    Stats['RMSE']=[];Stats['BIAS']=[];Stats['CorrP']=[];Stats['CorrS']=[]
    Stats['LossTrain']=[];Stats['LossVal']=[]

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
                      
            loss = Loss( outputs.float() , target.float() )
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
        Stats['LossVal'].append( Loss( output_val.float() , target_val.float() ).item() )

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
