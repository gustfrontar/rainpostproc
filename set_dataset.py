from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


#Construye un objeto que contiene los datos
class set_up_data(Dataset):
#    "Para utilizarse con el DataLoader de PyTorch"
    def __init__(self , Data , Input, Target):
        self.x_data = Input
        self.y_data = Target
        
        if len(self.x_data.shape)==3 :
            self.x_data = self.x_data[:,np.newaxis,:,:]  #Agrego una dimension de canales
        if len(self.y_data.shape)==3 :
            self.y_data = self.y_data[:,np.newaxis,:,:]  #Agrego una dimension de canales
        
       #Parametros para la normalizacion
        self.xmin, self.xmax = Data["XMin"], Data["XMax"]
        self.ymin, self.ymax = Data["YMin"], Data["YMax"]

        #Normalizacion de los datos
        #Basicamente esta diciendo:
        #argumentos: Input, normalizo min, normalizo max 
        self.x_data = self.normx( self.x_data )
        self.y_data = self.normy( self.y_data )   
        #self.x_data = norm( self.x_data, self.xmin, self.xmax)
        #self.y_data = norm( self.y_data, self.ymin, self.ymax)

    def __len__(self):
#        "Denoto el numero total de muestras"
        return self.x_data.shape[0]

    #Aca lo transformo a tensor finalmente  
        
    def __getitem__(self,index):
        x = torch.tensor(self.x_data[index,:,:], dtype=torch.float)
        y = torch.tensor(self.y_data[index,:,:], dtype=torch.float)
        return x, y
            
    #Funciones de normalizacion la que uso arriba
    def normx( self , data ):
        #return 2.0*(data-datamin)/(datamax-datamin)-1.0 #Normalizacion [-1,1]
        data = ( data - self.xmin ) / ( self.xmax - self.xmin ) 
        return data

    def normy( self , data ):
        #return 2.0*(data-datamin)/(datamax-datamin)-1.0 #Normalizacion [-1,1]
        data = ( data - self.ymin ) / ( self.ymax - self.ymin ) 
        return data

    
    def denormx( self , data ) :
        #return 0.5*(data+1.0)*(datamax-datamin)+datamin #Normalizacion [-1,1]
        data = ( data ) * ( self.xmax - self.xmin ) + self.xmin  #Normalizacion [0,1]
        return data
 
    def denormy( self , data ) :
        #return 0.5*(data+1.0)*(datamax-datamin)+datamin #Normalizacion [-1,1]
        data = ( data ) * ( self.ymax - self.ymin ) + self.ymin  #Normalizacion [0,1]
        return data
        
    def tonp( self , denorm=False ) : 
        if denorm :
            return self.denormx( self.x_data ) , self.denormy( self.y_data )
        else      :
            return self.x_data , self.y_data 
     


#Funcion para abrir los datos .npz y extraer las variables que elegimos, y lo guardamos en un diccionario
def get_data( Conf ) :
    

    #Cargamos el archivo npz
    data_from_file  = np.load( Conf['DataPath'] ) 

    #Generamos un diccionario vacio para almacenar variables
    Data=dict()

    #Cargamos todos los datos de precipitacion de nuestro modelo
    if Conf['TempFix'] : #Temporal fix for data size difference among experiments.
       Input = data_from_file[ Conf['InputName'] ][:,:-1,:-1]
    else  :
       Input = data_from_file[ Conf['InputName'] ][:,:,:]

    #Cargamos todos los datos de precipitacion observado
    if Conf['TempFix'] : #Temporal fix for data size difference among experiments.
       Target = data_from_file[ Conf['TargetName'] ][:,:-1,:-1]
    else  :
       Target = data_from_file[ Conf['TargetName'] ][:,:,:]
  
    #En este caso tanto el input como el target tiene las mismas dimensiones, por eso podemos usarlas para ambos
    Data['TotalLen'], Data['Nx'], Data['Ny']  = Input.shape
    
    indices = range(Data['TotalLen']) #Con el largo del dataset cuento cuantos hay y genero un vector de indices

    #Separo en los conjuntos de Entrenamiento y un conjunto que será Validacion/Testing
    train_ids, rest_ids = train_test_split(indices, test_size=1 - Conf['TrainRatio'] , shuffle=False )
    #Ahora a ese conjunto restante lo divido en Validacion y testing propiamente.
    val_ids, test_ids   = train_test_split(rest_ids, test_size= Conf['TestRatio'] / ( Conf['TestRatio'] + Conf['ValRatio'] ) , shuffle=False ) 


    #Guardo e imprimo por pantalla la cantidad de datos en cada conjunto
    Data['TrainLen'], Data['ValLen'], Data['TestLen'] = len(train_ids), len(val_ids), len(test_ids)

    print('--------------Indices------------------------')

    #-------------------------------------------------------
    print('Training set starts at :', str( np.min( train_ids) ) , ' and ends at: ', str( np.max( train_ids ) ) )
    print('Validation set starts at :', str( np.min( val_ids ) ) , ' and ends at: ', str( np.max( val_ids ) ) )
    print('Testing set starts at: ', str( np.min( test_ids ) ) , ' and ends at: ', str( np.max( test_ids ) ) )
    
    #Seleccionamos datos para nuestro entrenamiento
    train_x_data = Input[train_ids,:,:]
    train_y_data = Target[train_ids,:,:]
    
    #Seleccionamos datos para la validacion
    val_x_data = Input[val_ids,:,:]
    val_y_data = Target[val_ids,:,:]

    #Seleccionamos datos para el testeo
    test_x_data = Input[test_ids,:,:]
    test_y_data = Target[test_ids,:,:]

    #Separamos los minimos  y maximos para luego realizar una normalizacion de los datos
    Data["XMin"], Data["XMax"] = np.append(train_x_data,val_x_data,axis=0).min() , np.append(train_x_data,val_x_data,axis=0).max()
    Data["YMin"], Data["YMax"] = np.append(train_y_data,val_y_data,axis=0).min() , np.append(train_y_data,val_y_data,axis=0).max()
    
    Data['TrainDataSet']    = set_up_data(Data=Data, Input = train_x_data , Target = train_y_data )
    Data['ValDataSet']      = set_up_data(Data=Data, Input = val_x_data   , Target = val_y_data   )
    Data['TestDataSet']     = set_up_data(Data=Data, Input = test_x_data  , Target = test_y_data  )
    
    #Data['TrainDataLoader'] = torch.utils.data.DataLoader( TrainDataSet , batch_size=Conf['BatchSize'] , shuffle=True )
    #Data['ValDataLoader']   = torch.utils.data.DataLoader( ValDataSet   , batch_size=Data['ValLen']    , shuffle=False ) 
    #Data['TestDataLoader']  = torch.utils.data.DataLoader( TestDataSet  , batch_size=Data['TestLen']   , shuffle=False ) 
    
    #--------------------------------------------------------------------------------------------
    #Cargamos las fechas-------------------------------seccion nueva
    if Conf['DateFileName'] is not None:
        fechas = data_from_file[ Conf['DateFileName'] ]
        print('.')
        print('.')
        print('--------------Fechas------------------------------')
        print('Training set starts at :', str( fechas[np.min( train_ids)] ) , ' and ends at: ', str( fechas[np.max( train_ids )] ) )
        print('Validation set starts at :', str(fechas[ np.min( val_ids ) ]) , ' and ends at: ', str( fechas[np.max( val_ids ) ]) )
        print('Testing set starts at: ', str( fechas[np.min( test_ids )] ) , ' and ends at: ', str( fechas[np.max( test_ids ) ]) )

        Data['FechasTest'] = fechas[np.min(test_ids):np.max(test_ids)+1]
        
        

    return Data
