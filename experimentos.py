#-------Librerias----
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import plots
import random
import set_dataset as ds
import verificacion as ver
from sklearn.model_selection import train_test_split

#------Semilla--------
def define_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

define_seed(seed=1029)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#------------------------------

#Funcion para abrir los datos .npz y extraer las variables que elegimos, y lo guardamos en un diccionario
def get_data(path, var_input, var_target, train_ratio, val_ratio, test_ratio):
    

    #Cargamos el archivo npz
    data_from_file  = np.load(path) 

    #Generamos un diccionario vacio para almacenar variables
    Data=dict()

    #Cargamos todos los datos de precipitacion de nuestro modelo
    Input = data_from_file[var_input]

    #Cargamos todos los datos de precipitacion observado
    Target = data_from_file[var_target]
    
    #En este caso tanto el input como el target tiene las mismas dimensiones, por eso podemos usarlas para ambos
    Data["len_total"], Data["nx"], Data["ny"]  = Input.shape
    
    indices = range(Data["len_total"]) #Con el largo del dataset cuento cuantos hay y genero un vector de indices

    #Separo en los conjuntos de Entrenamiento y un conjunto que será Validacion/Testing
    train_ids, rest_ids = train_test_split(indices, test_size=1 - train_ratio , shuffle=False )
    #Ahora a ese conjunto restante lo divido en Validacion y testing propiamente.
    val_ids, test_ids   = train_test_split(rest_ids, test_size=test_ratio/(test_ratio + val_ratio) , shuffle=False ) 

    #Guardo e imprimo por pantalla la cantidad de datos en cada conjunto
    Data["len_train"], Data["len_val"], Data["len_test"] = len(train_ids), len(val_ids), len(test_ids)
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
    Data["xmin"], Data["xmax"] = np.append(train_x_data,val_x_data,axis=0).min() , np.append(train_x_data,val_x_data,axis=0).max()
    Data["ymin"], Data["ymax"] = np.append(train_y_data,val_y_data,axis=0).min() , np.append(train_y_data,val_y_data,axis=0).max()
    
    return Data, train_x_data, train_y_data, val_x_data, val_y_data, test_x_data, test_y_data


#-----------------------------------------------------------------------------------

#NOMBRES
Input_name = 'pp_medios_gfs'
Target_name = 'pp_medios_gsmap'
Experimento = Input_name + 'vs' + Target_name

#---------------------------------------------------

#Porcentaje de reparticion de los conjuntos de Train / Validation / Testing
train_ratio = .7
val_ratio = .2
test_ratio = .1

path = '/home/fernando.huaranca/datosmunin/regiones_R_025/medios.npz'


# Lectura de los datos
#Reflectividad de radar simulada
#Tasa de precicpitacion
Data, x_train, y_train, x_val, y_val, x_test, y_test = get_data(path=path,
                                                                   var_input = Input_name,
                                                                   var_target = Target_name,
                                                                   train_ratio = train_ratio,
                                                                   val_ratio = val_ratio,
                                                                   test_ratio = test_ratio)
nx, ny = Data["nx"], Data["ny"]

print("Muestras de Train / Valid / Test: ",(Data["len_train"],Data["len_val"],Data["len_test"]))

#--------------------------------------------------------------------------------

guardar_modelo = True

#Una forma de identificar la configuración que utilizamos
Modelo = "Convolucion"
Numero_exp = 3 #aca estaria bueno ir cambiando con los experimentos que vamos haciendo

#HIPERPARAMETROS
batch_size= 50
max_epochs = 5000
learning_rate = 1e-3

#El learning rate decay seria algo que ayuda a converger. A partir de una cierta epoca el learning rate
#decay va decayendo a un ritmo de 0.1 
lr_decay = False
if lr_decay:
    milestones = [45] ; gamma=0.1

#Definimos la función de costo que queremos minimizar, y también el método de calculo sobre el batch.
MSE_Loss = torch.nn.MSELoss(reduction='mean')

#---------------------------------------------------------------------------

Directorio_out = "/home/fernando.huaranca/datosmunin/salidas/"+Modelo+"_"+Experimento+"_"+str(Numero_exp)+"/"

if not os.path.exists(Directorio_out):
    # Creo un nuevo directorio si no existe (para guardar las imagenes y datos)
    os.makedirs(Directorio_out)
    print("El nuevo directorio asociado a "+ Experimento +" "+str(Numero_exp)+" ha sido creado!")

#----------------------------------------------------------------------------------

class Convolucional(nn.Module):

    def __init__(self, nx, ny,filters): #Definimos los atributos de la clase
        super().__init__()

        #Nx serian tus entradas latitud (en nuestro ejemplo)
        #Ny tambien tu otra entrada lo que seria longitud
        self.nx, self.ny = nx, ny #Si se usan linears, es bueno guardalo...

        #in_chanels: seria tu matriz de entrada que es 1
        #out chanel: tu matriz de salida que es 16 salidas, una por cada kernel
        #kernel_size: que tamaño tiene el kernel 5x5
        #stride: pasos de iteracion en este caso se mueve un punto de reticula
        #padding: agrega 2 columna y dos filas
        #paddin_mode : reflect es que el borde se pone, en este caso uno similar al punto de reticula, podria ser 0
        #bias : True es buena idea activarlo
        self.conv_1 = nn.Conv2d(in_channels = filters[0],out_channels = filters[1], kernel_size=5 ,stride = 1, padding= 2, bias=True, padding_mode='reflect')
        self.conv_2 = nn.Conv2d(in_channels = filters[1], out_channels = filters[2], kernel_size=5 ,stride = 1, padding= 2, bias=True, padding_mode='reflect')
        
        self.activation_1 = nn.ReLU()
        self.activation_2 = nn.SiLU()
        
        #self.dropout_2d = nn.Dropout2d(p=0.2, inplace=False)
        
        #self.maxpool = nn.MaxPool2d(2)
        
        #self.transpconv_1 = nn.ConvTranspose2d(in_channels = filters[1],out_channels = filters[2], kernel_size=5 ,stride = 2, padding= 2, output_padding=1, bias=True)
        
        #self.Conv_BN_1 = torch.nn.BatchNorm2d(filters[1],affine=True) #Igual cantidad que la cantidad de filtros de salida
        

    def forward(self, x): #Ejecutamos con este método los atributos que definimos

        #de vector a matrices
        x = torch.unsqueeze(x,1) # x - Shape 4D: (batch size, filtros, nx, ny)
        
        x = self.conv_1(x)
        x = self.activation_1(x)

        x = self.conv_2(x)
        x = self.activation_2(x)
        #de matrices a vector
        return torch.squeeze(x)
    
#una imagen de entrada en el primer filtro

filters = [1,16,1]

#Aca instancia la clase como modelo
model = Convolucional(nx,ny,filters)
model.to(device) #Cargamos en memoria, de haber disponible GPU se computa por ahí
print("nx = {nx}","ny = {ny}")

#------------------------------------------------------------

#Inicializas tu kernel de manera aleatoria
def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight) #ayuda a que el primer peso sea efectivo
            torch.nn.init.constant_(m.bias, 0)

#Mi modelo ya tiene un valor para los pesos 
initialize_weights(model) #Definimos los pesos iniciales con los cuales inicia el modelo antes de entrenarlo

#Definimos el optimizador (descenso de gradiente)
#Params conjunto de parametros que definen mi kernel
#Para cada celdita del kernel tenes un 1 parametro X la cantidad de parametros 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

#Definimos el Scheduler para el Learning Rate
if lr_decay:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = gamma, verbose = True)

#-------------------------------------------------------------------------------------


#Preparamos los 3 conjuntos que definimos entrenamiento, validación y testing para que ingresen al dataloader

#Input matriz de 40x40x31
#Output matriz de 40x40x31
train_subset = ds.set_up_data(Data=Data, Input = x_train, Target = y_train)
val_subset = ds.set_up_data(Data=Data, Input = x_val,   Target = y_val)
test_subset = ds.set_up_data(Data=Data, Input = x_test,  Target = y_test)

#Definimos los dataloaders
#El dataloader de train tiene definido el batch_size, no asi los de validacion y test.
#Generalmente porque en comparación son mucho más chicos.
dataloader_train = DataLoader(train_subset, batch_size = batch_size, shuffle=True)   
dataloader_val   = DataLoader(val_subset , batch_size=len( val_subset) )
dataloader_test  = DataLoader(test_subset , batch_size=len( test_subset) )

#-----------------------------------------------------------------------------------------

#Listas donde guardamos loss de entrenamiento, y para el de validación la loss y las métricas de evaluación.
RMSE, BIAS, Corr_P, Corr_S = [], [], [], []
loss_train, loss_val = [], []

#Uno recorre varias veces los mismos datos para acercarse al minimo 
#recordemos que 
for epoch in range(max_epochs):
  print('Epoca: '+ str(epoch+1) + ' de ' + str(max_epochs) )

  #Entrenamiento del modelo        
  model.train()  #Esto le dice al modelo que se comporte en modo entrenamiento.

  sum_loss = 0.0
  batch_counter = 0

  # Iteramos sobre los minibatches. 
  for inputs, target in dataloader_train:

    #Enviamos los datos a la memoria.
    inputs, target = inputs.to(device), target.to(device)
    #-print( 'Batch ' + str(batch_counter) )

    optimizer.zero_grad()

    #Sobre los 100 inputs que salieron del dataloader despues de la relu silu y todo eso
    outputs = model(inputs) 
            
    loss = MSE_Loss(outputs.float(), target.float())
                    
    loss.backward() #la derivada
    optimizer.step() #calcula el learning rate y todo eso
                    
    batch_counter += 1
    sum_loss = sum_loss + loss.item()

  if lr_decay:
    scheduler.step()

  #Calculamos la loss media sobre todos los minibatches 
  loss_train.append( sum_loss / batch_counter )

  #---------------------------------------------------------------------

  #ACA EVALUA COMO TE VA DANDO EN CADA EPOCA EL CONJUNTO DE VALIDACION Y TESTEO NO ENTRENA SOLO EVALUA
  #SI LO HARIAS EN OTRO BLOQUE LO QUE PASARIA ES Q LO EVALUARIAS EN TODOO EL CON JUNTO
  #Calculamos la funcion de costo para la muestra de validacion.
  input_val , target_val = next( iter( dataloader_val ) )
  input_val , target_val = input_val.detach().to(device) , target_val.detach().to(device) 
  with torch.no_grad():
    output_val = model( input_val )

  loss_val.append( MSE_Loss( output_val , target_val ).item() )

  #Calculamos la funcion de costo para la muestra de testing.
  model.eval()   #Esto le dice al modelo que lo usaremos para evaluarlo (no para entrenamiento)
  input_test , target_test = next( iter( dataloader_test ) )
  input_test , target_test = input_test.detach().to(device) , target_test.detach().to(device) 
  #zzprint( input_test.shape )
  with torch.no_grad():
    output_test = model( input_test )

  #Calculo de la loss de la epoca
  print('Loss Train: ', str(loss_train[epoch]))
  print('Loss Val:   ', str(loss_val[epoch]))

  ###################################
  np_input_test  = ds.denorm( input_test.numpy()  , dataloader_test.dataset.xmin, dataloader_test.dataset.xmax )
  np_target_test = ds.denorm( target_test.numpy() , dataloader_test.dataset.ymin, dataloader_test.dataset.ymax )
    
  #Mis matrices desnormalizadas
  np_output_test = ds.denorm( output_test.detach().numpy() , dataloader_test.dataset.ymin, dataloader_test.dataset.ymax )
    
  np_input_val  = ds.denorm( input_val.numpy()  , dataloader_val.dataset.xmin, dataloader_val.dataset.xmax )
  np_target_val = ds.denorm( target_val.numpy() , dataloader_val.dataset.ymin, dataloader_val.dataset.ymax )
  np_output_val = ds.denorm( output_val.detach().numpy() , dataloader_val.dataset.ymin, dataloader_val.dataset.ymax )
  
  #Calculo de metricas RMSE, BIAS, Correlacion de Pearson y Spearman
  RMSE.append( ver.rmse( np_output_val , np_target_val ) )
  BIAS.append( ver.bias( np_output_val , np_target_val ) )
  Corr_P.append( ver.corr_P( np_output_val , np_target_val ) )
  Corr_S.append( ver.corr_S( np_output_val , np_target_val ) ) 

  #Me gustaria ver el del train tambien

#----------------------------------------------------------------------------------
  
print('Proceso completado!')
print('Iniciando guardado de variables y modelo')


#Aca me estoy llevando las matrices DESnormalizadas. Nose si llevarlas normalizadas te cambia algo
if guardar_modelo:
    torch.save(model.state_dict(), Directorio_out+"Modelo_exp_"+str(Numero_exp)+".pth")
    
if guardar_modelo: #Guardar los datos de test
    np.savez(Directorio_out+"Datos_test_"+Input_name+"_vs_"+Target_name+"_"+str(Numero_exp)+".npz",
         Input= np_input_test,
         Target= np_target_test,
         Modelo= np_output_test,
         loss_train = loss_train, loss_val = loss_val,
         RMSE = RMSE, BIAS = BIAS, Corr_P = Corr_P, Corr_S = Corr_S,
         Experimento = Experimento, Input_name = Input_name, Target_name = Target_name,
         max_epochs = max_epochs, nx = nx, ny = ny)
    
print('Guardado completado!')
print('Fin')