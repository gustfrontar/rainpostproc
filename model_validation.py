import numpy as np
import plots
import set_dataset as ds
import pickle
import models
import verificacion as ver
import train_utils as tu

ExpConf = dict()
#Exp type
ExpConf['ExpName']    = 'ENCODECO'
ExpConf['ExpNumber']  = 2
ExpConf['ExpPath'] = "./salidas/"+ ExpConf['ExpName'] +"_"+str(ExpConf['ExpNumber'])+"/"

#Validation parameters
ValConf = dict()
ValConf['NBins']  = 50  #Number of histogram bins
ValConf['Thresh'] = [1,5,10,20,50,75,100] #Thresholds for categorical verifcation
ValConf['PlotExtreme'] = True
ValConf['NumExtreme']  = 10


#Obtenemos la configuracion del experimento
with open( ExpConf['ExpPath'] + 'ExpConf.pkl', 'rb') as handle:
    Conf = pickle.load(handle)
    
#Obtenemos el modelo
TrainedModel = models.load_model( ExpConf['ExpPath'] )  

#Obtenemos el conjunto de datos (dataloaders)
Data = ds.get_data( Conf ) 

#Aplicamos el modelo al conjunto de testiong
input , target , output = tu.model_eval( TrainedModel , Data['TestDataSet'] , denorm = True )
output=output.detach().numpy()
target=target.detach().numpy()
input = input.detach().numpy()

#Computation of categorical indices
CmInd = ver.confusion_matrix_indices( output , target , ValConf['Thresh'] )  
#Plot categorical indices
plots.PlotCatInd( CmInd , ExpConf['ExpPath'] )

#Plot the most extreme cases
mtarget = np.mean( np.mean( target , 2 ) , 1 )                      #Compute the mean precipitation
sortind = np.flip( np.argsort( mtarget ) )[0:ValConf['NumExtreme']] #Select the Nth most extreme cases (based on the mean precipitation)
eoutput = output[sortind,:,:]
einput  = input[sortind,:,:]
etarget = target[sortind,:,:]
plots.PlotCases( einput , etarget , eoutput , ExpConf['ExpPath'] )


