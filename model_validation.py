import numpy as np
import plots
import set_dataset as ds
import pickle
import models
import verificacion as ver
import train_utils as tu

ExpConf = dict()
#Exp type
ExpConf['ExpName']    = '/EARLY_STOP/UNET_MEDIOS/MULTI-UNET'
ExpConf['ExpNumber']  = 0
ExpConf['ExpPath'] = "../experiments/"+ ExpConf['ExpName'] +"_"+str(ExpConf['ExpNumber'])+"/"
#Validation parameters
ValConf = dict()
ValConf['NBins']  = 50  #Number of histogram bins
ValConf['Thresh'] = [1,5,10,20,50,75,100] #Thresholds for categorical verifcation
ValConf['PlotExtreme'] = True
ValConf['NumExtreme']  = 10


#Obtenemos la configuracion del experimento
with open( ExpConf['ExpPath'] + 'TrainConf.pkl', 'rb') as handle:
    Conf = pickle.load(handle)

Conf['TempFix'] = True #This is a temporal fix to fix a small change in the size of the data among experiments.

#Obtenemos el modelo
TrainedModel = models.load_model( ExpConf['ExpPath'] )  

#Obtenemos el conjunto de datos (dataloaders)
Data = ds.get_data( Conf ) 

#Aplicamos el modelo al conjunto de testing
minput , target , output = tu.model_eval( TrainedModel , Data['TestDataSet'] , denorm = True )
output=output.detach().numpy()
target=target.detach().numpy()
minput = minput.detach().numpy()

print( np.max( output ) , np.min( output ) , np.mean( output ))

#Computation of categorical indices
CmInd = ver.confusion_matrix_indices( output , target , ValConf['Thresh'] )  
#Plot categorical indices
plots.PlotCatInd( CmInd , ExpConf['ExpPath'] )

#Plot the most extreme cases
mtarget = np.mean( np.mean( target , 2 ) , 1 )                      #Compute the mean precipitation
sortind = np.flip( np.argsort( mtarget ) )[0:ValConf['NumExtreme']] #Select the Nth most extreme cases (based on the mean precipitation)
eoutput = output[sortind,:,:]
einput  = minput[sortind,:,:]
etarget = target[sortind,:,:]
plots.PlotCases( einput , etarget , eoutput , ExpConf['ExpPath'] )


