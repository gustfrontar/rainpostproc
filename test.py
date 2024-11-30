import numpy as np
import plots
import models
import set_dataset as ds
import pickle
import train_utils as tu
import os
import default_conf as dc
import copy
import multiprocessing as mp

os.environ['OMP_NUM_THREADS']="4"
max_proc=1

TrainConf = dc.TrainConf #Get default configuration.

#Exp type
TrainConf['ModelClass'] = models.unet  #Determino el modelo a utilizar.
TrainConf['ExpName']    = 'TEST_TEST'

#####################################################
## Parameters to be tested >
TestParameters = dict()
RandomSeed   = [1029]  #As many random seed as initailization experiments we want to perform.
TestParameters['BatchSize'] = [100 ]           #As many batch sizes as we want to test
TestParameters['LearningRate'] = [1.0e-3 ]     #As many learning rates as we want to test
TestParameters['WeightDecay']  = [0.0 ]        #As many Weight decay rates as we want to test
TestParameters['KernelSize']   = [3 ]
TestParameters['Pool']         = [2 ]
TestParameters['BatchNorm']    = [False]       #Wether batch normalization will be applied or not.
TestParameters['Bias']         = [False]
TestParameters['OutActivation']= ['Identity'] #['Softplus']
TestParameters['Channels']     = [8]           #Number of filters in the first convolution. (the others will be increased by a factor of 2)

#TrainConf['MaxEpochs']= 1

#Build the base configuration 
ParameterList = TestParameters.keys()
TrainConfList = []
for ipar , mypar in enumerate( ParameterList ) :
    if mypar in TrainConf['ModelConf'].keys() :
        TrainConf['ModelConf'][mypar] = TestParameters[mypar][0]
    else :
        TrainConf[mypar] = TestParameters[mypar][0]

#Build the sequence of configuration dictionaries that will be pased to the meta_model_training function

ExpNumber = 0 
for ipar , mypar in enumerate( ParameterList ) :
    ParameterValueList = TestParameters[ mypar ]

    for iparval , myparval in enumerate( ParameterValueList ) :
 
        if ( iparval >= 1 or ipar == 0 ) :   #Avoid runing the base configuration several times.
            for myseed in RandomSeed :
                TrainConfList.append( copy.deepcopy( TrainConf ) )
                #print( TrainConf['ModelConf']['Pool'] )
                if mypar in TrainConfList[-1]['ModelConf'].keys() :
                    TrainConfList[-1]['ModelConf'][ mypar ] = myparval
                else :
                    TrainConfList[-1][ mypar ] = myparval 
                TrainConfList[-1]['RandomSeed'] = myseed
                TrainConfList[-1]['HyperParameter'] = mypar   #Store the hyper parameter being explored
                TrainConfList[-1]['ExpNumber'] = 1 #ExpNumber
                ExpNumber = ExpNumber + 1 

########################################################################################################

#print( TrainConfList[1] )
#print( len( TrainConfList ) )
#def dummy( input ) :
#    print(input['ModelConf']['Pool'] , input['ModelConf']['KernelSize'])
#    return 0

for my_conf in TrainConfList :
    tu.meta_model_train( my_conf )


#print(' We will perform ' + str( len(TrainConfList) ) + ' experiments ')   
#pool = mp.Pool( min( max_proc , len( TrainConfList ) ) )
#pool.map( tu.meta_model_train , TrainConfList ) 

#tu.meta_model_train( TrainConfList[0] )
#pool.map( dummy , TrainConfList )

#pool.close()



