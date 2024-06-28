import numpy as np
import set_dataset as ds
from scipy.stats import spearmanr

def rmse( modeldata , targetdata ) :
    return np.sqrt( np.mean( (modeldata.flatten() - targetdata.flatten()) ** 2 ) )

def bias( modeldata , targetdata ) :
    return np.mean( modeldata.flatten() - targetdata.flatten() )

def corr_P( modeldata , targetdata ) :    
    return np.corrcoef( modeldata.flatten() , targetdata.flatten() )[0,1]

def corr_S( modeldata , targetdata ) :    
    return spearmanr( modeldata.flatten() , targetdata.flatten() )[0]


def confusion_matrix( output , target , thresh ) :
    NTre = len( thresh ) 
    cm = np.zeros((2,2,NTre))  #Initialize confusion matrix
    output = output.flatten()
    target = target.flatten()
    
    #Loop over the thresholds
    for it in range( NTre )  :
        cm[0,0,it] = np.sum( ( output >= thresh[it] ) & ( target >= thresh[it] ) )
        cm[0,1,it] = np.sum( ( output <= thresh[it] ) & ( target >= thresh[it] ) ) 
        cm[1,0,it] = np.sum( ( output >= thresh[it] ) & ( target <= thresh[it] ) )
        cm[1,1,it] = np.sum( ( output <= thresh[it] ) & ( target <= thresh[it] ) ) 
    #Normalization
    cm = cm / np.sum( cm )
    
        
    return cm 


def confusion_matrix_indices( output , target , thresh )  :
    #https://journals.ametsoc.org/view/journals/wefo/29/4/waf-d-13-00087_1.xml
    cm = confusion_matrix( output , target , thresh )
    NTre = cm.shape[2]
    O = cm[0,0,:] + cm[0,1,:]  #Observed frequency. 
    F = cm[0,0,:] + cm[1,0,:]  #Forecasted frequency.
    N = cm[0,0,:] + cm[0,1,:] + cm[1,0,:] + cm[1,1,:]
    H = cm[0,0,:]
    HN= cm[1,1,:]
    R = F * ( O / N )                     #Random hits

    CMInd = dict()
    CMInd['Thresholds'] = thresh
    CMInd['TS'] = H / ( O + F - H )                #Threat score
    CMInd['ETS'] = ( H - R ) / ( O + F - H - R )   #Equitable threat score
    CMInd['FB']  = F / O                           #Frequency Bias
    CMInd['ACC'] = ( H + HN ) / N                  #Accuracy
    CMInd['POD'] = H / O                           #Probability of detection
    CMInd['FAR'] = ( F - H ) / F                   #False alarm ratio
    CMInd['CM']  = cm                              #Confusion matrix
    
    return CMInd  
    
    


