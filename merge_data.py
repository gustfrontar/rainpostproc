import numpy as np

data=np.load( '../data/datos_train_val.npz' )

input_data = data['input_train'][...]

input_data = np.concatenate( (input_data , data['input_val']) , axis = 0 )


target_data = data['target_train'][...]

target_data = np.concatenate( (target_data , data['target_val']) , axis = 0 )

np.savez_compressed( '../data/datosRRQPE_DPR.npz' , RRQPE=input_data , DPR=target_data )


