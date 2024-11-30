import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LinearSegmentedColormap, BoundaryNorm, Normalize
import verificacion as ver

def plotting(Dir_plot, Input, Target, Modelo,
             loss_train, loss_val,
             RMSE, BIAS, Corr_P, Corr_S,
             Experimento, Input_name, Target_name,
             max_epochs, samples, nx ,ny):
    
    plot_losses(Dir_plot, max_epochs, loss_train, loss_val)
    FreqLogHist(Dir_plot, Target , Modelo, Experimento, Target_name)
    Target_vs_Modelo(Dir_plot, Modelo, Target, Experimento, Target_name)
    Input_vs_Target_vs_Modelo(Dir_plot, Input, Target, Modelo, Input_name, Target_name)
    plot_ejemplos(Dir_plot, Input, Target, Modelo, Experimento, Input_name, Target_name, samples, nx, ny)
    Mean_plot( Dir_plot, Input , Target , Modelo, Experimento, Input_name, Target_name, nx , ny)
    plot_scores(Dir_plot, RMSE, BIAS, Corr_P, Corr_S, max_epochs)
    plt.close()

        
def FreqLogHist(Dir_plot, Target , Modelo , Experimento, Target_name) :
    
    vmin_target,vmax_target,_,_,target_label,unit_target,_,_,_,_,_,bins,dticks_target = setplots_target(Target_name)
        
    hist_Modelo , _ = np.histogram( Modelo.flatten() , bins )
    hist_Target , _ = np.histogram( Target.flatten() , bins )
        
    hist_Modelo = hist_Modelo / np.sum( hist_Modelo )
    hist_Target = hist_Target / np.sum( hist_Target )

    plt.figure()
    plt.plot( bins[0:-1] , np.log( hist_Target) , '-b' , label=Target_name+" (Target)" )
    plt.plot( bins[0:-1] , np.log( hist_Modelo ) , '-r' , label='Modelo' )
    plt.xticks(np.arange(vmin_target,vmax_target+dticks_target,dticks_target))
    plt.yticks(np.arange(-15,0+1,1))
    plt.ylabel("Log(Frecuencia)")
    plt.xlabel(target_label+" "+unit_target)
    plt.title('Frecuencia de '+target_label+'(Modelo) vs '+Target_name+'   Exp: '+Experimento)
    plt.legend()
    plt.grid()

    plt.savefig(Dir_plot+"FreqLog_Histograma.png",dpi=100,bbox_inches='tight')
    plt.close()
   
def Target_vs_Modelo(Dir_plot, ModelData , TargetData, Experimento, Target_name) :
    _,_,_,_,target_label,unit_target,_,_,_,_,_,_,dticks_target = setplots_target(Target_name)
    vmin_target, vmax_target = TargetData.min(), TargetData.max()
    fig = plt.figure()
    plt.hexbin( TargetData.flatten() , ModelData.flatten() ,cmap = 'gist_ncar_r',bins='log',gridsize=50,extent=(vmin_target,vmax_target,vmin_target,vmax_target))
    plt.xlabel(target_label+" "+unit_target) ; plt.ylabel(target_label+" "+unit_target)
    plt.title('Scatterplot Target vs Modelo  - Exp: '+Experimento+
              '\nRMSE: '+str(round(ver.rmse(TargetData.flatten(),ModelData.flatten()),2))+
              '   BIAS: '+str(round(ver.bias(TargetData.flatten(),ModelData.flatten()),2))+
              '   Corr P: '+str(round(ver.corr_P(TargetData.flatten(),ModelData.flatten()),2))
              )
    plt.plot([vmin_target, vmax_target], [vmin_target, vmax_target])
    plt.xticks(np.arange(vmin_target,vmax_target+dticks_target,dticks_target))
    plt.yticks(np.arange(vmin_target,vmax_target+dticks_target,dticks_target))
    plt.xlabel("Modelo "+unit_target)
    plt.grid()

    cbar = plt.colorbar(orientation="vertical")
    cbar.set_label("Frecuencia")
    cbar.set_ticks([1,10,100,1000,10000,100000])
    cbar.set_ticklabels([r'10$^0$',r'10$^1$',r'10$^2$',r'10$^3$',r'10$^4$',r'10$^5$'])
    plt.savefig(Dir_plot+"Scatterplot_Target_Modelo.png",dpi=100,bbox_inches='tight')
    plt.close()

def Input_vs_Target_vs_Modelo(Dir_plot,Input, Target, Modelo , Input_name, Target_name) :

    _,_,_,_,input_label,unit_input,_,_,_,_,_,dticks_input = setplots_input(Input_name)
    _,_,_,_,target_label,unit_target,_,_,_,_,_,_,dticks_target = setplots_target(Target_name)
    vmin_input, vmax_input = Input.min(), Input.max()
    vmin_target, vmax_target = Target.min(), Target.max()
    fig = plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.hexbin( Input.flatten(), Target.flatten() ,cmap = 'gist_ncar_r',bins='log',marginals=True,gridsize=50,extent=(vmin_input,vmax_input,vmin_target,vmax_target))
    plt.xlabel(input_label+" "+unit_input);plt.ylabel(target_label+" "+unit_target)
    plt.title('Scatterplot '+input_label+' vs '+target_label+' (Target)')
    plt.xticks(np.arange(vmin_input,vmax_input+dticks_input,dticks_input))
    plt.yticks(np.arange(vmin_target,vmax_target+dticks_target,dticks_target))
    plt.grid()

    cbar = plt.colorbar(orientation="vertical")
    cbar.set_label("Frecuencia")
    cbar.set_ticks([1,10,100,1000,10000,100000])
    cbar.set_ticklabels([r'10$^0$',r'10$^1$',r'10$^2$',r'10$^3$',r'10$^4$',r'10$^5$'])

    plt.subplot(1,2,2)
    plt.hexbin( Input.flatten(), Modelo.flatten() ,cmap = 'gist_ncar_r',bins="log",gridsize=50,extent=(vmin_input,vmax_input,vmin_target,vmax_target))
    plt.xlabel(input_label+" "+unit_input);plt.ylabel(target_label+" "+unit_target)
    plt.title('Scatterplot '+input_label+' vs '+target_label+' (Modelo)')
    plt.xticks(np.arange(vmin_input,vmax_input+dticks_input,dticks_input))
    plt.yticks(np.arange(vmin_target,vmax_target+dticks_target,dticks_target))
    plt.grid()

    cbar = plt.colorbar(orientation="vertical")
    cbar.set_label("Frecuencia")
    cbar.set_ticks([1,10,100,1000,10000,100000])
    cbar.set_ticklabels([r'10$^0$',r'10$^1$',r'10$^2$',r'10$^3$',r'10$^4$',r'10$^5$'])

    plt.savefig(Dir_plot+"Input_vs_Target_vs_Modelo.png",dpi=100,bbox_inches='tight')
    plt.close() 

def plot_ejemplos( Dir_plot, Input , Target , Modelo , Experimento, Input_name, Target_name, samples, nx , ny) :

    vmin_input, vmax_input,_,_, input_label, unit_input, cmap_input, bounds_input, _, norm_input,_,_ = setplots_input(Input_name)
    vmin_target, vmax_target,_,_, target_label, unit_target, cmap_target, bounds_target, _, norm_target,_,_,_ = setplots_target(Target_name)
    
    for ii in samples:
        fig = plt.figure(figsize=(16,4))
        fig.suptitle("Numero de imagen: "+str(ii))

        plt.title("Experimento: "+Experimento)
        plt.subplot(1,3,1)
        cm1 = plt.pcolor( Input[ii,:,:], norm = norm_input, cmap = cmap_input)
        plt.xticks(np.arange(0,nx+5,5))
        plt.yticks(np.arange(0,ny+5,5))
        cbar=plt.colorbar(cm1)
        cbar.set_ticks(bounds_input)
        cbar.set_label(input_label+" "+unit_input)
        plt.title("Input: "+Input_name,fontsize=12)
        plt.grid()

        plt.subplot(1,3,2)
        cm2 = plt.pcolor( Target[ii,:,:], norm = norm_target, cmap = cmap_target)
        plt.xticks(np.arange(0,nx+5,5))
        plt.yticks(np.arange(0,ny+5,5))
        cbar=plt.colorbar(cm2, spacing='uniform')
        cbar.set_ticks(bounds_target)
        cbar.set_label(target_label+" "+unit_target)
        plt.title("Target: "+Target_name,fontsize=12)
        plt.grid()

        plt.subplot(1,3,3)
        cm3 = plt.pcolor( Modelo[ii,:,:], norm = norm_target, cmap = cmap_target)
        plt.xticks(np.arange(0,nx+5,5))
        plt.yticks(np.arange(0,ny+5,5))
        cbar=plt.colorbar(cm3, spacing='uniform')
        cbar.set_ticks(bounds_target)
        cbar.set_label(target_label+" "+unit_target)
        plt.title('Modelo',fontsize=12)
        plt.grid()
                
        plt.savefig(Dir_plot+"Ejemplo_test_"+str(ii)+".png",dpi=100,bbox_inches='tight')
        plt.close()

def Mean_plot( Dir_plot, Input , Target , Modelo, Experimento, Input_name, Target_name, nx , ny ):

    _,_, vmin_mean_input, vmax_mean_input, input_label, unit_input, cmap_input, _, bounds_mean_input, _, norm_mean_input,_ = setplots_input(Input_name)
    _,_, vmin_mean_target, vmax_mean_target, target_label, unit_target, cmap_target, _, bounds_mean_target, _, norm_mean_target, _, _ = setplots_target(Target_name)
    
    fig = plt.figure(figsize=(14,4))
    fig.suptitle("Promedio imagenes")
    plt.title("Experimento: "+Experimento)
    plt.subplot(1,3,1)
    cm_1 = plt.pcolor( np.mean(Input,axis=0), norm = norm_mean_input, cmap = cmap_input)
    plt.xticks(np.arange(0,nx+5,5))
    plt.yticks(np.arange(0,ny+5,5))
    cbar=plt.colorbar(cm_1)
    cbar.set_label(input_label+" "+unit_input)
    cbar.set_ticks(bounds_mean_input)
    plt.title("Input: "+Input_name,fontsize=12)
    plt.grid()

    plt.subplot(1,3,2)
    cm_2 = plt.pcolor( np.mean(Target,axis=0), norm = norm_mean_target, cmap = cmap_target)
    plt.xticks(np.arange(0,nx+5,5))
    plt.yticks(np.arange(0,ny+5,5))
    cbar=plt.colorbar(cm_2)
    cbar.set_label(target_label+" "+unit_target)
    cbar.set_ticks(bounds_mean_target)
    plt.title("Target: "+Target_name,fontsize=12)
    plt.grid()

    plt.subplot(1,3,3)
    cm_3 = plt.pcolor( np.mean(Modelo,axis=0), norm = norm_mean_target, cmap = cmap_target)
    plt.xticks(np.arange(0,nx+5,5))
    plt.yticks(np.arange(0,ny+5,5))
    cbar=plt.colorbar(cm_3)
    cbar.set_label(target_label+" "+unit_target)
    cbar.set_ticks(bounds_mean_target)
    plt.title("Modelo",fontsize=12)
    plt.grid()

    plt.savefig(Dir_plot+"Media_muestras_test.png",dpi=100,bbox_inches='tight')
    plt.close()

def plot_scores(Dir_plot, RMSE, BIAS, Corr_P, Corr_S, max_epochs):

    fig = plt.figure(figsize=(7,7))
    plt.suptitle("Metricas en funcion de la epoca - Validacion")
        
    plt.subplot(2,2,1)
    plt.title("RMSE")
    plt.plot(range(max_epochs), RMSE,color="red")
    plt.grid()

    plt.subplot(2,2,2)
    plt.title("BIAS")
    plt.plot(range(max_epochs), BIAS, color="orange")
    plt.grid()

    plt.subplot(2,2,3)
    plt.title("Correlacion de Pearson")
    plt.plot(range(max_epochs), Corr_P, color="green")
    plt.ylim(0,1)
    plt.xlabel("Epocas")
    plt.grid()
    
    plt.subplot(2,2,4)
    plt.title("Correlacion de Spearman")
    plt.plot(range(max_epochs), Corr_S, color="darkviolet")
    plt.xlabel("Epocas")
    plt.ylim(0,1)
    plt.grid()
        
    plt.savefig(Dir_plot+"Scores_series.png",dpi=100,bbox_inches="tight")
    plt.close()

def setplots_input(Input_name):
    
    if (Input_name == "mdbz") or (Input_name == "sdbz") :
        vmin_input, vmax_input = 0, 70
        vmin_mean_input, vmax_mean_input = 10,20
        input_label, unit_input = "Max dBZ", "[dBZ]"
        cmap,bounds,bounds_mean,norm,norm_mean = pallete(Input_name, vmin_input, vmax_input, vmin_mean_input, vmax_mean_input)
        dticks = 5
        return vmin_input,vmax_input,vmin_mean_input,vmax_mean_input,input_label,unit_input,cmap,bounds,bounds_mean,norm,norm_mean,dticks
    elif Input_name == "ctt":
        vmin_input, vmax_input = 183, 313
        vmin_mean_input, vmax_mean_input = 250, 275 #232 250
        input_label, unit_input = "Cloud Top Temp", "[K]"
        cmap,bounds,bounds_mean,norm,norm_mean = pallete(Input_name, vmin_input, vmax_input, vmin_mean_input, vmax_mean_input)
        dticks = 10
        return vmin_input,vmax_input,vmin_mean_input,vmax_mean_input,input_label,unit_input,cmap,bounds,bounds_mean,norm,norm_mean,dticks
    elif Input_name == "wmax":
        vmin_input, vmax_input = 0, 65
        vmin_mean_input, vmax_mean_input = 0, 3
        input_label, unit_input = "W Max", "[m/s]"
        cmap,bounds,bounds_mean,norm,norm_mean = pallete(Input_name, vmin_input, vmax_input, vmin_mean_input, vmax_mean_input)
        dticks = 5
        return vmin_input,vmax_input,vmin_mean_input,vmax_mean_input,input_label,unit_input,cmap,bounds,bounds_mean,norm,norm_mean,dticks

def setplots_target(Target_name):
    if Target_name == "rain":
        vmin_target, vmax_target = 0, 250
        vmin_mean_target, vmax_mean_target = 3, 13
        target_label, unit_target = "Rain Rate", "[mm/h]"
        cmap,bounds,bounds_mean,norm,norm_mean = pallete(Target_name, vmin_target, vmax_target, vmin_mean_target, vmax_mean_target)
        dticks = 25 ; bins = np.arange(vmin_target,vmax_target+1,1)
        return vmin_target,vmax_target,vmin_mean_target,vmax_mean_target,target_label,unit_target,cmap,bounds,bounds_mean,norm,norm_mean,bins,dticks
    elif Target_name == "mdbz" or Target_name == "sdbz" :
        vmin_target, vmax_target = 0, 70
        vmin_mean_target, vmax_mean_target = 10, 20
        target_label, unit_target = "Max dBZ", "[dBZ]"
        cmap,bounds,bounds_mean,norm,norm_mean= pallete(Target_name, vmin_target, vmax_target, vmin_mean_target, vmax_mean_target)
        dticks = 5 ; bins = np.arange(vmin_target,vmax_target+1,1)
        return vmin_target,vmax_target,vmin_mean_target,vmax_mean_target,target_label,unit_target,cmap,bounds,bounds_mean,norm,norm_mean,bins,dticks
    elif Target_name == "wmax":
        vmin_target, vmax_target = 0, 65
        vmin_mean_target, vmax_mean_target = 0, 3
        target_label, unit_target = "W Max", "[m/s]"
        cmap,bounds,bounds_mean,norm,norm_mean = pallete(Target_name, vmin_target, vmax_target, vmin_mean_target, vmax_mean_target)
        dticks = 5 ; bins = np.arange(vmin_target,vmax_target+1,1)
        return vmin_target,vmax_target,vmin_mean_target,vmax_mean_target,target_label,unit_target,cmap,bounds,bounds_mean,norm,norm_mean,bins,dticks

def pallete(Variable, vmin, vmax, vmin_mean, vmax_mean):
    #mdbz #rainrate #wmax #cape_2d #ctt
    gist_ncar_r = get_cmap('gist_ncar_r', 256)
    gist_ncar = get_cmap('gist_ncar', 256)
    greys = get_cmap('Greys', 256)
    nipy_spectral = get_cmap('nipy_spectral', 256)
    
    if (Variable == "mdbz") or (Variable == "sdbz") :
        
        bounds = [i for i in np.arange(0,70+5,5)]
        style_color = [[0,0,0], [0, 0, 255], [0, 203, 255], [0, 255, 255],
        [0, 255, 191], [0, 255, 0], [127, 255, 0], [255, 255, 0],
        [255, 127, 0], [255, 64, 0], [255, 0, 0], [127, 0, 0],
        [255, 0, 255], [64, 0, 127], [255, 255, 255]]
        
        #Transformo los valores de RGB al rango [0,1]
        color_arr = []
        for color in style_color:
            rgb = [float(value) / 255 for value in color]
            color_arr.append(rgb)
            
        #Normalize bound values
        norm = BoundaryNorm(bounds, ncolors=256)
        
        #Create a colormap
        mdbz_pallete = LinearSegmentedColormap.from_list('rainrate_pallete', color_arr, N = 256)
        bounds_mean = np.arange(vmin_mean,vmax_mean+1,1)
        norm_mean = BoundaryNorm(bounds_mean, ncolors=256)
        return mdbz_pallete, bounds, bounds_mean, norm, norm_mean
    
    if Variable == "rain":
        
        #bounds = [0, 0.1, 0.5, 1, 3, 5, 10, 20, 40, 60, 100, 150, 200, 250] #16 #12
        bounds = [0, 0.1, 0.5, 1, 5, 10, 25, 40, 50, 75, 100,150, 200, 250] 
        style_color = [[0,0,0], [0, 0, 255], [0, 203, 255],
        [0, 255, 0], [255, 255, 0], [255, 127, 0], [255, 0, 0], [127, 0, 0],
        [255, 0, 255], [64, 0, 127], [255, 255, 255]]
        
        # transform color rgb value to 0-1 range
        color_arr = []
        for color in style_color:
            rgb = [float(value) / 255 for value in color]
            color_arr.append(rgb)
            
        #Normalize bound values
        norm = BoundaryNorm(bounds, ncolors=256)
        
        #Create a colormap
        rainrate_pallete = LinearSegmentedColormap.from_list('rainate_pallete', color_arr, N = 256)
        bounds_mean = np.arange(vmin_mean,vmax_mean+0.5,0.5)
        norm_mean = BoundaryNorm(bounds_mean, ncolors=256)
        return rainrate_pallete, bounds, bounds_mean, norm, norm_mean
    
    if Variable == "wmax":
        
        w_max_custom_cmaps = gist_ncar(np.linspace(0, 1, 18))
        w_max_pallete = ListedColormap(w_max_custom_cmaps)
        
        bounds, bounds_mean = np.arange(vmin_mean,vmax_mean+5,5), np.arange(vmin_mean,vmax_mean+0.5,0.5)
        norm = None
        norm_mean = None
        return w_max_pallete, bounds, bounds_mean, norm, norm_mean
    
    if Variable == "cape_2d":
        
        cape_2d_custom_cmaps = nipy_spectral(np.linspace(0, 1, 18))
        cape_2d_pallete = ListedColormap(cape_2d_custom_cmaps)
        
        bounds = np.arange(vmin,vmax+250,250)
        bounds_mean = np.arange(vmin_mean,vmax_mean+200,200)
        norm = None
        norm_mean = None
        return cape_2d_pallete, bounds, bounds_mean, norm, norm_mean
    
    if Variable == "ctt":
        
        ctt_custom_cmaps = gist_ncar_r(np.linspace(0.1, 1, 18))
        ctt_custom_cmaps = np.append(ctt_custom_cmaps, greys(np.linspace(0, 1, 18)), axis = 0)
        ctt_pallete = ListedColormap(ctt_custom_cmaps)
        
        bounds, bounds_mean = np.arange(vmin_mean,vmax_mean+10,10), np.arange(vmin_mean,vmax_mean+3,3)
        norm = None
        norm_mean = None
        return ctt_pallete, bounds, bounds_mean, norm, norm_mean


def PlotModelStats( Stats , OutPath )  :
    
    max_epochs = len( Stats['RMSE'] )
    
    fig = plt.figure(figsize=(7,7))
    plt.suptitle("Metricas en funcion de la epoca - Validacion")
            
    plt.subplot(2,2,1)
    plt.title("RMSE")
    plt.plot(range(max_epochs), Stats['RMSE'] ,color="red")
    plt.grid()

    plt.subplot(2,2,2)
    plt.title("BIAS")
    plt.plot(range(max_epochs), Stats['BIAS'] , color="orange")
    plt.grid()

    plt.subplot(2,2,3)
    plt.title("Correlacion de Pearson")
    plt.plot(range(max_epochs), Stats['CorrP'] , color="green")
    plt.ylim(0,1)
    plt.xlabel("Epocas")
    plt.grid()

    plt.subplot(2,2,4)
    plt.title("Correlacion de Spearman")
    plt.plot(range(max_epochs), Stats['CorrS'] , color="darkviolet")
    plt.xlabel("Epocas")
    plt.ylim(0,1)
    plt.grid()
    
    plt.savefig( OutPath + '/ModelTrainingTestStats.png')
    plt.close()
    
    fig = plt.figure(figsize=(7,7))
    plt.suptitle("Metricas en funcion de la epoca - Validacion")
            
    plt.title("Loss")
    plt.plot(range(max_epochs), Stats['LossVal'] ,color="red" , label='Val')
    plt.plot(range(max_epochs), Stats['LossTrain'] , color="blue" , label='Train')
    plt.grid()
    plt.legend()
    
    plt.savefig( OutPath + '/ModelTrainingLossStats.png')
    plt.close()
        

def PlotCatInd( CmInd , OutPath )  :
    
    
    plt.figure(figsize=(7,7))
    plt.suptitle("Indices categoricos")
            
    plt.subplot(2,2,1)
    plt.title("ETS")
    plt.plot(CmInd['Thresholds'], CmInd['ETS'] ,color="red")
    plt.grid()

    plt.subplot(2,2,2)
    plt.title("Freq. BIAS")
    plt.plot(CmInd['Thresholds'], CmInd['FB'] , color="orange")
    plt.grid()

    plt.subplot(2,2,3)
    plt.title("Probability of detection")
    plt.plot(CmInd['Thresholds'], CmInd['POD'] , color="green")
    plt.ylim(0,1)
    plt.xlabel("Umbrales")
    plt.grid()

    plt.subplot(2,2,4)
    plt.title("False alarm ratio")
    plt.plot(CmInd['Thresholds'], CmInd['FAR'] , color="darkviolet")
    plt.xlabel("Umbrales")
    plt.ylim(0,1)
    plt.grid()
    
    plt.savefig( OutPath + '/ModelTestCategoricalIndices.png')
    plt.close()
    
def PlotCases( input , target , output , outpath , prefix='CasoN_' ) :
    # Se crea una figura y un arreglo de ejes
    NCases = input.shape[0]
    
    for ii in range( NCases ) :
        plt.figure( figsize=(20, 8))    
        plt.suptitle('Comparo salidas',fontsize=25)
        #-------------Input-----
        #Utilizando el metodo pcolormersh se crea un mapa de colores.
        #Transform especifica la proyeccion utilizada
        plt.subplot(1,3,1)
        plt.pcolormesh(input[ii,:,:], cmap='Blues', shading='auto',vmin=0,vmax=100)
        plt.grid()
        plt.title('Input',fontsize=20)
        plt.colorbar()
        #-------------------------Target-----------------
        plt.subplot(1,3,2)
        plt.pcolormesh(target[ii,:,:],cmap='Blues', shading='auto',vmin=0,vmax=100)
        plt.grid()
        plt.title('Target',fontsize=20)
        plt.colorbar()
        #-----------Output-------------
        plt.subplot(1,3,3)
        plt.pcolormesh(output[ii,:,:], cmap='Blues', shading='auto',vmin=0,vmax=100)
        #print(axs)
        plt.grid()
        plt.title('Network',fontsize=20)
        plt.colorbar()
        plt.savefig( outpath + '/' + prefix + str(ii) + '.png' )  
        plt.close()      
    
    
