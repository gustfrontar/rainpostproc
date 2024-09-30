import torch
import os
import numpy as np

def save_model( model , path ) :
    
    torch.save( model , path + "/model.pth")

def load_model( path ) :
    model = torch.load( path + '/model.pth' )     
    return model     

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


class Linear(torch.nn.Module):
    def __init__(self, model_conf ):
        super().__init__()
        nx = model_conf['Nx']
        ny = model_conf['Ny']
        bias = model_conf['Bias']
        self.nx, self.ny, self.dim = nx, ny, nx*ny
        self.Linear_1 = torch.nn.Linear(int(self.dim), int(self.dim), bias=bias)
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.unflatten = torch.nn.Unflatten(1,(self.nx,self.ny))

    def forward(self, x):
        x = self.flatten(x)
        x = self.Linear_1(x)
        return self.unflatten(x)
    
class fully_connected(torch.nn.Module):
    def __init__(self, model_conf ):
        super().__init__()
        self.nx = model_conf['Nx']
        self.ny = model_conf['Ny']
        self.dim = self.nx*self.ny
        self.bias = model_conf['Bias']
        self.layer1 = torch.nn.Linear(  int(self.dim), 12*int(self.dim), bias=self.bias)
        self.layer2 = torch.nn.Linear(12*int(self.dim), 6*int(self.dim), bias=self.bias)
        self.layer3 = torch.nn.Linear(6*int(self.dim),   int(self.dim), bias=self.bias)
        
        self.act = torch.nn.ReLU()
        #self.activation_2 = nn.SiLU()
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.unflatten = torch.nn.Unflatten(1,(self.nx,self.ny))
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)
        return self.unflatten(x)    
    


class Convolucional(torch.nn.Module):

    def __init__(self, model_conf  ):
        super().__init__()
        self.nx, self.ny = model_conf['Nx'], model_conf['Ny']

        if 'OutActivation' in model_conf.keys() :   
           self.out_act_type = model_conf['OutActivation']
        else :
           self.out_act_type = None
        if 'InActivation' in model_conf.keys() :   
           self.act_type = model_conf['InActivation']
        else :
           self.act_type = 'ReLU'
        if 'Channels' in model_conf.keys() :
            self.channel = model_conf['Channels']
        else :
            print('Warning: channels are not properly set, using default number of layers and filters')
            self.channel = [1,8,8,8,1] #[self.nc,32,64,128]
        self.nlayer = len(self.channel)-1
        
        if 'Bias' in model_conf.keys() and len(model_conf['Bias'] ) == self.nlayer :
            self.bias = model_conf['Bias']
        else :
            print('Warning: Bias is not properly set, turning off bias for all layers')
            self.bias = False
        if 'KernelSize' in model_conf.keys() and len(model_conf['KernelSize'] ) == self.nlayer :
            self.kernel_size = model_conf['KernelSize']
        else :
            print('Warning: Using default kernel sizes')
            self.kernel_size = 3
        
        self.padding = int( ( self.kernel_size - 1 ) / 2 )     
        
        #Genero en un loop la red convolucional con el numero solicitado de capas.     
        self.model_layers = torch.nn.ModuleList()
        for ii in range( self.nlayer ) :
            self.model_layers.append( torch.nn.Conv2d( self.channel[ii],self.channel[ii+1], kernel_size=self.kernel_size ,stride = 1, padding=self.padding , padding_mode='reflect' , bias = self.bias ) )
        exec('self.activation = torch.nn.' + self.act_type + '()' )
        if not self.out_act_type is None :
           exec('self.activation_out = torch.nn.' + self.out_act_type + '()' )

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(batch,self.channel[0],self.nx,self.ny) # x - Shape 4D: (batch size, filtros, nx, ny)
        for ii in range( self.nlayer ) :
            x = self.model_layers[ii]( x )
            if ii < self.nlayer - 1 :
               x = self.activation( x )
        if not self.out_act_type is None :
           x = self.activation_out(x)

        return torch.squeeze(x)


class conv_block(torch.nn.Module):
    def __init__(self, in_c, out_c , kernel_size = 3 , padding = 1 , act_type = 'ReLU' , bias = False , batchnorm = False ):
        super().__init__()
        self.batchnorm = batchnorm
        self.conv1 = torch.nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding , padding_mode='reflect' , bias = bias )
        self.bn1 = torch.nn.BatchNorm2d(out_c)
        self.conv2 = torch.nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=padding , padding_mode='reflect' , bias = bias )
        self.bn2 = torch.nn.BatchNorm2d(out_c)
        exec('self.act = torch.nn.' + act_type + '()' )
    def forward(self, inputs):
        x = self.conv1(inputs)
        if batchnorm :
          x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        if batchnorm :
          x = self.bn2(x)
        x = self.act(x)
        return x
    
class encoder_block_skip(torch.nn.Module):
    def __init__(self, in_c, out_c , pool = 2 , kernel_size = 3 , padding = 1 , act_type = 'ReLU' , bias = False , batchnorm = False ):
        super().__init__()
        self.conv = conv_block(in_c, out_c , kernel_size = kernel_size , padding = padding , act_type = act_type , bias = bias , batchnorm = batchnorm )
        self.pool = torch.nn.MaxPool2d((pool, pool))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class encoder_block(torch.nn.Module):
    def __init__(self, in_c, out_c , pool = 2 , kernel_size = 3 , padding = 1 , act_type = 'ReLU' , bias = False , batchnorm = False ):
        super().__init__()
        self.conv = conv_block(in_c, out_c , kernel_size = kernel_size , padding = padding , act_type = act_type , bias = bias , batchnorm = batchnorm )
        self.pool = torch.nn.MaxPool2d((pool, pool))
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        return x

class decoder_block(torch.nn.Module):
    def __init__(self, in_c, out_c , kernel_size = 3 , padding = 1 , decotype = 'bilinear' , act_type = 'ReLU' , bias = False , batchnorm = False ) :
        super().__init__()
        self.decotype = decotype
        self.upsample = torch.nn.Upsample        

        self.conv = conv_block( in_c , out_c , kernel_size = kernel_size , padding = padding , act_type = act_type , bias = bias , batchnorm = batchnorm )
        
    def forward(self, inputs , targetnx , targetny ) :
        self.up = self.upsample( size=(targetnx,targetny) , mode=self.decotype)
        x = self.up(inputs)
        x = self.conv(x)
        return x  

class decoder_block_skip(torch.nn.Module):
    def __init__(self, in_c, out_c , kernel_size = 3 , padding = 1 , decotype = 'bilinear' , act_type = 'ReLU' , bias = False , batchnorm = False ):
        super().__init__()
        self.upsample = torch.nn.Upsample
        self.decotype = decotype 

        self.conv = conv_block( out_c + in_c , out_c , kernel_size = kernel_size , padding = padding , act_type = act_type , bias = bias , batchnorm = batchnorm )
        
    def forward(self, inputs, skip):
        skipnx=skip.size()[2];skipny=skip.size()[3]
        #upsample layer is defined to conform the target size of the skip tensor. 
        self.up = self.upsample( size=(skipnx,skipny) , mode=self.decotype )
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x  
    
class EncoderDecoder(torch.nn.Module):
    #Este modelo se parece al implementado en Persiann CNN
    def __init__(self, model_conf ):
        super().__init__()
        self.nx, self.ny = model_conf['Nx'] , model_conf['Ny']
        if 'InActivation' in model_conf.keys() :
           self.act_type = model_conf['InActivation']
        else :
           self.act_type = 'ReLU'
        if 'OutActivation' in model_conf.keys() :
           self.out_act_type = model_conf['OutActivation']
        else :
           self.out_act_type = 'Identity'
        if 'KernelSize' in model_conf.keys() :
            self.kernel_size = model_conf['KernelSize']
        else :
            print('Warning: Using default kernel sizes')
            self.kernel_size = 3
        if 'Pool' in model_conf.keys() :
            self.pool = model_conf['Pool']
        else :
            self.pool = 2
        if 'Bias' in model_conf.keys() :
            self.bias = model_conf['Bias']
        else :
            self.bias = False 
        if 'DecoType' in model_conf.keys() :
            self.decotype = model_conf['DecoType']
        else :
            self.decotype = 'bilinear'
        if 'Channels' in model_conf.keys() :
            self.channels = model_conf['Channels']
        else :
            self.channels=[1,16,16,32,64,128,64,32,16,1]
        
        self.padding = int( ( self.kernel_size - 1 ) / 2 )
        exec('self.act     = torch.nn.' + self.act_type + '()' )
        exec('self.out_act = torch.nn.' + self.out_act_type + '()' )


        self.inputs = torch.nn.Conv2d( self.channels[0] , self.channels[1] , kernel_size = self.kernel_size , padding = self.padding , padding_mode = 'reflect' , bias = self.bias )
        """ Encoder """
        self.e1 = encoder_block( self.channels[1] , self.channels[2] , kernel_size = self.kernel_size , padding = self.padding , pool = self.pool , bias = self.bias , act_type = self.act_type )
        self.e2 = encoder_block( self.channels[2] , self.channels[3] , kernel_size = self.kernel_size , padding = self.padding , pool = self.pool , bias = self.bias , act_type = self.act_type)
        self.e3 = encoder_block( self.channels[3] , self.channels[4] , kernel_size = self.kernel_size , padding = self.padding , pool = self.pool , bias = self.bias , act_type = self.act_type)
        """ Bottleneck """
        self.b = conv_block( self.channels[4] , self.channels[5] , kernel_size = self.kernel_size , padding = self.padding , bias = self.bias , act_type = self.act_type )

        """ Decoder """
        self.d1 = decoder_block( self.channels[5] , self.channels[6] , decotype = self.decotype , kernel_size = self.kernel_size , padding = self.padding , bias = self.bias , act_type = self.act_type )
        self.d2 = decoder_block( self.channels[6] , self.channels[7] , decotype = self.decotype , kernel_size = self.kernel_size , padding = self.padding , bias = self.bias , act_type = self.act_type )
        self.d3 = decoder_block( self.channels[7] , self.channels[8] , decotype = self.decotype , kernel_size = self.kernel_size , padding = self.padding , bias = self.bias , act_type = self.act_type )
        """ Output Layer """
        self.output = torch.nn.Conv2d( self.channels[8] , self.channels[9] , kernel_size= self.kernel_size , padding= self.padding , bias = self.bias )
        
    def forward(self, x):
        batch = x.shape[0]
        x = x.view(batch,1,self.nx,self.ny) # x - Shape 4D: (batch size, filtros, nx, ny)
        """ Input Layer """
        x     = self.inputs( x )
        x     = self.act(x)
        """ Encoder """
        nx0 , ny0 = x.shape[2] , x.shape[3]
        x = self.e1(x)
        nx1 , ny1 = x.shape[2] , x.shape[3]
        x = self.e2(x)
        nx2 , ny2 = x.shape[2] , x.shape[3]
        x = self.e3(x)
        """ Bottleneck """
        x = self.b(x)
        """ Decoder """
        x = self.d1(x,nx2,ny2)
        x = self.d2(x,nx1,ny1)
        x = self.d3(x,nx0,ny0)
        """ Output Layer """
        x = self.output(x)
        x = self.out_act(x)

        return x.view(batch,self.nx,self.ny)

        #return torch.squeeze(x)    
          
class unet(torch.nn.Module):
    def __init__(self, model_conf ):
        super().__init__()
        self.nx = model_conf['Nx']
        self.ny = model_conf['Ny']
        if 'InActivation' in model_conf.keys() :
           self.act_type = model_conf['InActivation']
        else :
           self.decotype = 'ReLU'
        if 'OutActivation' in model_conf.keys() :
           self.out_act_type = model_conf['OutActivation']
        else :
           self.out_act_type = 'Identity'
        if 'DecoType' in model_conf.keys() :
           self.decotype = model_conf['DecoType']
        else :
           self.decotype = 'bilinear'
        if 'Pool' in model_conf.keys() :
           self.pool = model_conf['Pool']
        else :
           self.pool = 2
        if 'KernelSize' in model_conf.keys() :
           self.kernel_size = model_conf['KernelSize'] 
        else :
           self.kernel_size = 3
        if 'Bias' in model_conf.keys() :
           self.bias = model_conf['Bias']
        else :
           self.bias = False 
        if 'BatchNorm' in model_conf.keys()
           self.batchnorm = model_conf['BatchNorm']
        else :
           self.batchnorm = False

        self.padding = int( ( self.kernel_size  - 1 ) / 2 )
        exec('self.act     = torch.nn.' + self.act_type + '()' )
        exec('self.out_act = torch.nn.' + self.out_act_type + '()' )
        """ Input Layer """
        self.inputs = torch.nn.Conv2d( 1 , 16 , kernel_size = self.kernel_size , padding = self.padding , padding_mode = 'reflect' , bias = self.bias )
        self.inputbn = torch.nn.BatchNorm2d( 16 ) 
        """ Encoder """
        self.e1 = encoder_block_skip(16, 16 , kernel_size = self.kernel_size , padding = self.padding , pool = self.pool , bias = self.bias , batchnorm = self.batchnorm )
        self.e2 = encoder_block_skip(16, 32 , kernel_size = self.kernel_size , padding = self.padding , pool = self.pool , bias = self.bias , batchnorm = self.batchnorm )
        self.e3 = encoder_block_skip(32, 64 , kernel_size = self.kernel_size , padding = self.padding , pool = self.pool , bias = self.bias , batchnorm = self.batchnorm )
        """ Bottleneck """
        self.b = conv_block(64, 128 , kernel_size = self.kernel_size , padding = self.padding , bias = self.bias )
        
        """ Decoder """
        self.d1 = decoder_block_skip(128, 64 , decotype = self.decotype , kernel_size = self.kernel_size , padding = self.padding , bias = self.bias , batchnorm = self.batchnorm )
        self.d2 = decoder_block_skip(64, 32 , decotype = self.decotype , kernel_size = self.kernel_size , padding = self.padding , bias = self.bias , batchnorm = self.batchnorm )
        self.d3 = decoder_block_skip(32, 16 , decotype = self.decotype , kernel_size = self.kernel_size , padding = self.padding , bias = self.bias , batchnorm = self.batchnorm )
        """ Output Layer """
        self.output = torch.nn.Conv2d(32, 1, kernel_size= self.kernel_size , padding= self.padding , bias = self.bias )

    def forward(self, x ):
        batch = x.shape[0]
        x = x.view(batch,1,self.nx,self.ny) # x - Shape 4D: (batch size, filtros, nx, ny)
        """ Input Layer """
        x     = self.inputs( x )
        if self.batchnorm :
           x = self.inputbn( x )
        i1    = torch.clone( x )
        x     = self.act(x)
        """ Encoder """
        s1, x = self.e1(x)
        s2, x = self.e2(x)
        s3, x = self.e3(x)
        """ Bottleneck """
        x = self.b(x)
        """ Decoder """
        x = self.d1(x, s3)
        x = self.d2(x, s2)
        x = self.d3(x, s1)
        """ Output Layer """
        x = torch.cat([x, i1], axis=1)
        x = self.output(x)
        x = self.out_act(x) 
        

        return x.view(batch,self.nx,self.ny)
