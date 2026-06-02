import torch.nn as nn
from . import element
import collections.abc
from itertools import repeat

nside = 512

class SeqName:
    def __init__(self, module_name):
        ''' define the name of sequence, to be used by class LinearSeq & Conv2dSeq '''
        self.moduleName = int(module_name)
    def seq_name(self):
        self.moduleName += 1
        return str(self.moduleName)

class BatchNorm(object):
    ''' define Batch Normalization, to be used by class LinearSeq & Conv2dSeq '''
    def _batchnorm2d(self, name, out_channel):
        self.seq.add_module(name, nn.BatchNorm2d(out_channel, eps=self.eps, momentum=self.momentum))

class Activation(object):
    ''' define activation functions, to be used by class LinearSeq & Conv2dSeq '''
    def _activation(self, module_name, active_name):
        self.seq.add_module(module_name, element.get_activation(active_name=active_name))

class Pooling(object):
    ''' define pooling, to be used by class LinearSeq & Conv2dSeq '''
    def _pooling(self, module_name, pool_name):
        self.seq.add_module(module_name, element.get_pooling(pool_name=pool_name))


def _ntuple(x, n=2):
    '''
    return a tuple
    
    :param x: an integer or a tuple with more than two elements
    :param n: the number to be repeated for an integer, it only works for an integer
    '''
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return x
    return tuple(repeat(x, n))

def multi_ntuple(x, n=2):
    '''
    return a tuple or a list that contain tuples
    
    :param x: an integer, a tuple or a list whose element is tuple (with more than two elements)
    :param n: the number to be repeated for an integer, it only works for an integer
    '''
    if isinstance(x, list):
        return [_ntuple(i, n=n) for i in x]
    return _ntuple(x, n=n)


class Conv2dSeq(SeqName,BatchNorm,Activation,Pooling):
    def __init__(self, channels, kernels_size=None, strides=None, extra_pads=None, mainBN=True, finalBN=True,
                 mainActive='relu', finalActive='relu', mainPool='None', finalPool='None', eps=1e-05, momentum=0.1, transConv2d=False,
                 upSample=False, upSample_mode='nearest', unPool=False, in_side=nside):
        ''' sequence of Conv2d or ConvTranspose2d '''
        
        super(Conv2dSeq, self).__init__('-1') #or SeqName.__init__(self, '-1')
        self.channels = channels
        self.layers = len(channels) - 1
        # Inicialización de hiperparámetros
        self.kernels_size = multi_ntuple(kernels_size) if kernels_size else [(3,3)] * self.layers
        self.strides = multi_ntuple(strides) if strides else [(2,2)] * self.layers
        self.extra_pads = multi_ntuple(extra_pads) if extra_pads else [(0,0)] * self.layers
        self.dilation = 1
        self.mainBN, self.finalBN = mainBN, finalBN
        self.mainActive, self.finalActive = mainActive, finalActive
        self.mainPool, self.finalPool = mainPool, finalPool
        self.eps, self.momentum = eps, momentum
        self.bias = True
        self.transConv2d = transConv2d
        self.upSample, self.unPool = upSample, unPool
        self.upSample_mode = upSample_mode
        self.sides = [_ntuple(in_side)]
        self.seq = nn.Sequential()
    
    def getPadding(self, extra_pad, kernel_size, stride, dilation):
        '''
        obtain the padding or output_padding for Conv2d and ConvTranspose2d when giving kernel size, stride, and extra_pad
        '''
        # Aumentamos el rango de búsqueda de padding por seguridad en redes profundas
        pads = range(16)
        #note that output padding must be smaller than either stride or dilation
        out_pads = range(min(stride, dilation))
        if self.transConv2d:
            for out_pad in out_pads:
                for pad in pads:
                    if kernel_size - stride - 2*pad + out_pad == extra_pad:
                        return pad, out_pad
        else:
            for pad in pads:
                # Fórmula estándar de salida de convolución de PyTorch
                if (2*pad - dilation*(kernel_size-1) - 1) // stride + 1 == extra_pad:
                    return pad
            print ('Warning: No padding matching !!!')
            return 0
    
    def __conv2d(self, name, in_channel, out_channel, kernel_size, stride, extra_pad):
        '''
        The default settings are 'extra_pad=0' and 'dilation=1', so, the size of channel will be reduced by half when using 'Conv2d',
        and will be increase to two times of the original size when using 'ConvTranspose2d'.
        
        For Conv2d, the output size is: H_out = H_in/stride + (2*padding - dilation*(kernel_size-1) -1)/stride + 1,
        for transConv2d, the output size is: H_out = H_in*stride + kernel_size-stride-2*padding + output_padding
        
        :param extra_pad: extra_pad is defined as "(2*padding - dilation*(kernel_size-1) -1)/stride + 1" for Conv2d,
                          and "kernel_size-stride-2*padding + output_padding" for transConv2d
        '''
        k, s, ep = _ntuple(kernel_size), _ntuple(stride), _ntuple(extra_pad)

        if self.transConv2d:
            res0 = self.getPadding(ep[0], k[0], s[0], self.dilation)
            res1 = self.getPadding(ep[1], k[1], s[1], self.dilation)
            p0, op0 = res0 if isinstance(res0, tuple) else (res0, 0)
            p1, op1 = res1 if isinstance(res1, tuple) else (res1, 0)
            padding, output_padding = (p0, p1), (op0, op1)
            # Cálculo de dimensiones de salida (H_out = (H_in-1)*stride + kernel - 2*pad + out_pad)
            side_H = (self.sides[-1][0]-1)*s[0] + k[0] - 2*padding[0] + output_padding[0]
            side_W = (self.sides[-1][1]-1)*s[1] + k[1] - 2*padding[1] + output_padding[1]
            self.seq.add_module(name, nn.ConvTranspose2d(in_channel, out_channel, k, s, padding, output_padding, bias=self.bias))
        else:
            p0 = self.getPadding(ep[0], k[0], s[0], 1)
            p1 = self.getPadding(ep[1], k[1], s[1], 1)
            padding = (p0, p1)
            side_H = self.sides[-1][0]//s[0] + (2*padding[0] - (k[0]-1) - 1)//s[0] + 1
            side_W = self.sides[-1][1]//s[1] + (2*padding[1] - (k[1]-1) - 1)//s[1] + 1
            self.seq.add_module(name, nn.Conv2d(in_channel, out_channel, k, s, padding, bias=self.bias))
        self.sides.append((side_H, side_W))

    def get_seq(self):
        for i in range(self.layers):
            # Check if it's the last layer
            is_final = (i == self.layers - 1)
            # Gestión de UpSampling (Si se activa antes de la conv/deconv)
            if self.upSample:
                self.seq.add_module(self.seq_name(), nn.Upsample(scale_factor=2, mode=self.upSample_mode))
                self.sides.append((self.sides[-1][0] * 2, self.sides[-1][1] * 2))
            if self.unPool:
                self.seq.add_module(self.seq_name(), nn.MaxUnpool2d(kernel_size=2))
                self.sides.append((self.sides[-1][0] * 2, self.sides[-1][1] * 2))
            # Capa de Convolución o Deconvolución
            self.__conv2d(self.seq_name(), self.channels[i], self.channels[i+1], 
                          self.kernels_size[i], self.strides[i], self.extra_pads[i])
            # Batch Normalization
            bn = self.finalBN if is_final else self.mainBN
            if bn: 
                self._batchnorm2d(self.seq_name(), self.channels[i+1])
            # Activación
            act = self.finalActive if is_final else self.mainActive
            if act.lower() != 'none': 
                self._activation(self.seq_name(), act)
            # Pooling (Solo si no es una capa de de-convolución usualmente)
            pool = self.finalPool if is_final else self.mainPool
            if pool.lower() != 'none': 
                self._pooling(self.seq_name(), pool)
                # Actualizamos la dimensión en la lista sides
                self.sides[-1] = (self.sides[-1][0] // 2, self.sides[-1][1] // 2)
        return self.seq