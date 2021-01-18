'''
Created on Jul 22, 2011

@author: seant
'''
DEBUG = False
import numpy as np
from yapocis.rpc import kernels, interfaces
from yapocis.yapocis_types import *


def histeq(im:Array, nbr_bins=2 ** 16) -> Array:
    '''histogram equalization'''
    #get image histogram
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    imhist[0] = 0
    
    cdf = imhist.cumsum() #cumulative distribution function
    cdf ** .5
    cdf = (2 ** 16 - 1) * cdf / cdf[-1] #normalize
    #cdf = cdf / (2**16.)  #normalize
    
    #use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    
    return np.array(im2, np.float32).reshape(im.shape)


def split_channels(rgb:Array) -> Tuple[Array,Array,Array]:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return r.copy(), g.copy(), b.copy()

def join_channels(r:Array, g:Array, b:Array) -> Array:
    shape = list(r.shape)
    shape.append(3)
    rgb = np.empty(tuple(shape), dtype=r.dtype)
    rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = r.copy(), g.copy(), b.copy()
    return rgb

program = kernels.load_program(interfaces.hsi)

def rgb2hsi(r:Array, g:Array, b:Array) -> Tuple[Array,Array,Array]:
    h, s, i = program.rgb2hsi(r, g, b) #@UnusedVariable
    return h, s, i

def hsi2rgb(h:Array, s:Array, i:Array) -> Tuple[Array,Array,Array]:
    r,g,b = program.hsi2rgb(h,s,i)
    return r,g,b


