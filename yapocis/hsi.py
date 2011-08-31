'''
Created on Jul 22, 2011

@author: seant
'''
DEBUG = False
import numpy as np

def histeq(im, nbr_bins=2 ** 16):
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

from rpc import kernels, interfaces

def splitChannels(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return r.copy(), g.copy(), b.copy()

def joinChannels(r, g, b):
    shape = list(r.shape)
    shape.append(3)
    rgb = np.empty(tuple(shape), dtype=r.dtype)
    rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = r.copy(), g.copy(), b.copy()
    return rgb

program = kernels.loadProgram(interfaces.hsi)

def rgb2hsi(r, g, b):
    h, s, i, trace = program.rgb2hsi(r, g, b) #@UnusedVariable
    return h, s, i

def hsi2rgb(h, s, i):
    r,g,b = program.hsi2rgb(h,s,i)
    return r,g,b
if __name__ == "__main__":
    from utils import showArray
    r = np.empty((256, 256), dtype=np.float32)
    g = np.empty_like(r)
    b = np.empty_like(r)
    for i in range(256):
        r[i, :] = i
        g[:, i] = i
        b[i, :] = 255 - i
    rgb = joinChannels(r, g, b)
    rgb /= 255.0
    showArray("Test data", rgb)
    rgb2 = joinChannels(*splitChannels(rgb))
    showArray("Test split and join", rgb2)
    h, s, i = rgb2hsi(*splitChannels(rgb))
    showArray("I", i)
    showArray("H", h)
    showArray("S", s)
    print program

