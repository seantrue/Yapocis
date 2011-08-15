'''
Created on Jul 22, 2011

@author: seant
'''

import numpy as np


from rpc import kernels, interfaces


gradientcl = kernels.loadProgram(interfaces.gradient).gradient

def gradient(image,reach=1):
    width,height = shape = image.shape
    shape = image.shape
    flat = image.flatten()  
    grad,angle = gradientcl(flat.size, height, flat, reach)
    return grad.reshape(shape),angle.reshape(shape) 
