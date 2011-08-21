'''
Created on Jul 22, 2011

@author: seant
'''

import numpy as np


from scikits.image.color import rgb2hsv, hsv2rgb #@UnusedImport
from rpc import kernels, interfaces


program = kernels.loadProgram(interfaces.median3x3, width=9, steps=[9])
median3x3cl = program.median3x3
def median3x3(image, iterations=1):
    width,height = shape = image.shape
    shape = image.shape
    flat = image.flatten()  
    for i in range(iterations):      
        flat = median3x3cl(flat.size, height, flat, np.zeros_like(flat))
    return flat.reshape(shape)
