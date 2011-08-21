'''
Provides 2d median filter for 2d arrays
Created on Jul 22, 2011
'''

import numpy as np
from rpc import kernels, interfaces

# Provision this program with just one median filter, fixed length of 9
program = kernels.loadProgram(interfaces.median3x3, width=9, steps=[9])
median3x3cl = program.median3x3
def median3x3(image, iterations=1):
    _,height = shape = image.shape #@UnusedVariable
    shape = image.shape
    flat = image.flatten()  
    while iterations > 0:
        flat = median3x3cl(flat.size, height, flat, np.zeros_like(flat))
        iterations -= 1
    return flat.reshape(shape)
