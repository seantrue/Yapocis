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

def median3x3fast(image, iterations=1):
    if iterations == 1:
        return median3x3(image)
    _,height = shape = image.shape #@UnusedVariable
    shape = image.shape
    input = image.copy().flatten()
    output = np.zeros_like(input)
    if iterations == 2:
        program.first(input.size, height, input, output)
        input,output = output,input
        program.last(input.size, height, input, output)
        return output.reshape(shape)
    program.first(input.size, height, input, output)
    input,output = output,input
    iterations -= 1
    while iterations > 1:
        program.step(input.size, height, input, output)
        input,output = output,input
        iterations -= 1
    input,output = output,input
    program.last(input.size, height, input, output)
    return output.reshape(shape)

def test_median3():
    a1 = np.random.sample((100,100)).astype(np.float32)
    a2 = a1.copy()
    from utils import stage
    stage("slow")
    b1 = median3x3(a1,5000)
    stage("fast")
    b2 = median3x3fast(a2,5000)
    stage()
    print "Error:", np.sum(np.abs(b1.flatten()-b2.flatten()))
if __name__ == "__main__":
    test_median3()
