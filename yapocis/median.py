'''
Provides 2d median filter for 2d arrays
Created on Jul 22, 2011
'''

import numpy as np
from rpc import kernels, interfaces

# Provision this program with just one median filter, fixed length of 9
program = kernels.loadProgram(interfaces.median3x3, width=9, steps=[9])
median3x3cl = program.median3x3
def median3x3slow(image, iterations=1):
    while iterations > 0:
        image = median3x3cl(image)
        iterations -= 1
    return image.copy()

def median3x3fast(image, iterations=1):
    if iterations == 1:
        # One pass through
        return median3x3cl(image)
    input = image
    output = np.zeros_like(input)
    if iterations == 2:
        # Send in data, don't retrieve
        program.first(input, output)
        input,output = output,input
        # Don't send in, retrieve data
        program.last(input, output)
        return output
    # Send in data, no retrieve
    program.first(input, output)
    input,output = output,input
    # Allow for first and last calls
    iterations -= 2
    while iterations > 1:
        # Iterate with resident data
        program.step(input, output)
        input,output = output,input
        iterations -= 1
    # And retrieve the data
    input,output = output,input
    program.last(input, output)
    return output.copy()

median3x3=median3x3fast

def test_median3():
    a1 = np.random.sample((500,500)).astype(np.float32)
    a2 = a1.copy()
    from utils import stage
    stage("slow")
    b1 = median3x3slow(a1,500)
    stage()
    stage("fast")
    b2 = median3x3fast(a2,500)
    stage()
    error = np.sum(np.abs(b1.flatten()-b2.flatten()))
    assert error == 0.0
    print "All is well"
if __name__ == "__main__":
    test_median3()
