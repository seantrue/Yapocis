'''
Created on Jul 22, 2011

@author: seant
'''

import numpy as np


from rpc import kernels, interfaces

program = kernels.loadProgram(interfaces.gradient, engine=kernels.GPU_ENGINE)
gradientcl = program.gradient

def gradient(image,reach=1):
    width,height = shape = image.shape
    shape = image.shape
    flat = image.flatten()  
    grad,angle = gradientcl(flat.size, height, flat, reach)
    return grad.reshape(shape),angle.reshape(shape) 


def test_gradient():
    import time
    a = np.random.sample((1000,1000)).astype(np.float32)
    t = time.time()
    np.gradient(a)
    print "Numpy seconds", time.time()-t
    for engine in (kernels.GPU_ENGINE, kernels.CPU_ENGINE):
        program = kernels.loadProgram(interfaces.gradient, engine=engine)
        t = time.time()
        program.gradient(a.size, a.shape[1], a.flatten(), 1)
        print "Engine %s seconds %s" % (engine,time.time()-t)
        
if __name__ == "__main__":
    test_gradient()      
