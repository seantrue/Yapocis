'''
Created on Jul 22, 2011

@author: seant
'''

import numpy as np


from rpc import kernels, interfaces

program = kernels.loadProgram(interfaces.gradient, engine=kernels.GPU_ENGINE)
gradientcl = program.gradient
gradient_res = program.gradient_res
def gradient(image, reach=1):
    grad,theta = gradientcl(image, reach)
    # TODO: gradientCL should not be returning nans, and is
    theta[np.where(np.isnan(theta))] = 0.0
    grad[np.where(np.isnan(grad))] = 0.0
    return grad, theta

def test_gradient():
    import time
    a = np.random.sample((1000,1000)).astype(np.float32)
    t = time.time()
    b = np.gradient(a)
    print "Numpy seconds", time.time()-t
    for engine in (kernels.GPU_ENGINE, kernels.CPU_ENGINE):
        program = kernels.loadProgram(interfaces.gradient, engine=engine)
        t = time.time()
        c = program.gradient(a, 1)
        print "Engine %s seconds %s" % (engine,time.time()-t)
if __name__ == "__main__":
    test_gradient()      
