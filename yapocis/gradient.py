'''
Created on Jul 22, 2011

@author: seant
'''

import numpy as np

from yapocis.rpc import kernels, interfaces

program = kernels.load_program(interfaces.gradient, engine=kernels.GPU_ENGINE)
gradientcl = program.gradient
gradient_res = program.gradient_res


def gradient(image, reach=1):
    grad, theta = gradientcl(image, reach)
    # TODO: gradientCL should not be returning nans, and is
    theta[np.where(np.isnan(theta))] = 0.0
    grad[np.where(np.isnan(grad))] = 0.0
    return grad, theta
