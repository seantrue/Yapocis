'''
Created on Jul 22, 2011

@author: seant
'''

import numpy as np

from yapocis.rpc import kernels, interfaces
from yapocis.yapocis_types import *

program = kernels.load_program(interfaces.gradient, engine=kernels.GPU_ENGINE)
gradientcl = program.gradient
gradient_res = program.gradient_res


def gradient(image:Array, reach:int=1) -> Tuple[Array,Array]:
    grad, theta = gradientcl(image, reach)
    # TODO: gradientCL should not be returning nans, and is
    theta[np.where(np.isnan(theta))] = 0.0
    grad[np.where(np.isnan(grad))] = 0.0
    return grad, theta
