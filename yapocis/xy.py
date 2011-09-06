import numpy as np
from rpc import kernels, interfaces
from utils import showArray
xy = kernels.loadProgram(interfaces.xy)
a = np.zeros((402,798),dtype=np.int32)
addr,x,y,label = xy.addr(a)
showArray("addr",addr)
showArray("x",x)
showArray("y", y)
showArray("label", label)
