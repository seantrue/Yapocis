import os

import pyopencl
from pyopencl import mem_flags as mf #@UnusedImport
import numpy as np

#ctx = pyopencl.create_some_context(interactive=False)
ctx = pyopencl.Context([pyopencl.get_platforms()[0].get_devices()[0]])
queue = pyopencl.CommandQueue(ctx)
mf = pyopencl.mem_flags

from mako.template import Template
from mako.lookup import TemplateLookup

#  Basic lookup path is in the directory containing this file, in CWD/rpc
_mydir = os.path.split(__file__)[0]
directories = [".","rpc","interfaces"]
mylookup = TemplateLookup(directories=directories)

def renderProgram(filename, **context):
    filename += ".mako"
    for d in directories:
        tfilename = os.path.join(d,filename)
        if os.path.exists(tfilename):
            break
    t = Template(filename=tfilename,lookup=mylookup)
    return str(t.render(**context))

def renderInterface(source, **context):
    t = Template(source, lookup=mylookup)
    return str(t.render(**context))

class CallableKernel:
    def __init__(self, kernel, params, ctx, queue):   
        self.kernel = kernel
        self.params = params
        self.ctx = ctx
        self.queue = queue
    def prepareArgs(self, args):
        self.returns = []
        pargs = []
        iarg = 0
        paramdict = {}
        for iparam,(direction, dtype, isbuffer, name) in enumerate(self.params):
            if direction != "outlike":
                paramdict[name] = (iparam, direction, dtype, isbuffer)
        assert len(args) == len(paramdict)
        iarg = 0
        for param in self.params:
            (direction, dtype, isbuffer, name) = param
            if dtype:
                dtype = getattr(np, dtype)
            else:
                dtype=None
            # Coerce all args to correct numpy types
            # Todo: Make sure that coercing an array that is already correct does not do too much
            if not isbuffer:
                arg = args[iarg]
                iarg += 1
                assert dtype
                parg = dtype(arg)
                pargs.append(parg)
                continue
            if direction == "in":
                dirflag = mf.READ_ONLY|mf.COPY_HOST_PTR
            elif direction in ("out","outlike"):
                dirflag = mf.WRITE_ONLY|mf.COPY_HOST_PTR
            else:
                dirflag = mf.READ_WRITE|mf.COPY_HOST_PTR
            if direction == "outlike":
                iparam, _, oldtype, _ = paramdict[name]
                if not dtype:
                    dtype = getattr(np, oldtype)
                parg = np.zeros_like(args[iparam], dtype=dtype)
            elif not hasattr(args[iarg], "dtype") or args[iarg].dtype != dtype: 
                arg = args[iarg]    
                parg = np.array(arg, dtype=dtype)
                iarg += 1
            else:
                arg = args[iarg]
                parg = arg
                iarg += 1
            # If we do not have an explicit global size, to to shape of first inbound array
            if self.global_size is None and direction in ("in","inout"):
                self.global_size = parg.shape
            buf = pyopencl.Buffer(self.ctx, dirflag, hostbuf = parg)
            if direction in ("out","inout","outlike"):
                self.returns.append((parg, buf))
            pargs.append(buf)
        return pargs
    
    def __call__(self, *args, **kwargs):
        self.global_size = kwargs.pop("global_size", None)
        self.local_size = kwargs.pop("local_size", None)
        args = self.prepareArgs(args)
        if not isinstance(self.global_size,(tuple,list)):
            self.global_size = (self.global_size,)
        self.evt = self.kernel(self.queue, self.global_size, self.local_size, *args)
        self.evt.wait()
        rval =  self.prepareReturn()
        return rval
            
    def prepareReturn(self):
        rvals = []
        for arg, buf in self.returns:
            pyopencl.enqueue_copy(self.queue, arg, buf).wait()
            rvals.append(arg)
        self.returns = []
        if len(rvals) == 0:
            return None
        elif len(rvals) == 1:
            return rvals[0]
        return tuple(rvals)

class ProgramCL:
    def __init__(self, interface):
        self.callable = {}
        self.interface = interface
        program = interface.program
        for kernel in interface.kernels():
            self.callable[kernel] = CallableKernel(getattr(program,kernel), interface.kernelparams(kernel), ctx, queue)
    def __str__(self):
        return self.interface.interfacename
    def __getattr__(self, attr):
        if not attr in self.callable:
            print "Kernel %s not found, %s available" % (attr, self.callable.keys())
            raise KeyError, attr
        return self.callable[attr]

from interfacecl_parser import getInterfaceCL
def loadProgram(source, debug=False,**context):
    source = renderInterface(source, **context)
    interface = getInterfaceCL(source)
    if debug: print "Interface", interface
    source = renderProgram(interface.interfacename, **context)
    if debug: print "Program", source
    binary = pyopencl.Program(ctx, source)
    program = binary.build()
    interface.program = program
    return ProgramCL(interface)
 
   
from interfaces import *
if __name__ == "__main__":
    print "Interface search path", directories
    interfaces = [(boundedmedian, dict(maxbuf=31, steps=[3,5,7,9,11,15,19,25,31])),
                  (convolve, dict(name="convolve", conv=[1,2,3,4,3,2,1])),
                  (applysegments, dict(name="applysegments", steps=[3,5,8,13,21,31],maxbuf=31)),
                  (hmedian, dict(name="hmedian", width=5, steps=[5])),
                  (median3x3, dict(steps=[9], width=9)),
                  (gradient, {}),
                  (oldhsi, {}),
                  (hsi, {}),
                  (convolves,dict(convs=[("boxone",[1,1,1]),("triangle",[.5,1,.5])])),
                  (mandelbrot,{})
                  ]
    for interface, context in interfaces:
        program =loadProgram(interface, **context)
        print "Interface", program.interface.interfacename
        for kernel in program.interface.kernels():
            print "Kernel", kernel
            print "Params", program.interface.kernelparams(kernel)
            print "OpenCL entry", getattr(program.interface.program, kernel)
            print "Callable", getattr(program, kernel)
        print
