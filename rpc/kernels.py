import os, time

import pyopencl
from pyopencl import mem_flags as mf #@UnusedImport
import numpy as np

#ctx = pyopencl.create_some_context(interactive=False)
ctx = pyopencl.Context([pyopencl.get_platforms()[0].get_devices()[0]])
queue = pyopencl.CommandQueue(ctx)
mf = pyopencl.mem_flags

from mako.template import Template
from mako.lookup import TemplateLookup

directories = [".","rpc","interfaces"]
mylookup = TemplateLookup(directories=directories)

def renderProgram(filename, **context):
    """
    Takes the basename of a mako template file, and uses context to generate an OpenCL program
    """
    filename += ".mako"
    for d in directories:
        tfilename = os.path.join(d,filename)
        if os.path.exists(tfilename):
            break
    t = Template(filename=tfilename,lookup=mylookup)
    return str(t.render(**context))

def renderInterface(source, **context):
    """
    Takes the source for a mako definition of a yapocis.rpc interface and generates
    a parsable definition. Useful when the template can generate multiple parameterized
    routines.
    """
    t = Template(source, lookup=mylookup)
    return str(t.render(**context))

# Todo: Not thread safe: allocate a new Program and Kernels per thread
class Kernel:
    """
    Provide a callable interface to an OpenCL kernel function.
    """
    def __init__(self, program, kernel, params, ctx, queue):   
        self.program = program
        self.kernel = kernel
        self.params = params
        self.ctx = ctx
        self.queue = queue
    def prepareArgs(self, args):
        "Takes the actual arguments and maps to the arguments needed by the kernel"
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
#            if direction == "in":
#                dirflag = mf.READ_ONLY|mf.COPY_HOST_PTR
#            elif direction in ("out"):
#                dirflag = mf.WRITE_ONLY|mf.COPY_HOST_PTR
#            else:
#                dirflag = mf.READ_WRITE|mf.COPY_HOST_PTR
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
            buf = self.program.findBuffer(parg, (dirflag, param))
            if direction in ("out","inout","outlike"):
                self.returns.append((parg, buf))
            pargs.append(buf)
        return pargs
    
    def __call__(self, *args, **kwargs):
        "Call a kernel, prepare arguments, and track performance"
        self.program.calls += 1
        t = time.time()
        self.global_size = kwargs.pop("global_size", None)
        self.local_size = kwargs.pop("local_size", None)
        args = self.prepareArgs(args)
        if not isinstance(self.global_size,(tuple,list)):
            self.global_size = (self.global_size,)
        self.evt = self.kernel(self.queue, self.global_size, self.local_size, *args)
        self.evt.wait()
        rval =  self.prepareReturn()
        self.program.runtime += (time.time()-t)
        return rval
            
    def prepareReturn(self):
        "Takes the return value buffers (out and outlike) and prepares proper Numpy arrays"
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

from weakref import WeakValueDictionary
class Program:
    """
    Takes a parsed yapocis interface and returns a class/module like structure
    which supports calling into kernel methods of the interface.
    Manages buffers on the OpenCL device to prevent over-allocation.
    """
    def __init__(self, interface):
        self.ctx = ctx
        self.callable = {}
        self.context = WeakValueDictionary()
        self.context_meta = {}
        self.buffers = {}
        self.interface = interface
        program = interface.program
        self.hits = 0
        self.misses = 0
        self.calls = 0
        self.runtime = 0.0
        self.queue = queue
        for kernel in interface.kernels():
            self.callable[kernel] = Kernel(self, getattr(program,kernel), interface.kernelparams(kernel), ctx, queue)
    def __getattr__(self, attr):
        "Any unbound attribute is checked to see if it is a callable"
        if not attr in self.callable:
            print "Kernel %s not found, %s available" % (attr, self.callable.keys())
            raise KeyError, attr
        return self.callable[attr]
    def purgeBuffers(self):
        "Purges mapped buffers if the Numpy array has been garbage collected"
        for key in self.buffers.keys():
            if not key in self.context:
                del self.buffers[key]
        for key in self.context_meta.keys():
            if not key in self.context:
                del self.context_meta[key]
    def findBuffer(self, arg, metadata):
        "Finds or allocates an appropriately mapped buffer"
        self.purgeBuffers()
        argid = id(arg)
        dirflag, param = metadata
        living = self.context.get(argid, None)
        # TODO: Make a better force miss
        #living is None
        if living is None:
            buf = pyopencl.Buffer(self.ctx, dirflag, hostbuf = arg)
            self.context[argid] = arg
            self.buffers[argid] = buf
            self.context_meta[argid] = metadata
            self.misses += 1
        else:
            buf = self.buffers[argid]
            pyopencl.enqueue_copy(self.queue, buf, arg).wait()
            self.hits += 1
        return buf
    def __str__(self):
        self.purgeBuffers()
        return "%s calls:%s hits:%s misses:%s cached:%s time:%s" % (self.interface.interfacename, self.calls, self.hits, self.misses, len(self.buffers), self.runtime)
    
from interfacecl_parser import getInterfaceCL
def loadProgram(source, debug=False,**context):
    """
    Primary interface for yapocis.rpc
    Factory returning runnable interface based on a interface specification.
    """
    source = renderInterface(source, **context)
    interface = getInterfaceCL(source)
    if debug: print "Interface", interface
    source = renderProgram(interface.interfacename, **context)
    if debug: print "Program", source
    binary = pyopencl.Program(ctx, source)
    program = binary.build(options=["-w"])
    interface.program = program
    return Program(interface)

# Make the standard yapocis interfaces visible immediately.
from interfaces import * #@UnusedWildImport

def test_compiling():
    print "Interface search path", directories
    interfaces = [(convolve, dict(name="convolve", conv=[1,2,3,4,3,2,1])),
                  (hmedian, dict(name="hmedian", width=5, steps=[5])),
                  (median3x3, dict(steps=[9], width=9)),
                  (gradient, {}),
                  (oldhsi, {}),
                  (hsi, {}),
                  (convolves,dict(convs=[("boxone",[1,1,1]),("triangle",[.5,1,.5])])),
                  (mandelbrot,{}),
                  (averagesegments,{}),
                  (boundedaverage,{}),
                  (label, {}),
                  (demo,{}),
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
   
if __name__ == "__main__":
    test_compiling()
