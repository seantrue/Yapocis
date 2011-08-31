import os, time

import pyopencl
from pyopencl import mem_flags as mf #@UnusedImport
import numpy as np

GPU_ENGINE=0
CPU_ENGINE=1

MEMFLAGS= mf.READ_WRITE|mf.COPY_HOST_PTR

from weakref import WeakValueDictionary
class Engine(object):
    def __init__(self, engine,autoflatten=True):
        self.ctx = pyopencl.Context([pyopencl.get_platforms()[0].get_devices()[engine]])
        self.queue = pyopencl.CommandQueue(self.ctx)
        # Buffer management
        self.context = WeakValueDictionary()
        self.context_meta = {}
        self.buffers = {}
        self.hits = 0
        self.misses = 0
        self.calls = 0
        self.runtime = 0.0
        self.autoflatten=autoflatten

    def build(self, source):
        binary = pyopencl.Program(self.ctx, source)
        program = binary.build(options=["-w"])
        return program

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
            buf = pyopencl.Buffer(self.ctx, dirflag, hostbuf=self.viewof(arg))
            self.context[argid] = arg
            self.buffers[argid] = buf
            self.context_meta[argid] = metadata
            self.misses += 1
        else:
            buf = self.buffers[argid]
            arg = self.viewof(arg)
            if param[0] != "resident":
                pyopencl.enqueue_copy(self.queue, buf, arg).wait()
            self.hits += 1
        return buf
    def viewof(self, arg):
        assert hasattr(arg,"dtype")
        if self.autoflatten and len(arg.shape) > 1:
            tmparg = arg.view()
            tmparg.shape = (arg.size,)
        else:
            tmparg = arg
        return tmparg
    def write(self, arg):
        # find or construct a buffer that matches the arg. 
        # Param data will be (iparam=0, bufferHint=mf.READ_WRITE|mf.COPY_HOST_PTR, dtype=dtype of arg, isbuffer=True)
        assert hasattr(arg,"dtype")
        param = ("resident",arg.dtype,True)
        buf = self.findBuffer(arg,(MEMFLAGS,param))
        arg = self.viewof(arg)
        pyopencl.enqueue_copy(self.queue, buf, arg).wait()
        return buf
    def read(self, arg):
        assert self.context.has_key(id(arg))
        param = ("resident",arg.dtype,True)
        buf = self.findBuffer(arg, (MEMFLAGS,param))
        tmparg = self.viewof(arg)
        pyopencl.enqueue_copy(self.queue, tmparg, buf).wait()
        del tmparg
        return arg

    
engines = {CPU_ENGINE:Engine(CPU_ENGINE),GPU_ENGINE:Engine(GPU_ENGINE)}
def getEngine(engine):
    """Takes the identifier for an OpenCL engine (CPU_ENGINE, GPU_ENGINE)
    and returns a Engine with buffer management ."""
    return engines[engine]

from mako.template import Template
from mako.lookup import TemplateLookup

def findFile(dirs, subdirs, filename):
    for d in dirs:
        for sd in subdirs:
            pth = os.path.join(d,sd,filename)
            if os.path.exists(pth):
                return pth
    return None
directories = [".","rpc","interfaces"]
def renderProgram(filename, **context):
    """
    Takes the basename of a mako template file, and uses context to generate an OpenCL program
    """
    filename += ".mako"
    tfilename = findFile([os.getcwd(), os.path.abspath(os.path.dirname(__file__))],
                         [".","rpc","interfaces"],
                         filename
                         )
    if not tfilename:
        raise Exception("Can't find template '%s'" % filename)
    mylookup = TemplateLookup(directories=[os.path.dirname(tfilename)])
    t = Template(filename=tfilename,lookup=mylookup)
    return str(t.render(**context))

def renderInterface(source, **context):
    """
    Takes the source for a mako definition of a yapocis.rpc interface and generates
    a parsable definition. Useful when the template can generate multiple parameterized
    routines.
    """
    t = Template(source)
    return str(t.render(**context))

# Todo: Not thread safe: allocate a new Program and Kernels per thread
class Kernel:
    """
    Provide a callable interface to an OpenCL kernel function.
    """
    def __init__(self, program, kernel, params, engine):   
        self.program = program
        self.kernel = kernel
        self.params = params
        self.ctx = engine.ctx
        self.queue = engine.queue
        self.autoflatten = engine.autoflatten
    def prepareArgs(self, args):
        "Takes the actual arguments and maps to the arguments needed by the kernel"
        self.returns = []
        pargs = []
        iarg = 0
        paramdict = {}
        iparam = 0
        for (bufferHint, dtype, isbuffer, name) in self.params:
            if not bufferHint in ("outlike","sizeof","widthof","heightof"):
                paramdict[name] = (iparam, bufferHint, dtype, isbuffer)
                iparam += 1
        assert len(args) == len(paramdict)
        iarg = 0
        for param in self.params:
            (bufferHint, dtype, isbuffer, name) = param
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
            dirflag = mf.READ_WRITE|mf.COPY_HOST_PTR
            buf = None
            if bufferHint in ("sizeof","widthof","heightof"):
                iparam, _, _, _ = paramdict[name]
                arg = args[iparam]
                if not dtype:
                    dtype = np.int32
                size,shape = arg.size, arg.shape
                if len(shape) == 1:
                    assert bufferHint == "sizeof"
                if len(shape) == 2:
                    width,height = shape
                if bufferHint == "sizeof":
                    parg = size
                elif bufferHint == "widthof":
                    parg = width
                elif bufferHint == "heightof":
                    parg = height
                parg = dtype(parg)
                pargs.append(parg)
                continue
            elif bufferHint == "outlike":
                iparam, _, oldtype, _ = paramdict[name]
                if not dtype:
                    dtype = getattr(np, oldtype)
                parg = np.zeros_like(args[iparam], dtype=dtype)
            elif bufferHint == "resident":
                arg = args[iarg]
                parg = arg
                iarg += 1
                buf = self.program.findBuffer(parg, (dirflag, param))
                assert buf, "Resident buffer is not present, array have gone out of scope"
            elif not hasattr(args[iarg], "dtype") or args[iarg].dtype != dtype: 
                arg = args[iarg]    
                parg = np.array(arg, dtype=dtype)
                iarg += 1
            else:
                arg = args[iarg]
                parg = arg
                iarg += 1
            # If we do not have an explicit global size, set to shape of first inbound array
            if self.global_size is None and bufferHint in ("in","inout","resident"):
                if self.autoflatten:
                    self.global_size = parg.size
                else:
                    self.global_size = parg.shape
            if buf is None:
                buf = self.program.findBuffer(parg, (dirflag, param))
            if bufferHint in ("out","inout","outlike"):
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
            pyopencl.enqueue_copy(self.queue, self.program.viewof(arg), buf).wait()
            rvals.append(arg)
        self.returns = []
        if len(rvals) == 0:
            return None
        elif len(rvals) == 1:
            return rvals[0]
        return tuple(rvals)

class Program:
    """
    Takes a parsed yapocis interface and returns a class/module like structure
    which supports calling into kernel methods of the interface.
    Manages buffers on the OpenCL device to prevent over-allocation.
    """
    def __init__(self, interface, engine, debug=False, **context):
        source = renderProgram(interface.interfacename, **context)
        self.source = source
        if debug: 
            print "Source"
            print source
        mf = pyopencl.mem_flags
        self.engine = getEngine(engine)
        interface.program = self.engine.build(source)
        self.callable = {}
        self.context = self.engine.context
        self.context_meta = self.engine.context_meta
        self.ctx = self.engine.ctx
        self.queue = self.engine.queue
        self.buffers = self.engine.buffers
        self.interface = interface
        program = interface.program
        self.hits = 0
        self.misses = 0
        self.calls = 0
        self.runtime = 0.0
        for kernel in interface.kernels():
            alias = interface.kernelalias(kernel)
            self.callable[kernel] = Kernel(self, getattr(program,alias), interface.kernelparams(kernel), self.engine)
    def __getattr__(self, attr):
        "Any unbound attribute is checked to see if it is a callable"
        if not attr in self.callable:
            print "Kernel %s not found, %s available" % (attr, self.callable.keys())
            raise KeyError, attr
        return self.callable[attr]
    def purgeBuffers(self):
        self.engine.purgeBuffers()
    def findBuffer(self,*args):
        return self.engine.findBuffer(*args)
    def read(self, *args):
        return self.engine.read(*args)
    def write(self, *args):
        return self.engine.write(*args)
    def viewof(self, *args):
        return self.engine.viewof(*args)
    def stats(self):
        return dict(calls=self.calls, hits=self.hits, misses=self.misses, buffers=len(self.buffers), runtime=self.runtime)
    def __str__(self):
        self.purgeBuffers()
        return "%s calls:%s hits:%s misses:%s cached:%s time:%s" % (self.interface.interfacename, self.calls, self.hits, self.misses, len(self.buffers), self.runtime)
    
from interfacecl_parser import getInterfaceCL
def loadProgram(source, engine=CPU_ENGINE, debug=False, **context):
    """
    Primary interface for yapocis.rpc
    Factory returning runnable interface based on a interface specification.
    """
    source = renderInterface(source, **context)
    interface = getInterfaceCL(source)
    interface.source = source
    if debug: print "Interface", interface
    return Program(interface, engine, debug=debug, **context)

# Make the standard yapocis interfaces visible immediately.
from interfaces import * #@UnusedWildImport

def test_compiling():
    # TODO: as tests, this may fail on other systems. Replace with known hash
    
    sourcehashes = [6202013264865897877, -8119117248955560468, 7139354972159999745, -682776529868236564, 1403331523463003145, -7762762614049038277, 8451406568839298774]   
    interfacehashes =[6402197007761216735, -110486948446408996, 5834752055632543161, -9192547793151607825, 6003157927413075666, 1424139564779745970, 4108900838076230702]
    print "Interface search path", directories
    interfaces = [(convolve, dict(name="convolve", conv=[1,2,3,4,3,2,1])),
                  (median3x3, dict(steps=[9], width=9)),
                  (gradient, {}),
                  (hsi, {}),
                  (convolves,dict(convs=[("boxone",[1,1,1]),("triangle",[.5,1,.5])])),
                  (mandelbrot,{}),
                  (demo,{}),
                  ]
    for itest, (interface, context) in enumerate(interfaces):
        program =loadProgram(interface, engine=GPU_ENGINE, debug=True,**context)
        assert interfacehashes[itest] == hash(program.interface.source), (interface, hash(program.interface.source))
        assert sourcehashes[itest] == hash(program.source), hash(program.source)
        print "Interface", program.interface.interfacename
        for kernel in program.interface.kernels():
            print "Kernel", kernel
            print "Params", program.interface.kernelparams(kernel)
            print "OpenCL entry", getattr(program.interface.program, program.interface.kernelalias(kernel))
            print "Callable", getattr(program, kernel)
        print

def test_context():
    engine = getEngine(GPU_ENGINE)
    a = np.random.sample((100,)).astype(np.float32)
    b = a.copy()
    engine.write(a)
    a[:] = 0
    engine.read(a)
    diff = np.abs(a-b).sum()
    assert diff == 0.0
    del a
    try:
        engine.read(b)
        print "Did not fail on unmapped array"
    except AssertionError:
        pass
    assert (engine.hits,engine.misses) == (1,1)
   
if __name__ == "__main__":
    test_compiling()
    test_context()
    print "All is well"
