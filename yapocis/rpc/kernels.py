import os, time

import pyopencl
from pyopencl import mem_flags as mf #@UnusedImport
import numpy as np

CPU_ENGINE=0
GPU_ENGINE=1
engine_ids = {"gpu":GPU_ENGINE,"cpu":CPU_ENGINE}
engine_names = {GPU_ENGINE:"gpu",CPU_ENGINE:"cpu"}

MEMFLAGS= mf.READ_WRITE|mf.COPY_HOST_PTR

from buffers import BufferManager

class Engine(object):
    def __init__(self, engine=None,autoflatten=True):
        engine = int(os.environ.get("ENGINE",engine))
        self.engine_name = engine_names[engine]
        self.ctx = pyopencl.Context([pyopencl.get_platforms()[0].get_devices()[engine]])
        self.queue = pyopencl.CommandQueue(self.ctx)
        self.bmgr = BufferManager(context=self.ctx, queue=self.queue)
        self.context_meta = {}
        self.calls = 0
        self.runtime = 0.0
        self.autoflatten=autoflatten

    def build(self, source):
        binary = pyopencl.Program(self.ctx, source)
        program = binary.build(options=["-w"])
        return program
    def purge(self):
        self.bmgr.purgeBuffers()
    def write(self, a):
        assert hasattr(a,"dtype")
        return self.bmgr.writeBuffer(a)
    def read(self, a):
        return self.bmgr.readBuffer(a)
    def ensure(self, a):
        return self.bmgr.ensureBuffer(a)
engines = {CPU_ENGINE:Engine(CPU_ENGINE),GPU_ENGINE:Engine(GPU_ENGINE)}

def getEngine(engine):
    """Takes the identifier for an OpenCL engine (CPU_ENGINE, GPU_ENGINE)
    and returns a Engine with buffer management ."""
    if isinstance(engine, basestring):
        assert engine.lower() in engine_ids.keys(), "Unknown engine name %s" % engine
        engine = engine_ids[engine.lower()]
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
        self.context_meta = self.engine.context_meta
        self.ctx = self.engine.ctx
        self.queue = self.engine.queue
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
    def purge(self):
        self.engine.purge()
    def findBuffer(self,*args):
        return self.engine.findBuffer(*args)
    def read(self, *args):
        return self.engine.read(*args)
    def write(self, *args):
        return self.engine.write(*args)
    def ensure(self, *args):
        return self.engine.ensure(*args)
    def stats(self):
        return dict(interfacename=self.interface.interfacename,
                    calls=self.calls,
                    hits=self.engine.bmgr.hits,
                    misses=self.engine.bmgr.misses,
                    buffers=len(self.engine.bmgr.buffers),
                    runtime=self.runtime,
                    engine = self.engine.engine_name)
    def __str__(self):
        self.purge()
        return "%(interfacename)s engine:%(engine)s calls:%(calls)s hits:%(hits)s misses:%(misses)s buffers:%(buffers)s runtime:%(runtime)s" % self.stats()

    
from interfacecl_parser import getInterfaceCL
def loadProgram(source, engine=GPU_ENGINE, debug=False, **context):
    """
    Primary interface for yapocis.rpc
    Factory returning runnable interface based on a interface specification.
    """
    source = renderInterface(source, **context)
    interface = getInterfaceCL(source)
    interface.source = source
    if debug: print "Interface", interface
    return Program(interface, engine, debug=debug, **context)

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
        self.mapArguments()
#        print self.argspecs
#        print self.posbyname
    def read(self, *args):
        return self.program.read(*args)
    def write(self, *args):
        return self.program.write(*args)
    def ensure(self, *args):
        return self.program.ensure(*args)
    def mapArguments(self):
        argspecs = [None]*len(self.params)
        posbyname = {}
        iarg = 0
        for iparam, (bufferHint, dtype, isbuffer, name) in enumerate(self.params):
            if bufferHint in ("outlike","sizeof","widthof","heightof","depthof"):
                argspecs[iparam] = (bufferHint, name)
            else:
                posbyname[name] = iarg
                argspecs[iparam] = (bufferHint, iarg)
                iarg += 1
        self.argspecs = argspecs
        self.posbyname = posbyname
    def prepareArgs(self, args):
        "Takes the actual arguments and maps to the arguments needed by the kernel"
        assert len(self.posbyname) == len(args), "Expected %s arguments, got %s" % (len(self.posbyname),len(args))
        self.returns = []
        realargs = [None] * len(self.params)
        for iparam, (bufferHint, value) in enumerate(self.argspecs):
            (_, dtype, isbuffer, name) = self.params[iparam]
            if dtype in ("char","uchar") :
                dtype = "uint8"
            if dtype:
                dtype = getattr(np, dtype)
            else:
                dtype=None
            if not isbuffer:
                assert dtype, "Must have type for scalar argument %s" % iparam
                arg = args[value]
                realargs[iparam] = dtype(arg)
                continue
            methodname = "handle_%s" % bufferHint
            realargs[iparam] = getattr(self,methodname)(args, value, dtype)
        if self.global_size is None:
            for arg in args:
                if hasattr(arg,"size"):
                    if self.autoflatten:
                        self.global_size = arg.size
                    else:
                        self.global_size = arg.shape
                    break
        return realargs
    def handle_sizeof(self, args, name, dtype):
        arg = args[self.posbyname[name]]
        if not dtype:
            dtype = np.int32
        return dtype(arg.size)
    # Assume row major storage.
    # addr(x,y) = (x*height)+y
    #  x = addr/height
    #  y = (addr - (addr/height)*height)
    def handle_widthof(self, args, name, dtype):
        arg = args[self.posbyname[name]]
        if not dtype:
            dtype = np.int32
        return dtype(arg.shape[0])
    def handle_heightof(self, args, name, dtype):
        arg = args[self.posbyname[name]]
        if not dtype:
            dtype = np.int32
        return dtype(arg.shape[1])
    def handle_depthof(self, args, name, dtype):
        arg = args[self.posbyname[name]]
        if not dtype:
            dtype = np.int32
        return dtype(arg.shape[2])
    def handle_outlike(self,args, name, dtype):
        arg = args[self.posbyname[name]]
        if not dtype:
            dtype = arg.dtype
        a = np.zeros_like(arg, dtype=dtype)
        buf = self.ensure(a)
        self.returns.append((a,buf))
        return buf
    def handle_in(self, args, pos, dtype):
        arg = args[pos]
        assert dtype
        if arg.dtype != dtype:
            arg = arg.astype(dtype)
        buf = self.write(arg)
        return buf
    def handle_inout(self, args, pos, dtype):
        arg = args[pos]
        assert dtype
        if arg.dtype != dtype:
            arg = arg.astype(dtype)
        buf = self.write(arg)
        self.returns.append((arg,buf))
        return buf
    def handle_out(self, args, pos, dtype):
        arg = args[pos]
        assert dtype
        if arg.dtype != dtype:
            arg = arg.astype(dtype)
        buf = self.ensure(arg)
        self.returns.append((arg,buf))
        return buf
    handle_resident = handle_out
    
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
        poll = kwargs.pop("poll", None)
        if poll:
            while(True):
                status = self.evt.get_info(pyopencl.event_info.COMMAND_EXECUTION_STATUS)
                if status == pyopencl.command_execution_status.COMPLETE:
                    break
                time.sleep(poll)
        else:
            self.evt.wait()
        #pyopencl.enqueue_barrier(self.queue)
        rval =  self.prepareReturn()
        self.program.runtime += (time.time()-t)
        return rval
            
    def prepareReturn(self):
        "Takes the return value buffers (out and outlike) and prepares proper Numpy arrays"
        rvals = []
        for arg, buf in self.returns:
            self.read(arg)
            rvals.append(arg)
        self.returns = []
        if len(rvals) == 0:
            return None
        elif len(rvals) == 1:
            return rvals[0]
        return tuple(rvals)



# Make the standard yapocis interfaces visible immediately.
from interfaces import * #@UnusedWildImport

def test_compiling():
    print "Interface search path", directories
    interfaces = [(convolve, dict(name="convolve", conv=[1,2,3,4,3,2,1])),
                  (median3x3, dict(steps=[9], width=9)),
                  (gradient, {}),
                  (convolves,dict(convs=[("boxone",[1,1,1]),("triangle",[.5,1,.5])])),
                  (mandelbrot,{}),
                  (demo,{}),
                  (xy,{}),
                  (hsi, {}),
                  ]
    for itest, (interface, context) in enumerate(interfaces):
        program =loadProgram(interface, engine=CPU_ENGINE, debug=True,**context)
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
    #b[0]=b[1]
    engine.write(a)
    engine.read(a)
    diff = np.abs(a-b).sum()
    assert diff == 0.0
    del a
    try:
        engine.read(b)
        print "Did not fail on unmapped array"
    except ValueError:
        pass

def test_demo():
    from interfaces import demo
    program = loadProgram(demo)
    print program.sum
    a = np.array([0,1,2,3]).astype(np.float32)
    b = np.array([1,1,1,1]).astype(np.float32)
    print program.sum(a,b)
    print program
    
if __name__ == "__main__":
    test_compiling()
    #test_context()
    #test_demo()
    print "All is well"
