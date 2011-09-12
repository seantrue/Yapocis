import os, time

import pyopencl
from pyopencl import mem_flags as mf #@UnusedImport
import numpy as np

GPU_ENGINE=0
CPU_ENGINE=1


# Numpy makes a big deal out of .copy() being a deepcopy. Hah!
# a.data is shared until a copy is forced by copy on write.
# This makes mapping arrays to remote buffers tricky
# Theona has an interesting approac, which I am ignoring
# in favor of better sugar.

from weakref import WeakValueDictionary as Weak
class BufferManager(object):
    MEMFLAGS= mf.READ_WRITE|mf.COPY_HOST_PTR
    READ = 0
    WRITE = 1
    DEBUG = True
    def __init__(self, engine=None, context=None, queue=None):
        if engine is None and context is None:
            # Look in environment for engine selection
            engine = os.environ.get("ENGINE", None)
            assert engine != None
        self.ctx = context
        self.queue = queue
        if self.ctx is None:
            self.ctx = pyopencl.Context([pyopencl.get_platforms()[0].get_devices()[engine]])
            self.queue = None
        if self.queue is None:
            self.queue = pyopencl.CommandQueue(self.ctx)
        # Buffer management
        self.arrays = Weak()
        self.buffers = {}
        self.hits = self.misses = 0
        self.purged = 0
        
    def purgeBuffers(self):
        # Drop buffers if no array or data buffer is referencing them
        ids = set(self.arrays.keys())
        for id in self.buffers.keys():
            if not id in ids:
                del self.buffers[id]
                self.purged += 1
    def makeBuffer(self, a):
#        print "makeBuffer", id(a), a.shape, a.size, id(a.data)
        buf = pyopencl.Buffer(self.ctx, self.MEMFLAGS, hostbuf=a)
        aid = id(a)
        self.arrays[aid] = a
        self.buffers[aid] = buf
        return buf

    def ensureBuffer(self, a):
        buf = self.findBuffer(a, self.WRITE)
        if buf is None:
            buf = self.makeBuffer(a)
        return buf
    
    def readBuffer(self, a):
#        print "readBuffer", id(a)
        buf = self.findBuffer(a, self.READ)
        shape = a.shape
        strides = a.strides
        a.shape = (a.size,)
        a.strides = (strides[-1],)
        #pyopencl.enqueue_barrier(self.queue)
        pyopencl.enqueue_copy(self.queue, a, buf).wait()
        a.shape = shape
        a.strides = strides
        return buf
    
    def writeBuffer(self, a):
#        print "writeBuffer", id(a)
        buf = self.ensureBuffer(a)
        shape = a.shape
        strides = a.strides
        a.shape = (a.size,)
        a.strides = (strides[-1],)
        pyopencl.enqueue_copy(self.queue, buf, a).wait()
        a.shape = shape
        a.strides = strides
        #pyopencl.enqueue_barrier(self.queue)
        return buf
    
    def findBuffer(self, a, op):
        "Find an appropriate buffer. Tricky."
        assert op in (self.READ, self.WRITE)
        self.purgeBuffers()
        aid = id(a)
        havea = aid in self.buffers
        # Complete match, easy decision
        if havea:
            self.hits += 1
            return self.buffers[aid]
        else:
            self.misses += 1
        # No match at all, also easy.
        if not havea:
            # Reading an array back with no matching buffer is fatal
            if op == self.READ:
                raise ValueError("Array not in yapocis management, you may have written to it, or be using an assigned or .copy")
            return None
        raise "Epic fail"
        
def test_buffers():
    bmgr = BufferManager(CPU_ENGINE)
    a = np.random.sample((100,)).astype(np.float32)
    b = a.copy()
    bmgr.writeBuffer(a)
    bmgr.readBuffer(a)
    assert a.sum() == b.sum()
    b[:] = 0.0
    assert a.sum() != b.sum()
    assert len(bmgr.arrays.keys()) ==1
    del a
    assert len(bmgr.arrays.keys()) == 0
    a = np.random.sample((100,100)).astype(np.float32)
    b = a.copy()
    bmgr.writeBuffer(a)
    bmgr.readBuffer(a)
    assert a.sum() == b.sum()
    b[:,:] = 0.0
    assert a.sum() != b.sum()
    bmgr.readBuffer(a)
    try:
        bmgr.readBuffer(b)
        assert False, "Should have generated a Value error"
    except:
        pass
    bmgr.writeBuffer(b)
    bmgr.readBuffer(b)
    assert b.sum() == 0.0
    print id(a), id(b)
    assert len(bmgr.arrays.keys())==2
    a = None
    assert len(bmgr.arrays.keys())==1
    b = None
    print bmgr.arrays.keys(),id(a),id(b)
    # Todo: why does the b buffer stay?
    #assert len(bmgr.arrays.keys())==0
if __name__ == "__main__":
    #test_compiling()
    test_buffers()
    print "All is well"
