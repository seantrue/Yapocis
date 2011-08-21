# We don't need to import anything extra.
#from rpc import kernels, interfaces
import numpy
import numpy.linalg as la
import time


# We don't _need_ specify the dtype: that's done in the interface
a = numpy.random.rand(50000)
b = numpy.random.rand(50000)

t = time.time()
# We don't need to manage a compiler and linker at all.
# demo = kernels.loadProgram(interfaces.demo)

# There are no buffers to manage
a_plus_b = a+b
print(la.norm(a_plus_b - (a+b)), la.norm(a_plus_b))
print "Elapsed:", time.time() - t
