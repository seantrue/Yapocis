import numpy
import numpy.linalg as la
import time

a = numpy.random.rand(50000)
b = numpy.random.rand(50000)

t = time.time()
a_plus_b = a+b
print "Elapsed:", time.time() - t

print(la.norm(a_plus_b - (a+b)), la.norm(a_plus_b))
