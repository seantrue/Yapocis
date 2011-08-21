import numpy as np
import time
from scipy.ndimage.filters import median_filter
import median
def median3x3(input, iterations=1):
    for i in range(iterations):
        input = median_filter(input,3)
    return input

sizes = []
areas = []
yapocis_deltas = []
numpy_deltas = []
errors = []
for i in range(0,50):
    width = (i+1)*20
    img = np.float32(np.random.random_sample((width,width)))
    
    t = time.time()
    output1 = median3x3(img,10)[1:-1,1:-1]
    numpy_delta = time.time() - t
    
    t = time.time()
    output2 = median.median3x3(img,10)[1:-1,1:-1]
    yapocis_delta = time.time() - t
    
    error = np.abs(output1-output2).sum()
    print width, numpy_delta, yapocis_delta, error
    sizes.append(width)
    areas.append(width*width)
    numpy_deltas.append(numpy_delta)
    yapocis_deltas.append(yapocis_delta)
    errors.append(error)
    
from matplotlib import pyplot as plot

plot.fill_between(sizes, numpy_deltas, yapocis_deltas)
plot.fill_between(areas, numpy_deltas, yapocis_deltas)
plot.show()
