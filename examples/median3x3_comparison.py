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
    width = (i+1)*40
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
areas = np.array(areas)
numpy_deltas = np.array(numpy_deltas)
yapocis_deltas = np.array(yapocis_deltas)
ratio = yapocis_deltas/numpy_deltas
p = plot.subplot(111)
plot.title("Median 3x3 filter performance",fontsize="large")
l1 = plot.plot(areas, numpy_deltas)
l2 = plot.plot(areas, yapocis_deltas)
l3 = p.plot(areas,ratio)
plot.fill_between(areas, numpy_deltas, yapocis_deltas)
plot.figlegend( (l1, l2, l3), ('numpy', 'yapocis','ratio'), 'upper left', shadow=True)
plot.xlabel("Pixels",fontsize="medium")
plot.ylabel("Seconds",fontsize="medium")
plot.show()
