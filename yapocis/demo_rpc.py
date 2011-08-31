'''
Created on Jul 21, 2011
@summary: Test yapocis kernels
@author: seant
'''

import numpy as np
from utils import imread, toimage
from utils import showArray, showArrayGrad


def test():
    from median import median3x3
    from gradient import gradient
    from hsi import rgb2hsi, hsi2rgb, joinChannels, splitChannels

    # Create a noisy image with an embedded white square
    image = np.zeros((201,199),dtype=np.float32)
    width,height = image.shape
    x,y = width/2, height/2
    offset = 10
    image[x-offset:x+offset,y-offset:y+offset] = 2
    image += np.random.random_sample(image.shape)
    
    filtered = median3x3(image, 100)

    showArray("Noisy",image)
    showArray("Filtered",filtered)
    
    image = np.float32(imread("test.jpg"))
    image /= 256.

    showArray("Test HSI",image)
    r,g,b = splitChannels(image)
    h,s,i = rgb2hsi(r,g,b)
    showArray("I",i)
    showArray("S",s)
    showArray("H",h)

    g,a = gradient(i,5)
    showArray("Gradient",g)
    showArray("Angle", a)
    sat = np.ones_like(i)
    gimg = joinChannels(*hsi2rgb(a,sat,g))
    showArray("Color gradient with angle", gimg)
    showArrayGrad("Grad angle", image, a)
    showArrayGrad("Grad vectors", image, a,g)
   
if __name__ == '__main__':
    test()
