#http://mail.scipy.org/pipermail/scipy-user/2009-February/020125.html
# Anisotropic Diffusion, as per Perona and Malik's paper (see section V).

import numpy
import scipy.ndimage as ndimage
from yapocis.hsi import *

def _exp(image_gradient, scale):
   return numpy.exp(-(numpy.absolute(image_gradient)/scale)**2)

def _inv(image_gradient, scale):
   return 1 / (1 + (numpy.absolute(image_gradient)/scale)**2)

from yapocis.rpc import kernels
from yapocis.rpc import interfaces
program = kernels.load_program(interfaces.peronamalik)
peronamalik = program.filterImage
peronamalik_res = program.filterImage_res

def filterImageN(image, n, scale=3, step_size=0.2):
   if step_size > 0.25:
     raise ValueError('step_size parameter must be <= 0.25 for numerical stability.')
   image = image.copy()
   if image.max() <= 1.0:
      print("Rescaling")
      image *= 256.
      image1 = peronamalik(image, scale, step_size)/256.
      n -= 1
      while 1:
         if n <= 0:
            program.read(image1)
            return image1/256.
         peronamalik_res(image1, scale, step_size, image)
         n -= 1
         if n <= 0:
            program.read(image)
            return image/256.
         peronamalik_res(image, scale, step_size, image1)
         n-= 1

def filterImage(image, scale=3, step_size=0.2):
   if step_size > 0.25:
     raise ValueError('step_size parameter must be <= 0.25 for numerical stability.')
   image = image.copy()
   if image.max() <= 1.0:
      print("Rescaling")
      image *= 256.
   while 1:
      image = peronamalik(image, scale, step_size)
      yield image/256

def filterImageRGB(rgb, scale=3, step_size=.2):
   H,S,I = rgb2hsi(*split_channels(rgb))
   filtered = filterImage(I,scale=scale,step_size=step_size)
   frame = 0
   while 1:
      fI = next(filtered)
      rgb = joinChannels(*hsi2rgb(H, S, fI))
      frame += 1
      yield rgb

def filterImageRGBN(rgb, n, scale=3, step_size=.2):
   H,S,I = rgb2hsi(*split_channels(rgb))
   fI = filterImageN(I,n,scale=scale,step_size=step_size)
   rgb = joinChannels(*hsi2rgb(H, S, fI))
   return rgb
   

def filterImageRGBIS(rgb, scale=3, step_size=.2):
   H,S,I = rgb2hsi(*split_channels(rgb))
   filteredi = filterImage(I,scale=scale,step_size=step_size)
   filtereds = filterImage(S,scale=scale,step_size=step_size)
   frame = 0
   while 1:
      fI = next(filteredi)
      fS = next(filtereds)
      rgb = joinChannels(*hsi2rgb(H, fS, fI))
      frame += 1
      yield rgb
   
if __name__ == "__main__":
   from yapocis.utils import imread, show_array
   from hsi import *
   rgb = imread("test.jpg").astype(np.float32)/256
   show_array("org", rgb)
   from yapocis.utils import stage
   _,_,I = rgb2hsi(*split_channels(rgb))
   stage("generator")
   filtered = filterImage(I)
   for i in range(40):
      fI = next(filtered)
   stage()
   show_array("generator",fI)
   stage("filtern")
   fI = filterImageN(I,40)
   stage()
   show_array("filtern",fI)
