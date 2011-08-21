from scipy.misc import toimage
from PIL import Image #@UnresolvedImport
import numpy as np
DEBUG=False


import time
class Stage:
    def __init__(self):
        self.t = None
        self.stage = None
    def __call__(self, *args):
        stage = " ".join([str(arg) for arg in args])
        if self.stage:
            t = time.time()
            print self.stage, "done in", t-self.t
        if args:
            print "Start", stage 
            self.t = time.time()
            self.stage = stage
        else:
            self.t = self.stage = None

stage = Stage()

def sign(title, image):
    from PIL import ImageFont #@UnresolvedImport
    from PIL import ImageDraw #@UnresolvedImport
    if hasattr(image, "dtype"):
        shape = image.shape
        img = toimage(image)
    else:
        shape = image.size
        img = image
    width, height = shape
    #font = ImageFont.truetype("/System/Library/Fonts/AppleGothic.ttf",25)
    font = ImageFont.truetype("zapfino.ttf",15)
    draw = ImageDraw.Draw(img)
    try:
        pixels = [img.getpixel((x,height-15)) for x in range(10,30)]
        pixels = np.array(pixels)
        darkness = pixels.sum()/pixels.size
        if darkness < 128:
            ink = 255
        else:
            ink = 0
    except:
        ink = 0
    if len(shape) > 2:
        ink = [ink] * shape[-1]
        ink = tuple(ink)
    draw.text((10, height-75), title, ink,font=font)
    return img


def _showArray(title, image):
    from PIL import ImageFont #@UnresolvedImport
    from PIL import ImageDraw #@UnresolvedImport
    shape = image.shape
    if DEBUG and len(shape) == 2:
        print title, shape
        for y in range(shape[0]):
            print np.int32(image[y,:]*1000)/1000.
        print "="*32
        print
    img = toimage(image)
    font = ImageFont.truetype("/System/Library/Fonts/AppleGothic.ttf",25)
    #font = ImageFont.truetype("zapfino.ttf",25)
    draw = ImageDraw.Draw(img)
    try:
        pixels = [img.getpixel((x,15)) for x in range(10,30)]
        pixels = np.array(pixels)
        darkness = pixels.sum()/pixels.size
        if darkness < 128:
            ink = 255
        else:
            ink = 0
    except IndexError:
        ink = 0
    if len(shape) > 2:
        ink = [ink] * shape[-1]
        ink = tuple(ink)
    draw.text((10, 10), title, ink,font=font)
    return img

def showArray(title,image):
    img = _showArray(title, image)
    img.show()

def showArrayGrad(title, image, theta, grad=None):
    from PIL import ImageDraw #@UnresolvedImport
    img = _showArray(title, image).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size
    if grad is None:
        grad = np.zeros_like(theta)
        grad[:,:] = 10
    else:
        grad = grad.copy()
        grad -= grad.min()
        grad /= grad.max()
        grad *= 10
    cos = np.cos(theta*2*3.14159)
    sin = np.sin(theta*2*3.14159)
    for x in range(10,width-10,5):
        for y in range(10,height-10,5):
            i,j  =y, x
            dx = grad[i,j] * cos[i,j]
            dy = grad[i,j] * sin[i,j]
            try:
                x1,y1 = int(x-dx), int(y-dy)
                x2,y2 = int(x+dx), int(y+dy)
            except ValueError:
                # Nans happen
                continue
            a,b,c = img.getpixel((x,y))
            grey = (a+b+c)/3
            if grey > 128:
                color="black"
            else:
                color = "white"
            draw.line([(x1,y1),(x2,y2)],fill=color)
    img.show()

class Shaper:
    def __init__(self, data):
        assert len(data.shape) == 2, "Shaper requires 2-d data"
        self.shape = data.shape
        self.order = None
        self.data = data[:,:]
    def update(self, data):
        assert self.data.shape == data.shape, "Must conform"
        self.data = data
    def asimage(self):
        if self.order:
            self.data = self.data.reshape(self.shape,order=self.order)
            self.order = None
        return self.data
    def asrows(self):
        if self.order == 'C':
            return self.data
        if self.order:
            self.data = self.asimage()
        if self.order == None:
            self.order = 'C'
            self.data = self.data.reshape(-1, order=self.order)
            return self.data
    def ascols(self):
        if self.order == 'F':
            return self.data
        if self.order:
            self.data = self.asimage()
        if self.order == None:
            self.order = 'F'
            self.data = self.data.reshape(-1, order=self.order)
        return self.data

def histeq_large(im,nbr_bins=2**16):
    #get image histogram
    imhist,bins = numpy.histogram(im.flatten(),nbr_bins,normed=True)
    imhist[0] = 0
   
    cdf = imhist.cumsum() #cumulative distribution function
    # TODO: And here we have a non-linear something or other
    cdf ** .5
    # TODO: This normalization looks wrong for a different number of bins.
    cdf = (2**16-1) * cdf / cdf[-1] #normalize

    #cdf = cdf / (2**16.)  #normalize
    #use linear interpolation of cdf to find new pixel values
    im2 = numpy.interp(im.flatten(),bins[:-1],cdf)
    return numpy.array(im2,  numpy.float32).reshape(im.shape)

def histeq(im,nbr_bins=256):
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape), cdf

if __name__ == "__main__":
    # TODO: Add tests here
    pass

    
