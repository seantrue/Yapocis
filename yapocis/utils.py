from PIL import Image #@UnresolvedImport
import numpy as np
DEBUG=False


# bytescale, fromimage, toimage borrowed from scipy.misc
# Used under the liberal BSD license
# Returns a byte-scaled image
def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).

    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin :  Scalar
        Bias scaling of small values, Default is data.min().
    cmax : scalar
        Bias scaling of large values, Default is data.max().
    high : scalar
        Scale max value to `high`.
    low : scalar
        Scale min value to `low`.

    Returns
    -------
    img_array : ndarray
        Bytescaled array.

    Examples
    --------
    >>> img = array([[ 91.06794177,   3.39058326,  84.4221549 ],
                     [ 73.88003259,  80.91433048,   4.88878881],
                     [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)

    """
    if data.dtype == np.uint8:
        return data
    high = high - low
    if cmin is None: cmin = data.min()
    if cmax is None: cmax = data.max()
    scale = high *1.0 / (cmax-cmin or 1)
    bytedata = ((data*1.0-cmin)*scale + 0.4999).astype(np.uint8)
    return bytedata + np.cast[np.uint8](low)

def fromimage(im, flatten=0):
    """
    Return a copy of a PIL image as a numpy array.

    Parameters
    ----------
    im : PIL image
        Input image.
    flatten : bool
        If true, convert the output to grey-scale.

    Returns
    -------
    fromimage : ndarray
        The different colour bands/channels are stored in the
        third dimension, such that a grey-image is MxN, an
        RGB-image MxNx3 and an RGBA-image MxNx4.

    """
    if not Image.isImageType(im):
        raise TypeError("Input is not a PIL image.")
    if flatten:
        im = im.convert('F')
    return np.array(im)

_errstr = "Mode is unknown or incompatible with input array shape."

def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.  The mode of the
    PIL image depends on the array shape, the pal keyword, and the mode
    keyword.

    For 2-D arrays, if pal is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then mode='P', otherwise mode='L', unless mode is given
    as 'F' or 'I' in which case a float and/or integer array is made

    For 3-D arrays, the channel_axis argument tells which dimension of the
      array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
      by default or 'YCbCr' if selected.
    if the

    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape)==2 or ((len(shape)==3) and \
                              ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for any mode.")
    if len(shape) == 2:
        shape = (shape[1],shape[0]) # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.fromstring(mode,shape,data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data,high=high,low=low,cmin=cmin,cmax=cmax)
            image = Image.fromstring('L',shape,bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal,dtype=uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = arange(0,256,1,dtype=uint8)[:,newaxis] * \
                      ones((3,),dtype=uint8)[newaxis,:]
                image.putpalette(np.asarray(pal,dtype=uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.fromstring('1',shape,bytedata.tostring())
            return image
        if cmin is None:
            cmin = amin(ravel(data))
        if cmax is None:
            cmax = amax(ravel(data))
        data = (data*1.0 - cmin)*(high-low)/(cmax-cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.fromstring(mode,shape,data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3,4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data,high=high,low=low,cmin=cmin,cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1],shape[0])
    elif ca == 1:
        strdata = transpose(bytedata,(0,2,1)).tostring()
        shape = (shape[2],shape[0])
    elif ca == 0:
        strdata = transpose(bytedata,(1,2,0)).tostring()
        shape = (shape[2],shape[1])
    if mode is None:
        if numch == 3: mode = 'RGB'
        else: mode = 'RGBA'


    if mode not in ['RGB','RGBA','YCbCr','CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.fromstring(mode, shape, strdata)
    return image

def imread(filename):
    img = Image.open(filename)
    return fromimage(img)

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
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    imhist[0] = 0
   
    cdf = imhist.cumsum() #cumulative distribution function
    # TODO: And here we have a non-linear something or other
    cdf ** .5
    # TODO: This normalization looks wrong for a different number of bins.
    cdf = (2**16-1) * cdf / cdf[-1] #normalize

    #cdf = cdf / (2**16.)  #normalize
    #use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    return np.array(im2,  np.float32).reshape(im.shape)

def histeq(im,nbr_bins=256):
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape), cdf

if __name__ == "__main__":
    # TODO: Add tests here
    pass

    
