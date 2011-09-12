import numpy as np
from rpc import interfaces, kernels

from utils import showArray, alignImage
from gradient import gradient, gradient_res

def areclose(a,b):
    eps = .00001
    return abs(a-b) <= eps

def gauss_1d(sigma=1., dt=1., limit=.01, normalize=True):
    """
    Get a Gaussian distribution that sums to 1 along 1 dimension, 
    quantized by discrete steps.
    """
    tail=[]
    x=0.
    val=1.
    two_times_sigma_squared=2*sigma*sigma
    scale_factor = 1./np.sqrt(two_times_sigma_squared*np.pi)
    while val > limit:
        val = float(np.exp(-(x*x)/two_times_sigma_squared)*scale_factor)
        tail.append(val)
        x+=dt

    k=tail[::-1]   # Make a copy of the reversed tail
    k.extend(tail[1:]) # Reflect
    
    # Normalize
    if normalize:
        k = np.divide(k,np.sum(k)/dt)
    
    return k

def test_gaussian_1d():
    "Make sure that the kernel sums back to 1"
    g = gauss_1d()
    assert areclose(g.sum(), 1.0)

def gaussians(maxwidth=100, sigma=1.0, scale=1.6, limit=.001, sigmas=[]):
    sigma = 1.0
    gs = []
    if sigmas:
        for sigma in sigmas:
            gs.append(gauss_1d(sigma,limit=limit))
    else:
        while 1:
            g = gauss_1d(sigma,limit=limit)
            if len(g) > maxwidth:
                break
            gs.append(g)
            sigma *= scale
    return gs

def test_gaussians():
    mw = 100
    gs = gaussians(maxwidth=mw)
    for g in gs:
        assert len(g) < mw

program = None
def gaussianKernels(gs):
    global program
    convs = [("gauss%s" % len(a),a) for a in gs]
    convsres = [("gauss%s_res" % len(a),a) for a in gs]
    program = kernels.loadProgram(interfaces.convolvesep,convs=convs)
    krnls = [getattr(program, name) for (name, conv) in convs]
    for i, (name, conv) in enumerate(convs):
        krnls[i].res = getattr(program,convsres[i][0])
        krnls[i].width = len(conv)
    return krnls

_gaussian_basis = gaussians()
def getGaussian(scale):
    assert 0 <= scale and scale < len(_gaussian_basis)
    return _gaussian_basis[scale]
def getScales():
    return(len(_gaussian_basis))
def getGaussianWidth(scale):
    return len(getGaussian(scale))
_gaussian_kernels = gaussianKernels(_gaussian_basis)
def getKernels():
    return _gaussian_kernels
def getKernel(scale):
    assert 0 <= scale and scale < len(_gaussian_kernels)
    return _gaussian_kernels[scale]

def test_kernels():
    for scale in range(getScales()):
        gkernel = getKernel(scale)
        a = np.zeros((256,256), dtype=np.float32)
        a[127,127] = 1.0
        b = gkernel(1,a)
        gsum = getGaussian(scale).sum()
        assert areclose(b[127,:].sum(),gsum)
        b = gkernel(0,a)
        assert areclose(b[:,127].sum(), gsum)
        # Compare in order of lowest precision
        assert areclose(a.sum(),1.0)

def gaussImage(a, scale):
    g = getKernel(scale)
    b = g(0,a)
    c = g(1,b)
    return c.copy()

def test_gaussImage():
    for scale in range(getScales()):
        a = np.zeros((400,700), dtype=np.float32)
        a[200,:] = 1.0
        a[:,200] = 1.0
        for xy in range(400):
            a[xy,xy] = 1.0
        b = gaussImage(a, scale)
        showArray("GI%s" % scale, b)
        
from operators import sub_res, sub
from zcs import zcs,  zcs_res

def setMargin(a, margin, value=0.0):
    a[:,-margin:] = value
    a[:,:margin] = value
    a[-margin:,:] = value
    a[:margin,:] = value
    

def zcsdog(a, scale,clearmargin=True,frame=True, res=True):
    if res:
        g = getKernel(scale)
        g1 = getKernel(scale+1)
        g.write(a)
        smaller = np.empty_like(a)
        larger = np.empty_like(a)
        tmp = np.empty_like(a)
        zca = np.empty_like(a)
        #g.write(smaller)
        #g.write(larger)
        #g.write(tmp)
        #g.write(zca)
        g.res(0,a,tmp)
        g.res(1,tmp,smaller)
        g1.res(0,a,tmp)
        g1.res(1,tmp,larger)
        sub_res(larger,smaller,tmp)
        zcs_res(tmp,zca)
        g.read(zca)
        g.read(smaller)
    else:
        smaller = gaussImage(a,scale)
        larger = gaussImage(a,scale+1)
        dog = smaller - larger
        zca = zcs(dog)
    zca = np.int32(zca)
    # Zero out of bounds and create a frame.
    grad,theta = gradient(smaller)
    margin = getGaussianWidth(scale)+1
    if clearmargin:
        setMargin(zca, margin)
        setMargin(grad,margin)
        setMargin(theta,margin)
    if frame:
        zca[0:,0:] = 1.0
        zca[0:,:-1] = 1.0
        zca[-1:,0:] = 1.0
        zca[-1:,-1:] = 1.0
        zca[margin,:-margin:margin] = 1.0
        zca[-margin:margin,margin] = 1.0
        zca[-margin,-margin:margin] = 1.0
        zca[-margin:margin,-margin] = 1.0
    return zca.copy(),grad.copy(),theta.copy()

def test_zcs():
    import time
    for res in (False, True):
        subtotal = 0.0
        for scale in range(getScales()-1):
            a = np.zeros((1200,1800), dtype=np.float32)
            a[200,:] = 1.0
            a[:,200] = 1.0
            t = time.time()
            zca,grad,theta = zcsdog(a,scale)
            subtotal += (time.time()-t)
            showArray("org %s" % scale, a)
            showArray("zcs %s" % scale, zca)
            showArray("grad %s" % scale, grad)
            showArray("theta %s" % scale, theta)
        print "Res", res, subtotal
        
if __name__ == "__main__":
    #test_gaussian_1d()
    #test_gaussians()
    #test_kernels()
    #test_gaussImage()
    test_zcs()
    print "All is well"
