import numpy as np
from rpc import interfaces, kernels

from utils import Shaper

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

def test_gaussian():
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
    

def gaussianKernels(gs, engine=kernels.CPU_ENGINE):
    convs = [("gauss%s" % len(a),a) for a in gs]
    program = kernels.loadProgram(interfaces.convolves,convs=convs)
    krnls = [getattr(program, name) for (name, conv) in convs]
    for i, (name, conv) in enumerate(convs):
        krnls[i].info = dict(left=len(conv)/2, width=len(conv))
    return krnls

def test_kernels():
    gs = gaussians()
    gkernels = gaussianKernels(gs)
    assert len(gs) == len(gkernels)
    for g, gkernel in zip(gs,gkernels):
        a = np.zeros((256,), dtype=np.float32)
        a[127] = 1.0
        a = gkernel(a)
        assert areclose(a.sum(), g.sum())
        # Compare in order of lowest precision
        assert areclose(a.sum(),1.0)

def gaussImage(image, kernel):
    shaper = Shaper(image)
    left = kernel.info["left"]
    flat = shaper.asrows()
    flat = kernel(flat)
    flat[:left] = 0
    flat[-left:] = 0
    shaper.update(flat)
    flat = shaper.ascols()
    flat = kernel(flat)
    flat[:left] = 0
    flat[-left:] = 0
    shaper.update(flat)
    return shaper.asimage()

def test_gaussImage():
    gs = gaussians()
    gkernels = gaussianKernels(gs, engine=kernels.GPU_ENGINE)
    assert len(gs) == len(gkernels)
    import time
    t = time.time()
    for g, gkernel in zip(gs,gkernels):
        a = np.zeros((512,512), dtype=np.float32)
        a[255,255] = 1.0
        a = gaussImage(a, gkernel)
        # Compare in order of lowest precision
    print "Seconds", time.time()-t
    

if __name__ == "__main__":
    test_gaussian()
    test_gaussians()
    test_kernels()
    test_gaussImage()
    print "All is well"
