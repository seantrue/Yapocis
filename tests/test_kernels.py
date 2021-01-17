import numpy as np
import time

from yapocis.xy import xy
from yapocis.zcs import zcs
from yapocis.gaussian import gauss_1d, are_close, gaussians, get_scales, get_gaussian_kernel, get_gaussian, gauss_image, \
    zcsdog
from yapocis.operators import add, add_res, program, sub, sub_res, mul, mul_res
from yapocis.rpc import kernels, interfaces
from yapocis.utils import show_array, stage
from yapocis.hsi import split_channels, join_channels, rgb2hsi, hsi2rgb
from yapocis.median import median3x3fast, median3x3slow

def test_gaussian_1d():
    "Make sure that the kernel sums back to 1"
    g = gauss_1d()
    assert are_close(g.sum(), 1.0)


def test_gaussians():
    mw = 100
    gs = gaussians(maxwidth=mw)
    for g in gs:
        assert len(g) < mw


def test_kernels():
    for scale in range(get_scales()):
        gkernel = get_gaussian_kernel(scale)
        a = np.zeros((256,256), dtype=np.float32)
        a[127,127] = 1.0
        b = gkernel(1,a)
        gsum = get_gaussian(scale).sum()
        assert are_close(b[127, :].sum(), gsum)
        b = gkernel(0,a)
        assert are_close(b[:, 127].sum(), gsum)
        # Compare in order of lowest precision
        assert are_close(a.sum(), 1.0)


def test_gauss_image():
    for scale in range(get_scales()):
        a = np.zeros((400,700), dtype=np.float32)
        a[200,:] = 1.0
        a[:,200] = 1.0
        for xy in range(400):
            a[xy,xy] = 1.0
        b = gauss_image(a, scale)
        show_array("GI%s" % scale, b)


def test_zcs():
    for res in (False, True):
        subtotal = 0.0
        for scale in range(get_scales() - 1):
            a = np.zeros((1200,1800), dtype=np.float32)
            a[200,:] = 1.0
            a[:,200] = 1.0
            t = time.time()
            zca,grad,theta = zcsdog(a,scale)
            subtotal += (time.time()-t)
            show_array("org %s" % scale, a)
            show_array("zcs %s" % scale, zca)
            show_array("grad %s" % scale, grad)
            show_array("theta %s" % scale, theta)
        print("Res", res, "seconds", subtotal)


def test_gradient():
    a = np.random.sample((1000,1000)).astype(np.float32)
    t = time.time()
    b = np.gradient(a)
    print("Numpy seconds", time.time()-t)
    for engine in (kernels.GPU_ENGINE, kernels.CPU_ENGINE):
        program = kernels.load_program(interfaces.gradient, engine=engine)
        t = time.time()
        c = program.gradient(a, 1)
        print("Engine %s seconds %s" % (engine,time.time()-t))

def test_hsi():
    r = np.empty((256, 256), dtype=np.float32)
    g = np.empty_like(r)
    b = np.empty_like(r)
    for i in range(256):
        r[i, :] = i
        g[:, i] = i
        b[i, :] = 255 - i
    rgb = join_channels(r, g, b)
    rgb /= 255.0
    show_array("Test data", rgb)
    rgb2 = join_channels(*split_channels(rgb))
    show_array("Test split and join", rgb2)
    h, s, i = rgb2hsi(*split_channels(rgb))
    show_array("I", i)
    show_array("H", h)
    show_array("S", s)
    print(program)

def test_median3():
    a1 = np.random.sample((500,500)).astype(np.float32)
    a2 = a1.copy()
    stage("slow")
    b1 = median3x3slow(a1,500)
    stage()
    stage("fast")
    b2 = median3x3fast(a2,500)
    stage()
    error = np.sum(np.abs(b1.flatten()-b2.flatten()))
    assert error == 0.0


def test_operators():
    # 1+0 -> 1
    a = np.ones((100,100), dtype=np.float32)
    b = np.zeros_like(a)
    c = add(a,b)
    d = np.empty_like(a)
    add_res(a,b,d)
    assert c.sum() == c.size*1.0
    program.read(d)
    assert a.sum() == c.sum()
    assert a.sum() == d.sum()
    # 1-1 == 0
    b[:,:] = 1.0
    c = sub(a,b)
    d[:,:]=10
    sub_res(a,b,d)
    program.read(d)
    assert c.sum() == 0.0
    assert c.sum() == d.sum()
    # 1*1 = 1
    c = mul(a,b)
    d[:,:]=10
    mul_res(a,b,d)
    program.read(d)
    assert a.sum() == c.sum()
    assert a.sum() == d.sum()
    # 1/1 = 1
    #c = div(a,b)
    #d[:,:]=10
    #div_res(a,b,d)
    #program.read(d)
    #assert a.sum() == c.sum()
    #assert a.sum() == d.sum()

def test_xy():
    a = np.zeros((402, 798), dtype=np.int32)
    addr, x, y, label = xy.addr(a)
    show_array("addr", addr)
    show_array("x", x)
    show_array("y", y)
    show_array("label", label)

def test_zcs():
    image = np.zeros((512,512), dtype=np.float32)
    image[255,255] = 1.0
    smaller = gauss_image(image, 2)
    larger = gauss_image(image, 3)
    dog = smaller-larger
    zca = zcs(dog)
    show_array("smaller", smaller)
    show_array("larger", larger)
    show_array("dog", dog)
    show_array("zca", zca)
