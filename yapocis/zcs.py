import numpy as np
from rpc import interfaces, kernels

program = kernels.loadProgram(interfaces.zcs)

zcs = program.zcs
zcs_res = program.zcs_res

def test_zcs():
    from utils import showArray
    from gaussian import gaussImage
    image = np.zeros((512,512), dtype=np.float32)
    image[255,255] = 1.0
    smaller = gaussImage(image, 2)
    larger = gaussImage(image, 3)
    dog = smaller-larger
    zca = zcs(dog)
    showArray("smaller", smaller)
    showArray("larger", larger)
    showArray("dog", dog)
    showArray("zca", zca)

if __name__ == "__main__":
    test_zcs()
