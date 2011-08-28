'''\
interfaces.py
Created on Aug 12, 2011
Copyright (c) 2011, Sean D. True
'''

# Standard demo, interface spec supports c=sum(a,b)
demo = """
interface demo {
    kernel sum(in float32 *a, in float32 *b, outlike a);
    };
"""

# 1D convolution kernel, with parameterized name
convolve = """
    interface convolve {
        kernel ${name}(in int32 width, in float32* a, outlike a);
    };
"""

# Multiple 1D kernels, with name and conv (mask) parameterized
convolves = """
    interface convolves {
%for name,conv in convs:
        kernel ${name}(in int32 width, in float32* a, outlike a);
%endfor 
    };
"""

# 2-D median filter, with support for fast iteration
median3x3 = """
    interface median3x3 {
        kernel median3x3(in int32 width, in int32 rowwidth, in float32* a, out float32* ret );
        alias first as median3x3(in int32 width, in int32 rowwidth, in float32* a, in float32* ret );
        alias step as median3x3(in int32 width, in int32 rowwidth, resident float32* a, resident float32* ret );
        alias last as median3x3(in int32 width, in int32 rowwidth, resident float32* a, out float32* ret );
    };
"""

# Conversion to and from Hue/Saturation/Intensity color space
hsi = """
    interface hsi {
        kernel rgb2hsi(in float32 *r, in float32 *g, in float32 *b, outlike r, outlike r, outlike r, outlike r);
        kernel hsi2rgb(in float32 *h, in float32 *s, in float32 *i, outlike h, outlike h, outlike h);
    };
"""

# Standard Mandelbrot kernel with return value, use as:
# output = mandelbrot(input, iterations)
mandelbrot = """
    interface mandelbrot {
        kernel mandelbrot(in complex64 *q, outlike int16 *q, in int32 maxiter);
};
"""

gradient = """
    interface gradient {
        kernel gradient(in int32 width, in int32 rowwidth, in float32* a, in int32 reach, outlike a, outlike a );
    };
"""
