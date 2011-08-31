'''\
interfaces.py
Created on Aug 12, 2011
Copyright (c) 2011, Sean D. True
'''

# Standard demo, interface spec supports c=sum(a,b)
demo = """
interface demo {
    kernel sum(in float *a, in float *b, outlike a);
    };
"""

# 1D convolution kernel, with parameterized name
convolve = """
    interface convolve {
        kernel ${name}(sizeof a, in float* a, outlike a);
    };
"""

# Multiple 1D kernels, with name and conv (mask) parameterized
convolves = """
    interface convolves {
%for name,conv in convs:
        kernel ${name}(sizeof a, in float* a, outlike a);
%endfor 
    };
"""

# 2-D median filter, with support for fast iteration
median3x3 = """
    interface median3x3 {
        kernel median3x3(sizeof int a, heightof int a, in float* a, outlike a );
        alias first as median3x3(sizeof int a, heightof int a, in float* a, in float* ret );
        alias step as median3x3(sizeof int a, heightof int a, resident float* a, resident float* ret );
        alias last as median3x3(sizeof int a, heightof int a, resident float* a, out float* ret );
    };
"""

# Conversion to and from Hue/Saturation/Intensity color space
hsi = """
    interface hsi {
        kernel rgb2hsi(in float *r, in float *g, in float *b, outlike r, outlike r, outlike r, outlike r);
        kernel hsi2rgb(in float *h, in float *s, in float *i, outlike h, outlike h, outlike h);
    };
"""

# Standard Mandelbrot kernel with return value, use as:
# output = mandelbrot(input, iterations)
mandelbrot = """
    interface mandelbrot {
        kernel mandelbrot(in complex64 *q, outlike short *q, in int maxiter);
};
"""

gradient = """
    interface gradient {
        kernel gradient(sizeof int a, heightof int a, in float* a, in int reach, outlike a, outlike a );
    };
"""
