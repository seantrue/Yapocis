'''
Created on Aug 12, 2011

@author: seant
'''
convolve = """
    interface convolve {
        kernel ${name}(in int32 width, in float32* a, out float32* ret);
    };
"""
convolves = """
    interface convolves {
%for name,conv in convs:
        kernel ${name}(in int32 width, in float32* a, out float32* ret);
%endfor 
    };
"""

hmedian = """
    interface hmedian {
        kernel ${name}(in int32 width, in float32* a, out float32* ret );
    };
"""

median3x3 = """
    interface median3x3 {
        kernel median3x3(in int32 width, in int32 rowwidth, in float32* a, out float32* ret );
    };
"""

# Two definitions for hsi: One returns by reference, other is a pythonic return
oldhsi = """
    interface hsi {
        kernel rgb2hsi(in float32 *r, in float32 *g, in float32 *b, out float32 *h, out float32 * s, out float32 *i);
        kernel hsi2rgb(in float32 *h, in float32 *s, in float32 *i, out float32 *r, out float32 *g, out float32 *b);
    };
"""

hsi = """
    interface hsi {
        kernel rgb2hsi(in float32 *r, in float32 *g, in float32 *b, outlike r, outlike r, outlike r);
        kernel hsi2rgb(in float32 *h, in float32 *s, in float32 *i, outlike h, outlike h, outlike h);
    };
"""

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
