'''
Created on Aug 12, 2011

@author: seant
'''
boundedmedian = """
    interface boundedmedian {
         kernel boundedmedian(in int32 offset, in float32 *input, in int32 *zcs, outlike input, outlike int16 input);
      };
"""
boundedaverage = """
    interface boundedaverage {
         kernel boundedaverage(in int32 width, in int32 height, in float32 *input, in int32 *zcs, outlike input);
         kernel boundedaverageangle(in int32 width, in int32 height, in float32 *input, in int32 *zcs, outlike input);
      };
"""
convolve = """
    interface convolve {
        kernel ${name}(in int32 width, in float32* a, outlike a);
    };
"""
convolves = """
    interface convolves {
%for name,conv in convs:
        kernel ${name}(in int32 width, in float32* a, outlike a);
%endfor 
    };
"""

applysegments = """
    interface applysegments {
        kernel applysegments(in int32 *segments,  in int32 width,  in int32 height, in float32 maxval, in float32 *img, out float32 *output);
        };
""" 

averagesegments = """
    interface averagesegments {
        kernel averagesegments(in int32 *segments,  in int32 width,  in int32 height, in float32 maxval, in float32 *img, out float32 *output);
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

label = """
    interface label {
        kernel label(in int32 width, in int32 rowwidth, in int32* a, in int32 *zcs, out int32* ret);
    };
"""


oldhsi = """
    interface hsi {
        kernel rgb2hsi(in float32 *r, in float32 *g, in float32 *b, out float32 *h, out float32 * s, out float32 *i);
        kernel hsi2rgb(in float32 *h, in float32 *s, in float32 *i, out float32 *r, out float32 *g, out float32 *b);
    };
"""

hsi = """
    interface hsi {
        kernel rgb2hsi(in float32 *r, in float32 *g, in float32 *b, outlike r, outlike r, outlike r, outlike r);
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
