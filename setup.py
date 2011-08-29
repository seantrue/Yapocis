from setuptools import setup, find_packages
setup(
    name = "yapocis",
    version = "0.2.2",
    packages = find_packages(),
    zip_safe=False,
    # Requirements are in requirements.txt
    package_data = {
        '': ['test.jpg','requirements.txt'],
    },

    # metadata for upload to PyPI
    author = "Sean D. True",
    author_email = "sean.true@gmail.com",
    description = "Yet another PyOpenCL interface specifier",
    license = "PSF",
    keywords = "opencl pyopencl thunk rpc image",
    url = "https://github.com/seantrue/Yapocis/",
    classifiers = ["Development Status :: 3 - Alpha",
                   "Intended Audience :: Developers",
                   "License :: OSI Approved :: Python Software Foundation License",
                   "Topic :: Scientific/Engineering :: Image Recognition",
                   ],
    long_description="""
Yapocis is "Yet another Python OpenCL interface specification".

Memorable. Easy to pronounce and spell. Clearly related to the semantics of the project.

Or not, all the good names are used and I am not in marketing

It includes easy-to-call-from-Python, reasonably performing implementations of the Mandelbrot kernel, 3x3 median filters, image gradients, and HSI <-> RGB image conversions. I use it to develop much less generally useful implementations of bizarre image processing ideas.
"""
)
