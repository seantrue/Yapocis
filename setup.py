from setuptools import setup, find_packages
setup(
    name = "yapocis",
    version = "0.2.1",
    packages = find_packages(),
    zip_safe=False,
    install_requires = [
        "Mako>=0.4.2",
        "MarkupSafe>=0.15",
        "Pillow>=1.7.4",
        "decorator>=3.3.1",
        "distribute>=0.6.21",
        "numpy>=1.6.1",
        "py>=1.4.5",
        "pyopencl>=2011.1.2",
        "pyparsing>=1.5.6",
        "pytest>=2.1.1",
        "pytools>=2011.3",
        ],

    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['test.jpg'],
    },

    # metadata for upload to PyPI
    author = "Sean D. True",
    author_email = "sean.true@gmail.com",
    description = "Yet another PyOpenCL interface specifier",
    license = "PSF",
    keywords = "opencl pyopencl thunk rpc image",
    url = "https://github.com/seantrue/Yapocis/", 
    long_description="""
Yapocis is "Yet another Python OpenCL interface specification".

Memorable. Easy to pronounce and spell. Clearly related to the semantics of the project.

Or not, all the good names are used and I am not in marketing

It includes easy-to-call-from-Python, reasonably performing implementations of the Mandelbrot kernel, 3x3 median filters, image gradients, and HSI <-> RGB image conversions. I use it to develop much less generally useful implementations of bizarre image processing ideas.
"""
)
