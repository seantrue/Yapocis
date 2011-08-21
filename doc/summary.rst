
Summary
=======

Yapocis is a Python package that makes it easier to call high performance OpenCL kernels
from high level Python code. It depends on, and was inspired by, PyOpenCL which does the
hard work of making OpenCL callable at all. Yapocis is intended to make calling it much 
less painful.

The code is currently developed and tested on OS/X Lion. It was originally developed on Snow Leopard,
and there were some code changes required to jump forward. Those changes are entirely in the OpenCL
kernel code itself, and adapting kernel interfaces to Yapocis should usually start with already
functioning OpenCL code.


Quick look
----------

And here's a fast comparison between three ways of getting an embarassingly parallel thing 
done from Python:

First the easy way, with numpy:

.. literalinclude:: /../examples/demo_numpy.py
   :language: python
   :linenos:
   
First the easy and fast way, with yapocis:

.. literalinclude:: /../examples/demo_yapocis.py
   :language: python
   :linenos: 
   
First the hard and fast way, with raw pyopencl:

.. literalinclude:: /../examples/demo_opencl.py
   :language: python
   :linenos:     