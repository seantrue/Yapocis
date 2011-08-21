
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

