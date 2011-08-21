
Introduction
============

Problem
-------

Image processing is inherently computationally challenging. Lots of data leads to lots of approximations,
shortcuts, and undone experiments. We need code that runs fast, and is easy to develop.

Past solutions
--------------

* CADDR-1 & convolution hardware
* Skipping lots of dedicated array processors
* Intel 3/486, lots of memory, and C/Assembly language
* Pentium computers, Python, and PIL
* Fast computers, Python, and Numpy

Current alternatives
--------------------

* Multi-processing
* GPU based solutions

  * NVidia  & Cuda
  * Everybody else & OpenCL

I don't have an NVidia card. And OpenCL runs directly (and fast) on Intel and AMD CPUs. And, more importantly,
my computer comes *out of the box* with a decent implementation.

Writing in OpenCL is a lot like writing in C. It helps to have other peoples examples, and to remember that premature optimization is
not a good idea. Optimized OpenCL can be hard to read, and can run at near FTL speed. Unoptimized OpenCL can be very easy to read, and can run
10 times faster than Numpy. 
  
PyOpenCL is terrific. The procedure call interface is a little rough: we can do better. Back to the future, with a modified 
Apollo RPC / Apollo NCS RPC, OSF DCE RPC, MS RPC (COM) interface specification (sweet and nutritious) and a tacky but easy to use
Python thunking layer (bitter, but you only have to swallow it once).
