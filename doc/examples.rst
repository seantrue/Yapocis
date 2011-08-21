
Short examples
==============

First, a kernel that implements a 3x3 median filter on a 2-d array:

.. literalinclude:: /../rpc/median3x3.mako
   :language: c
   :linenos:
   

Now, the wrapper code for loading that in an even prettier way:
   
.. literalinclude:: /../median.py
   :language: python
   :linenos:

And a driver program that compares running the median filter 10 times
on each of an increasing series of image sizes

.. literalinclude:: /../examples/median3x3_comparison.py
   :language: python
   :linenos: