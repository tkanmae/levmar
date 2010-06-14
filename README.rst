======================================================
PyLVMR: A Python binding to the levmar library
======================================================

PyLVMR is a Python_ binding to the levmar_ library.  The levmar_ is GPL'ed
ANSI C implementation of the Levenberg-Marquardt (LM) optimization algorithm.
The LM algorithm provides a numerical solution to the problem of minimizing a
function over a parameter space of a function.  The levmar_ library provides
implementation of both unconstrained and constrained LM algorithms (box, linear
equation, and linear inequality constraints).


Installation
============

Prerequisites
-------------

Building PyLVMR requires the following software installed:

1) Python 2.6, or higher
2) Cython 0.12, or higher
3) NumPy 1.3, or higher
4) [optional] SciPy 0.7, or higher
5) [optional] nose 0.11, or higher


Installing PyLVMR
-----------------

::

    # cd lvmr
    # cython -v lvmr/lvmr/_lvmr.pyx
    # python setup.py build
    # python setup.py install


Testing PyLVMR
--------------

To run the tests, you need to have nose_ installed.

::

    >>> import lvmr
    >>> lvmr.test()


Documentation
=============

See docstrings and demo scripts contained in the directory ``./lvmr/demos``.
Documentation of the levmar_ library can be found at
http://www.ics.forth.gr/~lourakis/levmar/.


Authors
=======

Takeshi Kanmae


License
=======

The MIT license applies to all the files except those in ./levmar-2.5.  All of
the software in ./levmar-2.5 and only the software therein is copyrighted by
Manolis Lourakis and is licensed under the terms and conditions of the GNU
General Public License (GPL).  See the file COPYING.


.. _levmar: http://www.ics.forth.gr/~lourakis/levmar/
.. _Python: http://www.python.org/
.. _nose: http://somethingaboutorange.com/mrl/projects/nose 


.. # vim: ft=rst
