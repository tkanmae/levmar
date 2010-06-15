==============================================
PyLVMR: A Python binding to the levmar library
==============================================

PyLVMR is a Python_ binding to the levmar_ library.  The levmar_ is GPL'ed
ANSI C implementation of the Levenberg-Marquardt (LM) optimization algorithm.
The LM algorithm provides a numerical solution to the problem of minimizing a
function over a parameter space of a function.  The levmar_ library provides
implementation of both unconstrained and constrained LM algorithms (box, linear
equation, and linear inequality constraints).


Installation
============

0. Prerequisites
----------------

Building PyLVMR requires the following software installed:

* Python_ >= 2.6
* Cython_ >= 0.12
* NumPy_ >= 1.3
* [optional] SciPy_ >= 0.7
* [optional] setuptools_ >= 0.6
* [optional] nose_ >= 0.11


1. Building PyLVMR
------------------

::

    $ cd lvmr
    $ cython -v lvmr/lvmr/_lvmr.pyx
    $ python setup.py build


3. Testing PyLVMR
-----------------

To run the tests, you need to have setuptools_ and nose_ installed.

::

    $ python setup.py test


4. Installing PyLVMR
--------------------

::

    $ python setup.py install

Then, verify a successful installation of PyLVMR:

::

    $ python
    >>> impport lvmr
    >>> lvmr.test()


Documentation
=============

See docstrings and demo scripts contained in the directory ``./lvmr/demos``.
Documentation of the levmar_ library can be found at
http://www.ics.forth.gr/~lourakis/levmar/.


Authors
=======

Takeshi Kanmae <tkanmae@gmail.com>


License
=======

The MIT license applies to all the files except those in ``./levmar-2.5``.  All
of the software in ``./levmar-2.5`` and only the software therein is
copyrighted by Manolis Lourakis and is licensed under the terms and conditions
of the GNU General Public License (GPL).  See the file COPYING.


.. _levmar: http://www.ics.forth.gr/~lourakis/levmar/
.. _Python: http://www.python.org/
.. _Cython: http://www.cython.org/
.. _NumPy: http://www.scipy.org/
.. _Scipy: http://www.scipy.org/
.. _setuptools: http://peak.telecommunity.com/DevCenter/setuptools
.. _nose: http://somethingaboutorange.com/mrl/projects/nose


.. # vim: ft=rst
