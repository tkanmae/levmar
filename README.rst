======
levmar
======

A Python binding to the levmar library.


Description
===========

The levmar is GPL'ed ANSI C implementation of the Levenberg-Marquardt
(LM) optimization algorithm.  The LM algorithm provides a numerical
solution to the problem of minimizing a function over a parameter space
of a function.  The levmar library provides implementation of both
unconstrained and constrained LM algorithms (box, linear equation, and
linear inequality constraints).


Installation
============

Building Levmar requires cython and numpy.

In order to install levmar, simply do::

    $ python -m pip install .


Then, verify a successful installation::

    $ python -m pip install pytest 
    $ pytest -v tests/


Documentation
=============

See docstrings and demo scripts contained in the directory
``./examples``.  Documentation of the levmar library can be found at
http://www.ics.forth.gr/~lourakis/levmar/.


Authors
=======

Takeshi Kanmae <tkanmae@gmail.com>


License
=======

The MIT license applies to all the files except those in
``./levmar-2.6``.  All of the software in ``./levmar-2.6`` and only the
software therein is copyrighted by Manolis Lourakis and is licensed
under the terms and conditions of the GNU General Public License (GPL).
See the file LICENSE.txt.


Resources
=========

* levmar: http://www.ics.forth.gr/~lourakis/levmar/
* Python: http://www.python.org/
* NumPy: http://www.scipy.org/
* Cython: http://www.cython.org/


.. # vim: ft=rst tw=72
