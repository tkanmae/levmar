#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
"""Demonstration of use of box constraints.

Curve fitting with a Gauss function.


A keyword parameter `bounds` is for box constraints lb[i]<=p[i]<=ub[i].
`bounds` must be a tuple/list, and its length must be the same as the
parameters.  A constraint for a parameter consists of a tuples of two
floats/Nones or Nones.  A tuple determines the lower and upper bound,
and a None means no constraint.  If one of two values in a tuple is
None, then the bound is semi-definite.  For example,

    >>> bounds = [None, (-10, 3), (None, 10), (0, None)]

specifies that the first parameter has no constraint, the second has -10
and 3 for the lower and upper bound respectively, the third has only the
upper bound of 10, and the fourth has to be greater than or equal to 0.
"""
from math import *
import numpy as np
try:
    from matplotlib import pyplot as plt
    has_mpl = True
except ImportError:
    has_mpl = False
import lvmr

## If you prefer to reproduce the result, set a seed to the generator.
# np.random.seed(1234)

__4ln2 = 4 * log(2)


def gauss(p, x):
    y = p[0] * np.exp(-__4ln2*((x-p[1])/p[2])**2) + p[3]
    return y


## Create input data
x = np.linspace(-5, 5)
pt = [250, -0.4, 2.0, 0.1]
yt = gauss(pt, x)
y = yt + np.sqrt(yt) * np.random.randn(x.size)

## Initial estimate
p0 = [1.0, -3.0, 3.0, 1.0]
## Ensure the width is (0, Inf).
bounds = [None, None, (1e-6, None), (-10, 10)]
## Run fitting routine
ret = lvmr.levmar(gauss, p0, y, args=(x,), bounds=bounds)

print ':Expected:'
print pt
print ':Estimate:'
print ret.p
print
## Print summary of the fitting
print ' Summary '.center(60, '*')
print ret
print ''.center(60, '*')
## Plot the result
if has_mpl:
    plt.plot(x, y, 'bo')
    plt.plot(x, yt, 'b-', label='true')
    plt.plot(x, gauss(ret.p, x), 'r-', label='fit')
    plt.legend()
    plt.show()
