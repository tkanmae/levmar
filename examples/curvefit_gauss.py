#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
"""Demostration of use of box constraints.

Curve ftting with a Gauss function.


A keyword parameter `bounds` is for box constraints.  `bounds` must be a
tuple/list consisting of tuples of two floats/None or Nones.  Its length
must be the same as the parameters.  A tuple specifies the lower and
upper bound for a parameter, and a None means no constraint.  If one of
two values in a tuple is None, then the bound is semi-definite.  For
example,

    >>> bounds = [None, (-10, 3), (None, 10), (0, None)]

specifies that the first parameter has no constraint, the second has -10
and 3 for the lower and upper bound respectively, the third has only the
upper bound of 10, and the firth has to be larger than 0.
"""
import numpy as np
try:
    from matplotlib import pyplot as plt
    has_mpl = True
except ImportError:
    has_mpl = False

import lvmr


def gauss(p, x):
    y = p[0] * np.exp(-((x-p[1])/p[2])**2) + p[3]
    return y


x = np.linspace(-5, 5)
pt = [250, -0.4, 2.0, 0.1]
yt = gauss(pt, x)
y = yt + np.sqrt(yt) * np.random.randn(x.size)

p0 = [1.0, -3.0, 3.0, 1]

## Ensure the width is positive.
bounds = [None, None, (0, None), None]

ret = lvmr.levmar(gauss, p0, y, args=(x,), bounds=bounds)

print ':Expected:'
print pt
print ':Estimate:'
print ret.p
print

print ' Summary '.center(60, '*')
print ret
print ''.center(60, '*')

if has_mpl:
    plt.plot(x, y, 'bo')
    plt.plot(x, yt, 'b-', label='true')
    plt.plot(x, gauss(ret.p, x), 'r-', label='fit')
    plt.legend()
    plt.show()
