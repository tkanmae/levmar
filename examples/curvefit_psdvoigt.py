#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
"""Demostration of use of box constraints.

Curve ftting with a psuedo-Voigt function.

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
from math import *
import numpy as np
try:
    from matplotlib import pyplot as plt
    has_mpl = True
except ImportError:
    has_mpl = False

import lvmr


__4ln2 = 4 * log(2)


def psd_voigt(p, x):
    peak, xc, w, m, const = p
    g = sqrt(__4ln2/pi)/w * np.exp(-__4ln2*((x-xc)/w)**2)
    l = 2/(pi*w) / (1 + 4*((x-xc)/w)**2)
    c = (1-m) * sqrt(__4ln2/pi)/w + m * 2/(pi*w)
    return peak * (((1-m)*g + m*l) / c) + const


x = np.linspace(-6, 6)
pt = [250, -0.4, 2.0, 0.4, 0.1]
yt = psd_voigt(pt, x)
y = yt + np.sqrt(yt) * np.random.randn(x.size)

p0 = [100, -3.0, 3.0, 0.2, 0.2]

## Ensure the width is positive, and the mixing ratio is [0, 1].
bounds = [None, None, (0, None), (0, 1), (-1, 1)]

ret = lvmr.levmar(psd_voigt, p0, y, args=(x,), bounds=bounds)

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
    plt.plot(x, psd_voigt(ret.p, x), 'r-', label='fit')
    plt.legend()
    plt.show()
