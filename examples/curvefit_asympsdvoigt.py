#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
"""Demonstration of use of box constraints.

Curve fitting with an asymmetric psuedo-Voigt function.


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


def psd_voigt(p, x):
    peak, xc, w, m = p
    g = sqrt(__4ln2/pi)/w * np.exp(-__4ln2*((x-xc)/w)**2)
    l = 2/(pi*w) / (1 + 4*((x-xc)/w)**2)
    c = (1-m) * sqrt(__4ln2/pi)/w + m * 2/(pi*w)
    return peak * (((1-m)*g + m*l) / c)


def asym_psd_voigt(p, x):
    peak, xc, w_l, w_r, m, const = p
    f = np.empty(x.size)
    idx = x.searchsorted(xc)
    f[:idx] = psd_voigt((1, xc, w_l, m), x[:idx])
    f[idx:] = psd_voigt((1, xc, w_r, m), x[idx:])
    return peak * f + const


## Create input data
x = np.linspace(-6, 6)
pt = [300, -0.4, 2.0, 3.0, 0.4, 0.1]
yt = asym_psd_voigt(pt, x)
y = yt + np.sqrt(yt) * np.random.randn(x.size)

## Initial estimate
p0 = [100, -1.0, 3.0, 4.0, 0.3, 0.2]
## Ensure the widths are (0, Inf), and the mixing ratio is [0, 1].
bounds = [(None, 5e+2), None, (1e-6, None), (1e-6, None), (1e-6, 1), (-10, 10)]
## Run fitting routine
ret = lvmr.levmar(asym_psd_voigt, p0, y, args=(x,), bounds=bounds)

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
    plt.plot(x, asym_psd_voigt(ret.p, x), 'r-', label='fit')
    plt.legend()
    plt.show()
