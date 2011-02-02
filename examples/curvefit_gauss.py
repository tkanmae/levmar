#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
"""Demonstration of use of box constraints.

Curve fitting with a Gauss function.


A keyword parameter `bounds` passed to  `levmar.levmar()` specifies box
constraints lb[i]<=p[i]<=ub[i].  `bounds` must be a tuple/list, and its
length must be the same as the parameters.  A constraint for a parameter
consists of a tuples of two floats/Nones or None.  A tuple determines
the (inclusive) lower and upper bound, and None means no constraint.  If
one of two values in a tuple is None, then the bound is semi-definite.
For example,

    >>> bounds = [None, (-10, 3), (None, 10), (0, None)]

specifies that the first parameter has no constraint, the second has -10
and 3 for the lower and upper bound respectively, the third has only the
upper bound of 10, and the fourth has to be greater than or equal to 0.
"""
from math import log
import numpy as np
import levmar

# If you prefer repeatable result, set a seed to the generator.
# np.random.seed(1234)

_4ln2 = 4 * log(2)


def gauss(p, x):
    return p[0] * np.exp(-_4ln2*((x-p[1])/p[2])**2) + p[3]


# Create input data
x = np.linspace(-5, 5)
pt = [230, -0.4, 2.0, 0.1]
yt = gauss(pt, x)
y = yt + np.sqrt(yt) * np.random.randn(x.size)

# Initial estimate
p0 = (1.0, -3.0, 3.0, 1.0)
# Ensure the width is (0, Inf).
bounds = [None, None, (1e-6, None), (-10, 10)]
# Run the fitting routine
output = levmar.levmar(gauss, p0, y, args=(x,), bounds=bounds, full_output=True)


# Print the result
print(":Expected:")
print("{0[0]:9f} {0[1]:9f} {0[2]:9f}".format(pt))
print(":Estimate:")
print("{0[0]:9f} {0[1]:9f} {0[2]:9f}".format(output.p))
print("")

print(" Summary ".center(60, '*'))
output.pprint()
print(''.center(60, '*'))


# Plot the result
try:
    import matplotlib.pyplot as plt

    plt.plot(x, y, 'bo')
    plt.plot(x, yt, 'b-', label='true')
    plt.plot(x, gauss(output.p, x), 'r-', label='fit')
    plt.legend()
    plt.show()
except ImportError:
    pass
