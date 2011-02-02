#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
"""Demonstration of use of analytic Jacobian.

Curve fitting with a exponential function.

A keyword parameter `jacf` passed to `levmar.levmar()` specifies the
Jacobian of `func`.  `jacf` must be a function or method computing the
Jacobian.  It takes, at least, one length of m vector and returns the
(nxm) Jacobian matrix or a compatible C-contiguous vector.  If `jacf` is
None, then the Jacobian will be approximated.
"""
import numpy as np
import levmar

# If you prefer repeatable result, set a seed to the generator.
# np.random.seed(1234)

def expf(p, x):
    return p[0] * np.exp(-p[1]*x) + p[2]


def jac_expf(p, x):
    jac = np.empty((x.shape[0], 3))
    jac[:,0] = np.exp(-p[1]*x)
    jac[:,1] = -p[0] * x * np.exp(-p[1]*x)
    jac[:,2] = np.ones(x.size)
    return jac


# Create input data
x = np.arange(40, dtype=np.float64)
pt = [5.0, 0.1, 1.0]
yt = expf(pt, x)
y = yt + 0.2 * np.random.randn(x.size)

# Initial estimate
p0 = [1.0, 0.5, 0.5]
# Fitting with analytic Jacobian
output = levmar.levmar(expf, p0, y, jacf=jac_expf, args=(x,), full_output=True)


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
    plt.plot(x, expf(output.p, x), 'r-', label='fit')
    plt.legend()
    plt.show()
except ImportError:
    pass
