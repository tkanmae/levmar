#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Demonstration of use of analytic Jacobian (curve fitting with an exponential
funtction).

A keyword parameter `jacf` passed to `levmar.levmar()` specifies the Jacobian of
`func`.  `jacf` must be a function or method computing the Jacobian.  If `jacf`
is None (default), then the Jacobian will be numerically approximated.
"""
from __future__ import division
import numpy as np
import levmar


np.random.seed(1234)


def expf(p, x):
    return p[0] * np.exp(-p[1] * x) + p[2]


def jac_expf(p, x):
    jac = np.empty((x.shape[0], 3))
    jac[:, 0] = np.exp(-p[1]*x)
    jac[:, 1] = -p[0] * x * np.exp(-p[1]*x)
    jac[:, 2] = np.ones(x.size)
    return jac


# Create input data
x_mea = np.arange(40, dtype=np.float64)
p_tru = [5.0, 0.1, 1.0]
y_tru = expf(p_tru, x_mea)
y_mea = y_tru + 0.2 * np.random.randn(x_mea.size)

# Initial estimate
p_ini = [1.0, 0.5, 0.5]
# Fitting with analytic Jacobian
p_opt, p_cov, info = levmar.levmar(expf, p_ini, y_mea,
                                   args=(x_mea,), jacf=jac_expf)

# Print the result
print("Expected:")
print("{0[0]:9f} {0[1]:9f} {0[2]:9f}".format(p_tru))
print("Estimate:")
print("{0[0]:9f} {0[1]:9f} {0[2]:9f}".format(p_opt))
print("")
