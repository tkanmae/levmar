#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Demonstration of use of box constraints (curve fitting with Gaussian).

`bc` passed to `levmar.levmar_bc()` specifies box constraints for the
parameters. `bc` is pairs of `(min, max)` for each element of the parameters,
specifying the (inclusive) lower and upper boundss.  Use None for one of `min`
or `max` for specifying no bound in that direction.  For example,

    >>> bc = [(None, None), (-10, 3), (None, 10), (0, None)]

specifies that the first parameter has no constraint, the second has -10
and 3 for the lower and upper bound respectively, the third has only the
upper bound of 10, and the fourth has only the lower bound of 0.

"""
from __future__ import division

from math import log
import numpy as np
import levmar


np.random.seed(1234)


def gauss(p, x):
    return p[0] * np.exp(-4 * log(2) * ((x - p[1]) / p[2])**2) + p[3]


# Create input data
x_mea = np.linspace(-5, 5)
p_tru = [230, -0.4, 2.0, 0.1]
y_tru = gauss(p_tru, x_mea)
y_mea = y_tru + np.sqrt(y_tru) * np.random.randn(x_mea.size)

# Initial estimate
p_ini = [1.0, -3.0, 3.0, 1.0]
# Ensure the width is (0, Inf).
bc = [(None, None), (None, None), (1.0e-6, None), (-10.0, 10.0)]
# Run the fitting routine
p_opt, p_cov, info = levmar.levmar_bc(gauss, p_ini, y_mea, bc, args=(x_mea,))

# Print the result
print("Expected:")
print("  {0[0]:9f} {0[1]:9f} {0[2]:9f}".format(p_tru))
print("Estimate:")
print("  {0[0]:9f} {0[1]:9f} {0[2]:9f}".format(p_opt))
print("")
