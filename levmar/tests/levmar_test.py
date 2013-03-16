#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

from math import (atan, pi, sqrt)
import numpy as np
from numpy.testing import assert_array_almost_equal
import levmar


OPTS = dict(eps1=1.0e-15, eps2=1.0e-15, eps3=1.0e-20)


def test_rosen():
    # Rosenbrock function
    def func(p):
        y = np.empty(2)
        y[0] = 1.0 - p[0]
        y[1] = 10.0 * (p[1] - p[0]*p[0])
        return y

    def jacf(p):
        j = np.empty((2, 2))
        j[0, 0] = -1.0
        j[0, 1] = 0.0
        j[1, 0] = -20.0 * p[0]
        j[1, 1] = 10.0
        return j

    y = np.zeros(2)
    p0 = [-1.2, 1.0]
    pt = [ 1.0, 1.0]

    p, pcov, info = levmar.levmar(func, p0, y, jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar(func, p0, y, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar(func, p0, y, cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt)


def test_modros():
    # Modified Rosenbrock problem
    MODROSLAM = 1E+02

    def func(p):
        y = np.empty(3)
        y[0] = 10.0 * (p[1] - p[0]*p[0])
        y[1] = 1.0 - p[0]
        y[2] = MODROSLAM
        return y

    def jacf(p):
        j = np.empty((3, 2))
        j[0, 0] = -20.0 * p[0]
        j[0, 1] = 10.0
        j[1, 0] = -1.0
        j[1, 1] = 0.0
        j[2, 0] = 0.0
        j[2, 1] = 0.0
        return j

    y = np.zeros(3)
    p0 = [-1.2, 2.0]
    pt = [ 1.0, 1.0]

    p, pcov, info = levmar.levmar(func, p0, y, jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar(func, p0, y, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar(func, p0, y, cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt)


def test_powell():
    # Powell's function.
    def func(p):
        y = np.empty(2)
        y[0] = p[0]
        y[1] = 10.0 * p[0] / (p[0] + 0.1) + 2.0 * p[1]*p[1]
        return y

    def jacf(p):
        j = np.empty((2, 2))
        j[0, 0] = 1.0
        j[0, 1] = 0.0
        j[1, 0] = 1.0 / ((p[0] + 0.1) * (p[0] + 0.1))
        j[1, 1] = 4.0 * p[1]
        return j

    y = np.zeros(2)
    p0 = [3.0, 1.0]
    pt = [0.0, 0.0]

    p, pcov, info = levmar.levmar(func, p0, y, jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt, decimal=4)
    p, pcov, info = levmar.levmar(func, p0, y, **OPTS)
    assert_array_almost_equal(p, pt, decimal=4)
    p, pcov, info = levmar.levmar(func, p0, y, cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt, decimal=4)


def test_wood():
    # Wood's function
    def func(p):
        p0, p1, p2, p3 = p
        y = np.empty(6)
        y[0] = 10.0 * (p1 - p0*p0)
        y[1] = 1.0 - p0
        y[2] = sqrt(90.0) * (p3 - p2*p2)
        y[3] = 1.0 - p2
        y[4] = sqrt(10) * (p1 + p3 - 2.0)
        y[5] = (p1 - p3) / sqrt(10.0)
        return y

    y = np.zeros(6)
    p0 = [-3.0, -1.0, -3.0, -1.0]
    pt = [ 1.0,  1.0,  1.0,  1.0]

    p, pcov, info = levmar.levmar(func, p0, y, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar(func, p0, y, cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt)


def test_meyer():
    # Meyer's (reformulated) problem
    def func(p, x):
        y = p[0] * np.exp(10.0 * p[1] / (x + p[2]) - 13.0)
        return y

    def jacf(p, x):
        p0, p1, p2 = p
        j = np.empty((16, 3))
        tmp = np.exp(10.0 * p1 / (x + p2) - 13.0)
        j[:,0] = tmp
        j[:,1] = 10.0 * p0 * tmp / (x + p2)
        j[:,2] = -10.0 * p0 * p1 * tmp / ((x + p2) * (x + p2))
        return j

    x = 0.45 + 0.05 * np.arange(16)
    y = np.asarray([34.780, 28.610, 23.650, 19.630,
                    16.370, 13.720, 11.540,  9.744,
                     8.261,  7.030,  6.005,  5.147,
                     4.427,  3.820,  3.307,  2.872])
    p0 = [8.85, 4.00, 2.50]
    pt = [2.48, 6.18, 3.45]

    p, pcov, info = levmar.levmar(func, p0, y, args=(x,), jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt, decimal=1)
    p, pcov, info = levmar.levmar(func, p0, y, args=(x,), **OPTS)
    assert_array_almost_equal(p, pt, decimal=1)
    p, pcov, info = levmar.levmar(func, p0, y, args=(x,), cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt, decimal=1)


def test_osborne():
    # Osborne's problem
    def func(p, x):
        y = p[0] + p[1] * np.exp(-p[3] * x) + p[2] * np.exp(-p[4] * x)
        return y

    def jacf(p, x):
        j= np.empty((33, 5))
        tmp1 = np.exp(-p[3] * x)
        tmp2 = np.exp(-p[4] * x)
        j[:, 0] = 1.0
        j[:, 1] = tmp1
        j[:, 2] = tmp2
        j[:, 3] = -p[1] * x * tmp1
        j[:, 4] = -p[2] * x * tmp2
        return j

    x = 10.0 * np.arange(33)
    y = np.asarray([8.44e-1, 9.08e-1, 9.32e-1, 9.36e-1, 9.25e-1,
                    9.08e-1, 8.81e-1, 8.50e-1, 8.18e-1, 7.84e-1,
                    7.51e-1, 7.18e-1, 6.85e-1, 6.58e-1, 6.28e-1,
                    6.03e-1, 5.80e-1, 5.58e-1, 5.38e-1, 5.22e-1,
                    5.06e-1, 4.90e-1, 4.78e-1, 4.67e-1, 4.57e-1,
                    4.48e-1, 4.38e-1, 4.31e-1, 4.24e-1, 4.20e-1,
                    4.14e-1, 4.11e-1, 4.06e-1])
    p0 = [0.5, 1.5, -1.0, 1.0e-2, 2.0e-2]
    pt = [0.3754, 1.9358, -1.4647, 0.0129, 0.0221]

    p, pcov, info = levmar.levmar(func, p0, y, args=(x,), jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt, decimal=4)
    p, pcov, info = levmar.levmar(func, p0, y, args=(x,), **OPTS)
    assert_array_almost_equal(p, pt, decimal=4)
    p, pcov, info = levmar.levmar(func, p0, y, args=(x,), cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt, decimal=4)


def test_helval():
    # Helical valley function
    def func(p):
        p0, p1, p2 = p
        y = np.empty(3)
        if p0 < 0:
            theta = atan(p1 / p0) / (2 * pi) + 0.5
        elif p0 > 0:
            theta = atan(p1 / p0) / (2 * pi)
        else:
            theta = 0.25 if p1 >= 0 else -0.25
        y[0] = 10.0*(p2 - 10.0 * theta)
        y[1] = 10.0*(sqrt(p0*p0 + p1*p1) - 1.0)
        y[2] = p2
        return y

    def jacf(p):
        p0, p1, p2 = p
        j = np.empty((3, 3))
        tmp = p0*p0 + p1*p1
        j[0, 0] =  50.0 * p1 / (pi * tmp)
        j[0, 1] = -50.0 * p0 / (pi * tmp)
        j[0, 2] = 10.0
        j[1, 0] = 10.0 * p0 / sqrt(tmp)
        j[1, 1] = 10.0 * p1 / sqrt(tmp)
        j[1, 2] = 0.0
        j[2, 0] = 0.0
        j[2, 1] = 0.0
        j[2, 2] = 1.0
        return j

    y = np.zeros(3)
    p0 = [-1.0, 2.0, 2.0]
    pt = [ 1.0, 0.0, 0.0]

    p, pcov, info = levmar.levmar(func, p0, y, jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar(func, p0, y, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar(func, p0, y, cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt)


def test_bt3():
    # Boggs-Tolle problem 3 (linearly constrained)
    def func(p):
        y = np.empty(5)
        t1 = p[0] - p[1]
        t2 = p[1] + p[2] - 2.0
        t3 = p[3] - 1.0
        t4 = p[4] - 1.0
        y[:] = t1*t1 + t2*t2 + t3*t3 + t4*t4
        return y

    def jacf(p):
        j = np.empty((5, 5))
        t1 = p[0] - p[1]
        t2 = p[1] + p[2] - 2.0
        t3 = p[3] - 1.0
        t4 = p[4] - 1.0
        j[:, 0] = 2.0 * t1
        j[:, 1] = 2.0 * (t2 - t1)
        j[:, 2] = 2.0 * t2
        j[:, 3] = 2.0 * t3
        j[:, 4] = 2.0 * t4
        return j

    y = np.zeros(5)
    p0 = [2.0, 2.0, 2.0, 2.0, 2.0]
    pt = [-0.76744, 0.25581, 0.62791, -0.11628, 0.25581]
    A = np.asarray([[1.0, 3.0, 0.0, 0.0,  0.0],
                    [0.0, 0.0, 1.0, 1.0, -2.0],
                    [0.0, 1.0, 0.0, 0.0, -1.0]])
    b = np.asarray([0.0, 0.0, 0.0])

    p, pcov, info = levmar.levmar_lec(func, p0, y, (A, b), jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt, decimal=4)
    p, pcov, info = levmar.levmar_lec(func, p0, y, (A, b), **OPTS)
    assert_array_almost_equal(p, pt, decimal=4)
    p, pcov, info = levmar.levmar_lec(func, p0, y, (A, b), cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt, decimal=4)


def test_hs28():
    # Hock-Schittkowski problem 28 (linearly constrained)
    def func(p):
        y = np.empty(3)
        t1 = p[0] + p[1]
        t2 = p[1] + p[2]
        y[:] = t1*t1 + t2*t2
        return y

    def jacf(p):
        j = np.empty((3, 3))
        t1 = p[0] + p[1]
        t2 = p[1] + p[2]
        j[:, 0] = 2.0 * t1
        j[:, 1] = 2.0 * (t1 + t2)
        j[:, 2] = 2.0 * t2
        return j

    y = np.zeros(3)
    p0 = [-4.0, 1.0, 1.0]
    pt = [ 0.5, -0.5, 0.5]
    A = np.asarray([1.0, 2.0, 3.0])
    b = np.asarray([1.0])

    p, pcov, info = levmar.levmar_lec(func, p0, y, (A, b), jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt, decimal=5)
    p, pcov, info = levmar.levmar_lec(func, p0, y, (A, b), **OPTS)
    assert_array_almost_equal(p, pt, decimal=4)
    p, pcov, info = levmar.levmar_lec(func, p0, y, (A, b), cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt, decimal=3)


def test_hs48():
    # Hock-Schittkowski problem 48 (linearly constrained)
    def func(p):
        y = np.empty(5)
        t1 = p[0] - 1.0
        t2 = p[1] - p[2]
        t3 = p[3] - p[4]
        y[:] = t1*t1 + t2*t2 + t3*t3
        return y

    def jacf(p):
        j = np.empty((5, 5))
        t1 = p[0] - 1.0
        t2 = p[1] - p[2]
        t3 = p[3] - p[4]
        j[:, 0] =  2.0*t1
        j[:, 1] =  2.0*t2
        j[:, 2] = -2.0*t2
        j[:, 3] =  2.0*t3
        j[:, 4] = -2.0*t3
        return j

    y = np.zeros(5, np.float64)
    p0 = [3.0, 5.0, -3.0, 2.0, -2.0]
    pt = [1.0, 1.0, 1.0, 1.0, 1.0]
    A = np.asarray([[1.0, 1.0, 1.0,  1.0,  1.0],
                    [0.0, 0.0, 1.0, -2.0, -2.0]])
    b = np.asarray([5.0, -3.0])

    p, pcov, info = levmar.levmar_lec(func, p0, y, (A, b), jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt, decimal=5)
    p, pcov, info = levmar.levmar_lec(func, p0, y, (A, b), **OPTS)
    assert_array_almost_equal(p, pt, decimal=5)
    p, pcov, info = levmar.levmar_lec(func, p0, y, (A, b), cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt, decimal=5)


def test_hs51():
    # Hock-Schittkowski problem 51 (linearly constrained)
    def func(p):
        y = np.empty(5)
        t1 = p[0] - p[1]
        t2 = p[1] + p[2] - 2.0
        t3 = p[3] - 1.0
        t4 = p[4] - 1.0
        y[:] = t1*t1 + t2*t2 + t3*t3 + t4*t4
        return y

    def jacf(p):
        j = np.empty((5,5))
        t1 = p[0] - p[1]
        t2 = p[1] + p[2] - 2.0
        t3 = p[3] - 1.0
        t4 = p[4] - 1.0
        j[:,0] = 2.0*t1
        j[:,1] = 2.0*(t2-t1)
        j[:,2] = 2.0*t2
        j[:,3] = 2.0*t3
        j[:,4] = 2.0*t4
        return j

    y = np.zeros(5, np.float64)
    p0 = [2.5, 0.5, 2.0, -1.0, 0.5]
    pt = [1.0, 1.0, 1.0, 1.0, 1.0]
    A = np.asarray([[1.0, 3.0, 0.0, 0.0,  0.0],
                     [0.0, 0.0, 1.0, 1.0, -2.0],
                     [0.0, 1.0, 0.0, 0.0, -1.0]])
    b = np.asarray([4.0, 0.0, 0.0])

    p, pcov, info = levmar.levmar_lec(func, p0, y, (A, b), jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt, decimal=4)
    p, pcov, info = levmar.levmar_lec(func, p0, y, (A, b), **OPTS)
    assert_array_almost_equal(p, pt, decimal=3)
    p, pcov, info = levmar.levmar_lec(func, p0, y, (A, b), cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt, decimal=3)


def test_hs01():
    # Hock-Schittkowski problem 01 (box constrained)
    def func(p):
        y = np.empty(2)
        y[0] = 10.0 * (p[1] - p[0]*p[0])
        y[1] = 1.0 - p[0]
        return y

    def jacf(p):
        j = np.empty((2, 2))
        j[0, 0] = -20.0 * p[0]
        j[0, 1] = 10.0
        j[1, 0] = -1.0
        j[1, 1] = 0.0
        return j

    y = np.zeros(2)
    p0 = [-2.0, 1.0]
    pt = [ 1.0, 1.0]
    bc = [(None, None), (-1.5, None)]

    p, pcov, info = levmar.levmar_bc(func, p0, y, bc, jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar_bc(func, p0, y, bc, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar_bc(func, p0, y, bc, cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt)


def test_hs21():
    # Hock-Schittkowski MODIFIED problem 21 (box constrained)
    def func(p):
        y = np.empty(2)
        y[0] = p[0] / 10.0
        y[1] = p[1]
        return y

    def jacf(p):
        j = np.empty((2, 2))
        j[0, 0] = 0.1
        j[0, 1] = 0.0
        j[1, 0] = 0.0
        j[1, 1] = 1.0
        return j

    y = np.zeros(2)
    p0 = [-1.0, -1.0]
    pt = [ 2.0,  0.0]
    bc = [(2.0, 50.0), (-50.0, 50.0)]

    p, pcov, info = levmar.levmar_bc(func, p0, y, bc, jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar_bc(func, p0, y, bc, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar_bc(func, p0, y, bc, cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt)


def test_hatfldb():
    # Problem hatfldb (box constrained)
    def func(p):
        y = np.empty(4)
        y[0] = p[0] - 1
        for i in range(1, 4):
            y[i] = p[i-1] - sqrt(p[i])
        return y

    def jacf(p):
        j = np.empty((4, 4))
        j[0, 0] = 1.0
        j[0, 1] = 0.0
        j[0, 2] = 0.0
        j[0, 3] = 0.0
        j[1, 0] = 1.0
        j[1, 1] = -0.5 / sqrt(p[1])
        j[1, 2] = 0.0
        j[1, 3] = 0.0
        j[2, 0] = 0.0
        j[2, 1] = 1.0
        j[2, 2] = -0.5 / sqrt(p[2])
        j[2, 3] = 0.0
        j[3, 0] = 0.0
        j[3, 1] = 0.0
        j[3, 2] = 1.0
        j[3, 3] = -0.5 / sqrt(p[3])
        return j

    y = np.zeros(4)
    p0 = [0.1, 0.1, 0.1, 0.1]
    pt = [0.947214, 0.8, 0.64, 0.4096]
    bc = [(0.0, None), (0.0, 0.8), (0.0, None), (0.0, None)]

    p, pcov, info = levmar.levmar_bc(func, p0, y, bc, jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar_bc(func, p0, y, bc, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar_bc(func, p0, y, bc, cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt)


def test_hatfldc():
    # Problem hatfldc (box constrained)
    def func(p):
        y = np.empty(4)
        y[0] = p[0] - 1.0
        for i in range(1, 3):
            y[i] = p[i-1] - sqrt(p[i])
        y[3] = p[3] - 1.0
        return y

    def jacf(p):
        j = np.empty((4, 4))
        j[0, 0] = 1.0
        j[0, 1] = 0.0
        j[0, 2] = 0.0
        j[0, 3] = 0.0
        j[1, 0] = 1.0
        j[1, 1] = -0.5 / sqrt(p[1])
        j[1, 2] = 0.0
        j[1, 3] = 0.0
        j[2, 0] = 0.0
        j[2, 1] = 1.0
        j[2, 2] = -0.5 / sqrt(p[2])
        j[2, 3] = 0.0
        j[3, 0] = 0.0
        j[3, 1] = 0.0
        j[3, 2] = 0.0
        j[3, 3] = 1.0
        return j

    y = np.zeros(4)
    p0 = [0.9, 0.9, 0.9, 0.9]
    pt = [1.0, 1.0, 1.0, 1.0]
    bc = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]

    p, pcov, info = levmar.levmar_bc(func, p0, y, bc, jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar_bc(func, p0, y, bc, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar_bc(func, p0, y, bc, cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt)


def test_combust():
    # Equilibrium combustion problem, constrained nonlinear equation from the
    # book by Floudas et al.
    R   = 10.0
    R5  = 0.193
    R6  = 4.10622*1e-4
    R7  = 5.45177*1e-4
    R8  = 4.4975*1e-7
    R9  = 3.40735*1e-5
    R10 = 9.615*1e-7

    def func(p):
        p0, p1, p2, p3, p4 = p
        y = np.empty(5)
        y[0] = p0 * p1 + p0 - 3.0 * p4
        y[1] = 2.0 * p0 * p1 + p0 + 3.0 * R10 * p1*p1 + p1 * p2*p2 + \
            R7 * p1 * p2 + R9 * p1 * p3 + R8 * p1 - R * p4
        y[2] = 2.0 * p1 * p2*p2 + R7 * p1 * p2 + 2.0 * R5 * p2*p2 + \
            R6 * p2 - 8.0*p4
        y[3] = R9 * p1 * p3 + 2.0 * p3*p3 - 4.0 * R * p4
        y[4] = p0 * p1 + p0 + R10 * p1*p1 + p1 * p2*p2 + \
            R7 * p1 * p2 + R9 * p1 * p3 + R8 * p1 + R5 * p2*p2 + \
            R6 * p2 + p3*p3 - 1.0
        return y

    def jacf(p):
        p0, p1, p2, p3, p4 = p
        j = np.empty((5, 5))
        j[0, 0] = p1 + 1.0
        j[0, 1] = p0
        j[0, 2] = 0.0
        j[0, 3] = 0.0
        j[0, 4] = -3.0
        j[1, 0] = 2.0 * p1 + 1.0
        j[1, 1] = 2.0 * p0 + 6.0 * R10 * p1 + p2*p2 + R7 * p2 + R9 * p3 + R8
        j[1, 2] = 2.0 * p1 * p2 + R7 * p1
        j[1, 3] = R9*p1
        j[1, 4] = -R
        j[2, 0] = 0.0
        j[2, 1] = 2.0 * p2*p2 + R7 * p2
        j[2, 2] = 4.0 * p1 * p2 + R7 * p1 + 4 * R5 * p2 + R6
        j[2, 3] = -8.0
        j[2, 4] = 0.0
        j[3, 0] = 0.0
        j[3, 1] = R9 * p3
        j[3, 2] = 0.0
        j[3, 3] = R9 * p1 + 4.0*p3
        j[3, 4] = -4.0 * R
        j[4, 0] = p1 + 1.0
        j[4, 1] = p0 + 2.0 * R10 * p1 + p2*p2 + R7 * p2 + R9 * p3 + R8
        j[4, 2] = 2.0 * p1 * p2 + R7 * p1 + 2.0 * R5 * p2 + R6
        j[4, 3] = R9 * p1 + 2.0 * p3
        j[4, 4] = 0
        return j

    y = np.zeros(5)
    p0 = [0.0010,  0.0010, 0.0010, 0.0010, 0.0010]
    pt = [0.0034, 31.3265, 0.0684, 0.8595, 0.0370]
    bc = [(0.001, 100), (0.001, 100), (0.001, 100), (0.001, 100), (0.001, 100)]

    p, pcov, info = levmar.levmar_bc(func, p0, y, bc, jacf=jacf,
                                     maxit=5000, **OPTS)
    assert_array_almost_equal(p, pt, decimal=1)


def test_mod1hs52():
    # Hock-Schittkowski (modified #1) problem 52 (box/linearly constrained)
    def func(p):
        y = np.empty(4)
        y[0] = 4.0 * p[0] - p[1]
        y[1] = p[1] + p[2] - 2.0
        y[2] = p[3] - 1.0
        y[3] = p[4] - 1.0
        return y

    def jacf(p):
        j = np.empty((4, 5))
        j[0, 0] = 4.0
        j[0, 1] = -1.0
        j[0, 2] = 0.0
        j[0, 3] = 0.0
        j[0, 4] = 0.0
        j[1, 0] = 0.0
        j[1, 1] = 1.0
        j[1, 2] = 1.0
        j[1, 3] = 0.0
        j[1, 4] = 0.0
        j[2, 0] = 0.0
        j[2, 1] = 0.0
        j[2, 2] = 0.0
        j[2, 3] = 1.0
        j[2, 4] = 0.0
        j[3, 0] = 0.0
        j[3, 1] = 0.0
        j[3, 2] = 0.0
        j[3, 3] = 0.0
        j[3, 4] = 1.0
        return j

    y = np.zeros(4)
    p0 = [2.0, 2.0, 2.0, 2.0, 2.0]
    pt = [-0.09, 0.03, 0.25, -0.19, 0.03]
    A = np.asarray([[1.0, 3.0, 0.0, 0.0,  0.0],
                    [0.0, 0.0, 1.0, 1.0, -2.0],
                    [0.0, 1.0, 0.0, 0.0, -1.0]])
    b = np.asarray([0.0, 0.0, 0.0])
    bc = [(-0.09, None), (0.0, 0.3), (None, 0.25), (-0.2, 0.3), (0.0, 0.3)]

    p, pcov, info = levmar.levmar_blec(func, p0, y, bc, (A, b), jacf=jacf,
                                       **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar_blec(func, p0, y, bc, (A, b), **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar_blec(func, p0, y, bc, (A, b),
                                       cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt, decimal=3)


def test_mod2hs52():
    # Hock-Schittkowski (modified #2) problem 52 (linear inequality constrained)
    def func(p):
        y = np.empty(5)
        y[0] = 4*p[0] - p[1]
        y[1] = p[1] + p[2] - 2
        y[2] = p[3] - 1
        y[3] = p[4] - 1
        y[4] = p[0] - 0.5
        return y

    def jacf(p):
        j = np.empty((5, 5))
        j[0, 0] = 4.0
        j[0, 1] = -1.0
        j[0, 2] = 0.0
        j[0, 3] = 0.0
        j[0, 4] = 0.0
        j[1, 0] = 0.0
        j[1, 1] = 1.0
        j[1, 2] = 1.0
        j[1, 3] = 0.0
        j[1, 4] = 0.0
        j[2, 0] = 0.0
        j[2, 1] = 0.0
        j[2, 2] = 0.0
        j[2, 3] = 1.0
        j[2, 4] = 0.0
        j[3, 0] = 0.0
        j[3, 1] = 0.0
        j[3, 2] = 0.0
        j[3, 3] = 0.0
        j[3, 4] = 1.0
        j[4, 0] = 1.0
        j[4, 1] = 0.0
        j[4, 2] = 0.0
        j[4, 3] = 0.0
        j[4, 4] = 0.0
        return j

    y = np.zeros(5)
    p0 = [2.0, 2.0, 2.0, 2.0, 2.0]
    pt = [0.5, 2.0, 0.0, 1.0, 1.0]
    C = np.asarray([[1.0,  3.0, 0.0, 0.0,  0.0],
                    [0.0,  0.0, 1.0, 1.0, -2.0],
                    [0.0, -1.0, 0.0, 0.0,  1.0]])
    d = np.asarray([-1.0, -2.0, -7.0])

    p, pcov, info = levmar.levmar_lic(func, p0, y, (C, d), jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar_lic(func, p0, y, (C, d), **OPTS)
    assert_array_almost_equal(p, pt, decimal=3)
    p, pcov, info = levmar.levmar_lic(func, p0, y, (C, d), cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt)


def test_mods235():
    # Schittkowski (modified) problem 235 (box/linearly constrained)
    def func(p):
        y = np.empty(2)
        y[0] = 0.1 * (p[0] - 1.0);
        y[1] = p[1] - p[0]*p[0];
        return y

    def jacf(p):
        j = np.empty((2, 3))
        j[0, 0] = 0.1
        j[0, 1] = 0.0
        j[0, 2] = 0.0
        j[1, 0] = -2.0 * p[0]
        j[1, 1] = 1.0
        j[1, 2] = 0.0
        return j

    y = np.zeros(2)
    p0 = [-2.000, 3.000, 1.000]
    pt = [-1.725, 2.900, 0.725]
    A = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, -4.0]])
    b = np.asarray([-1.0, 0.0])
    bc = ((None, None), (0.1, 2.9), (0.7, None))

    p, pcov, info = levmar.levmar_blec(func, p0, y, bc, (A, b),
                                       jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar_blec(func, p0, y, bc, (A, b), **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar_blec(func, p0, y, bc, (A, b),
                                       cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt)


def test_modbt7():
    # Boggs and Tolle modified problem 7 (box/linearly constrained)
    def func(p):
        y = np.empty(5)
        t1 = p[1] - p[0]*p[0]
        t2 = p[0] - 1.0
        y[:] = 100.0 * t1*t1 + t2*t2
        return y

    def jacf(p):
        j = np.empty((5, 5))
        t = p[1] - p[0]*p[0]
        j[:, 0] = -400.0 * t * p[0] + 2.0 * p[0] - 2.0
        j[:, 1] = 200.0 * t
        j[:, 2] = 0.0
        j[:, 3] = 0.0
        j[:, 4] = 0.0
        return j

    y = np.zeros(5)
    p0 = [-2.00, 1.00, 1.00, 1.00,  1.00]
    pt = [ 0.70, 0.49, 0.19, 1.19, -0.20]
    A = np.asarray([[1.0, 1.0, -1.0,  0.0, 0.0],
                    [1.0, 1.0,  0.0, -1.0, 0.0],
                    [1.0, 0.0,  0.0,  0.0, 1.0]])
    b = np.asarray([1.0, 0.0, 0.5])
    bc = [(None, 0.7), (None, None), (None, None), (None, None), (-0.3, None)]

    p, pcov, info = levmar.levmar_blec(func, p0, y, bc, (A, b),
                                       jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt)
    p, pcov, info = levmar.levmar_blec(func, p0, y, bc, (A, b),
                                       maxit=10000, **OPTS)
    assert_array_almost_equal(p, pt, decimal=3)
    p, pcov, info = levmar.levmar_blec(func, p0, y, bc, (A, b),
                                       maxit=10000, cdiff=True, **OPTS)
    assert_array_almost_equal(p, pt, decimal=4)


def test_modhs76():
    # Hock-Schittkowski (modified) problem 76 (linear inequalities & equations
    # constrained).
    def func(p):
        y = np.empty(4)
        y[0] = p[0]
        y[1] = sqrt(0.5) * p[1]
        y[2] = p[2]
        y[3] = sqrt(0.5) * p[3]
        return y

    def jacf(p):
        j = np.empty((4, 4))
        j[0, 0] = 1.0
        j[0, 1] = 0.0
        j[0, 2] = 0.0
        j[0, 3] = 0.0
        j[1, 0] = 0.0
        j[1, 1] = sqrt(0.5)
        j[1, 2] = 0.0
        j[1, 3] = 0.0
        j[2, 0] = 0.0
        j[2, 1] = 0.0
        j[2, 2] = 1.0
        j[2, 3] = 0.0
        j[3, 0] = 0.0
        j[3, 1] = 0.0
        j[3, 2] = 0.0
        j[3, 3] = sqrt(0.5)
        return j

    y = np.zeros(4)
    p0 = [0.5, 0.5, 0.5, 0.5]
    pt = [0.0, 0.00909091, 0.372727, 0.354545]
    A = np.asarray([[0.0, 1.0, 4.0, 0.0]])
    b = np.asarray([1.5])
    C = np.asarray([[-1.0, -2.0, -1.0, -1.0],
                    [-3.0, -1.0, -2.0,  1.0]])
    d = np.asarray([-5.0, -0.4])
    bc = [(0.0, None), (0.0, None), (0.0, None), (0.0, None)]

    p, pcov, info = levmar.levmar_bleic(func, p0, y, bc, (A, b), (C, d),
                                        jacf=jacf, **OPTS)
    assert_array_almost_equal(p, pt)
