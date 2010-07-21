#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
from __future__ import division
from math import (atan, pi, sqrt)
import numpy as np
from numpy.testing import *
from numpy import (arange, asarray, empty, float64, zeros)

from levmar._core import (levmar, LMUserFuncError)


OPTS = dict(eps1=1e-15, eps2=1e-15, eps3=1e-20)


class TestRosen(TestCase):
    __test__ = True
    ROSD = 105.0

    def setUp(self):
        self.x = zeros(2, dtype=float64)
        self.p0 = (-1.2, 1.0)
        self.pt = (1.0, 1.0)

    def rosen(self, p):
        p0, p1 = p
        y = empty(2)
        y[0] = (1.0-p0)
        y[1] = 10.0*(p1-p0*p0)
        return y

    def jac_rosen(self, p):
        p0, p1 = p
        jac = empty((2,2))
        jac[0,0] = -1.0
        jac[0,1] = 0.0
        jac[1,0] = -20.0*p0
        jac[1,1] = 10.0
        return jac

    def test_der(self):
        kw = dict(jacf=self.jac_rosen, **OPTS)
        p, covr, info = levmar(self.rosen, self.p0, self.x, **kw)
        assert_array_almost_equal(p, self.pt)

    def test_diff(self):
        kw = dict(OPTS)
        p, covr, info = levmar(self.rosen, self.p0, self.x, **kw)
        assert_array_almost_equal(p, self.pt)

    def test_cdiff(self):
        kw = dict(cdif=True, **OPTS)
        p, covr, info = levmar(self.rosen, self.p0, self.x, **kw)
        assert_array_almost_equal(p, self.pt)


class TestPowell(TestCase):
    __test__ = False

    def setUp(self):
        self.x = zeros(2, dtype=float64)
        self.p0 = (2.0, 1.0)
        self.pt = (0.0, 0.0)

    def powell(self, p):
        p0, p1 = p
        y = empty(2)
        y[0] = p0
        y[1] = 10.0*p0/(p0+0.1) + 2.0*p1*p1
        return y

    def jac_powell(self, p):
        p0, p1 = p
        jac = empty((2,2))
        jac[0,0] = 1.0
        jac[0,1] = 0.0
        jac[1,0] = 1.0/((p0+0.1)*(p0+0.1))
        jac[1,1] = 4.0*p1
        return jac

    # @decorators.knownfailureif(True)
    def test_der(self):
        kw = dict(jacf=self.jac_powell, **OPTS)
        p, covr, info = levmar(self.powell, self.p0, self.x, **kw)
        assert_array_almost_equal(p, self.pt)

    # @decorators.knownfailureif(True)
    def test_diff(self):
        kw = dict(OPTS)
        p, covr, info = levmar(self.powell, self.p0, self.x, **kw)
        assert_array_almost_equal(p, self.pt)

    # @decorators.knownfailureif(True)
    def test_cdiff(self):
        kw = dict(cdif=True, **OPTS)
        p, covr, info = levmar(self.powell, self.p0, self.x, **kw)
        assert_array_almost_equal(p, self.pt)


class TestModRos(TestCase):
    __test__ = True
    MODROSLAM = 1E+02

    def setUp(self):
        self.x = zeros(3, dtype=float64)
        self.p0 = (-1.2, 2.0)
        self.pt = (1.0, 1.0)

    def modros(self, p):
        p0, p1 = p
        y = empty(3)
        y[0] = 10.0*(p1-p0*p0)
        y[1] = 1.0 - p0
        y[2] = self.MODROSLAM
        return y

    def jac_modros(self, p):
        p0, p1 = p
        jac = empty((3,2))
        jac[0,0] = -20.0*p0
        jac[0,1] = 10.0
        jac[1,0] = -1.0
        jac[1,1] = 0.0
        jac[2,0] = 0.0
        jac[2,1] = 0.0
        return jac

    def test_der(self):
        kw = dict(jacf=self.jac_modros, **OPTS)
        p, covr, info = levmar(self.modros, self.p0, self.x, **kw)
        assert_array_almost_equal(p, self.pt, decimal=5)

    def test_diff(self):
        kw = dict(OPTS)
        p, covr, info = levmar(self.modros, self.p0, self.x, **kw)
        assert_array_almost_equal(p, self.pt)

    def test_cdiff(self):
        kw = dict(cdif=True, **OPTS)
        p, covr, info = levmar(self.modros, self.p0, self.x, **kw)
        assert_array_almost_equal(p, self.pt)


class TestWood(TestCase):
    __test__ = True

    def setUp(self):
        self.x = zeros(6, dtype=float64)
        self.p0 = (-3.0, -1.0, -3.0, -1.0)
        self.pt = (1.0, 1.0, 1.0, 1.0)

    def wood(self, p):
        p0, p1, p2, p3 = p
        y = empty(6)
        y[0] = 10.0*(p1-p0*p0)
        y[1] = 1.0 - p0
        y[2] = sqrt(90.0)*(p3-p2*p2)
        y[3] = 1.0 - p2
        y[4] = sqrt(10)*(p1+p3-2.0)
        y[5] = (p1-p3)/sqrt(10.0)
        return y

    def test_diff(self):
        kw = dict(OPTS)
        p, covr, info = levmar(self.wood, self.p0, self.x, **kw)
        assert_array_almost_equal(p, self.pt)

    def test_cdiff(self):
        kw = dict(cdif=True, **OPTS)
        p, covr, info = levmar(self.wood, self.p0, self.x, **kw)
        assert_array_almost_equal(p, self.pt)


class TestMeyer(TestCase):
    __test__ = True

    def setUp(self):
        self.x = 0.45 + 0.05*arange(16)
        self.y = asarray([34.780, 28.610, 23.650, 19.630,
                          16.370, 13.720, 11.540,  9.744,
                           8.261,  7.030,  6.005,  5.147,
                           4.427,  3.820,  3.307,  2.872])
        self.p0 = (8.85, 4.0,  2.5)
        self.pt = (2.48, 6.18, 3.45)

    def meyer(self, p, x):
        y = p[0]*np.exp(10.0*p[1]/(x+p[2]) - 13.0)
        return y

    def jac_meyer(self, p, x):
        p0, p1, p2 = p
        jac = empty((16,3))
        tmp = np.exp(10.0*p1/(x+p2) - 13.0)
        jac[:,0] = tmp
        jac[:,1] = 10.0*p0*tmp/(x+p2)
        jac[:,2] = -10.0*p0*p1*tmp/((x+p2)*(x+p2))
        return jac

    def test_der(self):
        kw = dict(args=(self.x,), jacf=self.jac_meyer, **OPTS)
        ret = levmar(self.meyer, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=1)

    def test_diff(self):
        kw = dict(args=(self.x,), **OPTS)
        ret = levmar(self.meyer, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=1)

    def test_cdiff(self):
        kw = dict(args=(self.x,), cdif=True, **OPTS)
        ret = levmar(self.meyer, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=1)


class TestOsborne(TestCase):
    __test__ = True

    def setUp(self):
        self.x = 10.0*arange(33)
        self.y = np.asarray([8.44E-1, 9.08E-1, 9.32E-1, 9.36E-1, 9.25E-1,
                             9.08E-1, 8.81E-1, 8.50E-1, 8.18E-1, 7.84E-1,
                             7.51E-1, 7.18E-1, 6.85E-1, 6.58E-1, 6.28E-1,
                             6.03E-1, 5.80E-1, 5.58E-1, 5.38E-1, 5.22E-1,
                             5.06E-1, 4.90E-1, 4.78E-1, 4.67E-1, 4.57E-1,
                             4.48E-1, 4.38E-1, 4.31E-1, 4.24E-1, 4.20E-1,
                             4.14E-1, 4.11E-1, 4.06E-1])
        self.p0 = (0.5, 1.5, -1.0, 1.0e-2, 2.0e-2)
        self.pt = (0.3754, 1.9358, -1.4647, 0.0129, 0.0221)

    def osborne(self, p, x):
        y = p[0] + p[1]*np.exp(-p[3]*x) + p[2]*np.exp(-p[4]*x);
        return y

    def jac_osborne(self, p, x):
        jac = empty((33,5))
        tmp1 = np.exp(-p[3]*x)
        tmp2 = np.exp(-p[4]*x)
        jac[:,0] = 1.0
        jac[:,1] = tmp1
        jac[:,2] = tmp2
        jac[:,3] = -p[1]*x*tmp1
        jac[:,4] = -p[2]*x*tmp2
        return jac

    def test_der(self):
        kw = dict(args=(self.x,), jacf=self.jac_osborne, **OPTS)
        ret = levmar(self.osborne, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=4)

    def test_diff(self):
        kw = dict(args=(self.x,), **OPTS)
        ret = levmar(self.osborne, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=4)

    def test_cdiff(self):
        kw = dict(args=(self.x,), cdif=True, **OPTS)
        ret = levmar(self.osborne, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=4)


class TestHelVal(TestCase):
    __test__ = True

    def setUp(self):
        self.y = zeros(3, float64)
        self.p0 = (-1.0, 2.0, 2.0)
        self.pt = (1.0, 0.0, 0.0)

    def helval(self, p):
        p0, p1, p2 = p
        y = empty(3)
        if p0 < 0:
            theta = atan(p1/p0)/(2*pi) + 0.5
        elif p0 > 0:
            theta = atan(p1/p0)/(2*pi)
        else:
            theta = 0.25 if p1 >= 0 else -0.25
        y[0] = 10.0*(p2 - 10.0*theta)
        y[1] = 10.0*(sqrt(p0*p0 + p1*p1) - 1.0)
        y[2] = p2
        return y

    def jac_helval(self, p):
        p0, p1, p2 = p
        jac = empty((3,3))
        tmp = p0*p0 + p1*p1
        jac[0,0] = 50.0*p1/(pi*tmp)
        jac[0,1] = -50.0*p0/(pi*tmp)
        jac[0,2] = 10.0
        jac[1,0] = 10.0*p0/sqrt(tmp)
        jac[1,1] = 10.0*p1/sqrt(tmp)
        jac[1,2] = 0.0
        jac[2,0] = 0.0
        jac[2,1] = 0.0
        jac[2,2] = 1.0
        return jac

    def test_der(self):
        kw = dict(jacf=self.jac_helval, **OPTS)
        ret = levmar(self.helval, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)

    def test_diff(self):
        kw = dict(OPTS)
        ret = levmar(self.helval, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)

    def test_cdiff(self):
        kw = dict(cdif=True, **OPTS)
        ret = levmar(self.helval, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)


class TestBt3(TestCase):
    __test__ = True

    def setUp(self):
        self.y = zeros(5, float64)
        self.p0 = (2.0, 2.0, 2.0, 2.0, 2.0)
        self.pt = (-0.76744, 0.25581, 0.62791, -0.11628, 0.25581)
        self.A = asarray([[1.0, 3.0, 0.0, 0.0,  0.0],
                          [0.0, 0.0, 1.0, 1.0, -2.0],
                          [0.0, 1.0, 0.0, 0.0, -1.0]])
        self.b = asarray([0.0, 0.0, 0.0])

    def bt3(self, p):
        y = np.empty(5)
        t1 = p[0] - p[1]
        t2 = p[1] + p[2] - 2.0
        t3 = p[3] - 1.0
        t4 = p[4] - 1.0
        y[:] = t1*t1 + t2*t2 + t3*t3 + t4*t4
        return y

    def jac_bt3(self, p):
        jac = np.empty((5,5))
        t1 = p[0] - p[1]
        t2 = p[1] + p[2] - 2.0
        t3 = p[3] - 1.0
        t4 = p[4] - 1.0
        jac[:,0] = 2.0*t1
        jac[:,1] = 2.0*(t2-t1)
        jac[:,2] = 2.0*t2
        jac[:,3] = 2.0*t3
        jac[:,4] = 2.0*t4
        return jac

    def test_der(self):
        kw =  dict(jacf=self.jac_bt3, A=self.A, b=self.b, **OPTS)
        ret = levmar(self.bt3, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=4)

    def test_diff(self):
        kw = dict(A=self.A, b=self.b, **OPTS)
        ret = levmar(self.bt3, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=4)

    def test_cdiff(self):
        kw = dict(A=self.A, b=self.b, cdif=True, **OPTS)
        ret = levmar(self.bt3, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=4)


class TestHS28(TestCase):
    __test__ = True

    def setUp(self):
        self.y = zeros(3, float64)
        self.p0 = (-4.0, 1.0, 1.0)
        self.pt = (0.5, -0.5, 0.5)
        self.A = asarray([1.0, 2.0, 3.0])
        self.b = asarray([1.0])

    def hs28(self, p):
        y = empty(3)
        t1 = p[0] + p[1]
        t2 = p[1] + p[2]
        y[:] = t1*t1 + t2*t2
        return y

    def jac_hs28(self, p):
        jac = empty((3,3))
        t1 = p[0] + p[1]
        t2 = p[1] + p[2]
        jac[:,0] = 2.0*t1
        jac[:,1] = 2.0*(t1+t2)
        jac[:,2] = 2.0*t2
        return jac

    def test_der(self):
        kw =dict(jacf=self.jac_hs28, A=self.A, b=self.b, **OPTS)
        ret = levmar(self.hs28, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=5)

    def test_diff(self):
        kw = dict(A=self.A, b=self.b, **OPTS)
        ret = levmar(self.hs28, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=3)

    def test_cdiff(self):
        kw = dict(A=self.A, b=self.b, cdif=True, **OPTS)
        ret = levmar(self.hs28, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=4)


class TestHS48(TestCase):
    __test__ = True

    def setUp(self):
        self.y = zeros(5, float64)
        self.p0 = (3.0, 5.0, -3.0, 2.0, -2.0)
        self.pt = (1.0, 1.0, 1.0, 1.0, 1.0)
        self.A = asarray([[1.0, 1.0, 1.0,  1.0,  1.0],
                          [0.0, 0.0, 1.0, -2.0, -2.0]])
        self.b = asarray([5.0, -3.0])

    def hs48(self, p):
        y = empty(5)
        t1 = p[0] - 1.0
        t2 = p[1] - p[2]
        t3 = p[3] - p[4]
        y[:] = t1*t1 + t2*t2 + t3*t3
        return y

    def jac_hs48(self, p):
        jac = empty((5,5))
        t1 = p[0] - 1.0
        t2 = p[1] - p[2]
        t3 = p[3] - p[4]
        jac[:,0] = 2.0*t1
        jac[:,1] = 2.0*t2
        jac[:,2] = -2.0*t2
        jac[:,3] = 2.0*t3
        jac[:,4] = -2.0*t3
        return jac

    def test_der(self):
        kw = dict(jacf=self.jac_hs48, A=self.A, b=self.b, **OPTS)
        ret = levmar(self.hs48, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=5)

    def test_diff(self):
        kw = dict(A=self.A, b=self.b, **OPTS)
        ret = levmar(self.hs48, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=5)

    def test_cdiff(self):
        kw = dict(A=self.A, b=self.b, cdif=True, **OPTS)
        ret = levmar(self.hs48, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=5)


class TestHS51(TestCase):
    __test__ = True

    def setUp(self):
        self.y = zeros(5, float64)
        self.p0 = (2.5, 0.5, 2.0, -1.0, 0.5)
        self.pt = (1.0, 1.0, 1.0, 1.0, 1.0)
        self.A = asarray([[1.0, 3.0, 0.0, 0.0,  0.0],
                          [0.0, 0.0, 1.0, 1.0, -2.0],
                          [0.0, 1.0, 0.0, 0.0, -1.0]])
        self.b = asarray([4.0, 0.0, 0.0])

    def hs51(self, p):
        y = empty(5)
        t1 = p[0] - p[1]
        t2 = p[1] + p[2] - 2.0
        t3 = p[3] - 1.0
        t4 = p[4] - 1.0
        y[:] = t1*t1 + t2*t2 + t3*t3 + t4*t4
        return y

    def jac_hs51(self, p):
        jac = empty((5,5))
        t1 = p[0] - p[1]
        t2 = p[1] + p[2] - 2.0
        t3 = p[3] - 1.0
        t4 = p[4] - 1.0
        jac[:,0] = 2.0*t1
        jac[:,1] = 2.0*(t2-t1)
        jac[:,2] = 2.0*t2
        jac[:,3] = 2.0*t3
        jac[:,4] = 2.0*t4
        return jac

    def test_der(self):
        kw = dict(jacf=self.jac_hs51, A=self.A, b=self.b, **OPTS)
        ret = levmar(self.hs51, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=4)

    def test_diff(self):
        kw = dict(A=self.A, b=self.b, **OPTS)
        ret = levmar(self.hs51, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=3)

    def test_cdiff(self):
        kw = dict(A=self.A, b=self.b, cdif=True, **OPTS)
        ret = levmar(self.hs51, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=3)


class TestHS01(TestCase):
    __test__ = True

    def setUp(self):
        self.y = zeros(2, float64)
        self.p0 = (-2.0, 1.0)
        self.pt = (1.0, 1.0)
        self.bounds = (None, (-1.5, None))

    def hs01(self, p):
        p0, p1 = p
        y = empty(2)
        y[0] = 10.0*(p1-p0*p0)
        y[1] = 1.0 - p0
        return y

    def jac_hs01(self, p):
        jac = empty((2,2))
        jac[0,0] = -20.0*p[0]
        jac[0,1] = 10.0
        jac[1,0] = -1.0
        jac[1,1] = 0.0
        return jac

    def test_der(self):
        kw = dict(jacf=self.jac_hs01, bounds=self.bounds, **OPTS)
        ret = levmar(self.hs01, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)

    def test_diff(self):
        kw = dict(bounds=self.bounds, **OPTS)
        ret = levmar(self.hs01, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)

    def test_cdiff(self):
        kw = dict(bounds=self.bounds, cdif=True, **OPTS)
        ret = levmar(self.hs01, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)


class TestHS21(TestCase):
    __test__ = True

    def setUp(self):
        self.y = zeros(2, float64)
        self.p0 = (4.0, -1.0)
        self.pt = (2.0, 0.0)
        self.bounds = ((2.0, 50.0), (-50.0, 50.0))

    def hs21(self, p):
        y = empty(2)
        y[0] = p[0]/10.0
        y[1] = p[1]
        return y

    def jac_hs21(self, p):
        jac = empty((2,2))
        jac[0,0] = 0.1
        jac[0,1] = 0.0
        jac[1,0] = 0.0
        jac[1,1] = 1.0
        return jac

    def test_der(self):
        kw = dict(jacf=self.jac_hs21, bounds=self.bounds, **OPTS)
        ret = levmar(self.hs21, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)

    def test_diff(self):
        kw = dict(bounds=self.bounds, **OPTS)
        ret = levmar(self.hs21, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)

    def test_cdiff(self):
        kw = dict(bounds=self.bounds, cdif=True, **OPTS)
        ret = levmar(self.hs21, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)


class TestHatfldB(TestCase):
    __test__ = True

    def setUp(self):
        self.y = zeros(4, float64)
        self.p0 = (0.1, 0.1, 0.1, 0.1)
        self.pt = (0.947214, 0.8, 0.64, 0.4096)
        self.bounds = ((0, None), (0, 0.8), (0, None), (0, None))

    def hatfldb(self, p):
        y = empty(4)
        y[0] = p[0] - 1
        for i in range(1,4):
            y[i] = p[i-1] - sqrt(p[i])
        return y

    def jac_hatfldb(self, p):
        jac = empty((4,4))
        jac[0,0] = 1.0
        jac[0,1] = 0.0
        jac[0,2] = 0.0
        jac[0,3] = 0.0
        jac[1,0] = 1.0
        jac[1,1] = -0.5/sqrt(p[1])
        jac[1,2] = 0.0
        jac[1,3] = 0.0
        jac[2,0] = 0.0
        jac[2,1] = 1.0
        jac[2,2] = -0.5/sqrt(p[2])
        jac[2,3] = 0.0
        jac[3,0] = 0.0
        jac[3,1] = 0.0
        jac[3,2] = 1.0
        jac[3,3] = -0.5/sqrt(p[3])
        return jac

    def test_der(self):
        kw = dict(jacf=self.jac_hatfldb, bounds=self.bounds, **OPTS)
        ret = levmar(self.hatfldb, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)

    def test_diff(self):
        kw = dict(bounds=self.bounds, **OPTS)
        ret = levmar(self.hatfldb, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)

    def test_cdiff(self):
        kw = dict(bounds=self.bounds, cdif=True, **OPTS)
        ret = levmar(self.hatfldb, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)


class TestHatfldC(TestCase):
    __test__ = True

    def setUp(self):
        self.y = zeros(4, float64)
        self.p0 = (0.9, 0.9, 0.9, 0.9)
        self.pt = (1.0, 1.0, 1.0, 1.0)
        self.bounds = ((0, 10), (0, 10), (0, 10), (0, 10))

    def hatfldc(self, p):
        y = empty(4)
        y[0] = p[0] - 1.0
        for i in range(1,3):
            y[i] = p[i-1] - sqrt(p[i])
        y[3] = p[3] - 1.0
        return y

    def jac_hatfldc(self, p):
        jac = empty((4,4))
        jac[0,0] = 1.0
        jac[0,1] = 0.0
        jac[0,2] = 0.0
        jac[0,3] = 0.0
        jac[1,0] = 1.0
        jac[1,1] = -0.5/sqrt(p[1])
        jac[1,2] = 0.0
        jac[1,3] = 0.0
        jac[2,0] = 0.0
        jac[2,1] = 1.0
        jac[2,2] = -0.5/sqrt(p[2])
        jac[2,3] = 0.0
        jac[3,0] = 0.0
        jac[3,1] = 0.0
        jac[3,2] = 0.0
        jac[3,3] = 1.0
        return jac

    def test_der(self):
        kw = dict(jacf=self.jac_hatfldc, bounds=self.bounds, **OPTS)
        ret = levmar(self.hatfldc, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)

    def test_diff(self):
        kw = dict(bounds=self.bounds, **OPTS)
        ret = levmar(self.hatfldc, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)

    def test_cdiff(self):
        kw = dict(bounds=self.bounds, cdif=True, **OPTS)
        ret = levmar(self.hatfldc, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)


class TestCombust(TestCase):
    __test__ = True

    R = 10
    R5 = 0.193
    R6 = 4.10622*1e-4
    R7 = 5.45177*1e-4
    R8 = 4.4975*1e-7
    R9 = 3.40735*1e-5
    R10 = 9.615*1e-7

    def setUp(self):
        self.y = zeros(5, float64)
        self.p0 = (1e-3, 1e-3, 1e-3, 1e-3, 1e-3)
        self.pt = (0.0034, 31.3265, 0.0684, 0.8595, 0.0370)
        self.bounds = ((1e-3, 100), (1e-3, 100), (1e-3, 100), (1e-3, 100), (1e-3, 100))

    def combust(self, p):
        p0, p1, p2, p3, p4 = p
        y = empty(5)
        y[0] = p0*p1 + p0 - 3.0*p4
        y[1] = 2.0*p0*p1 + p0 + 3.0*self.R10*p1*p1 + p1*p2*p2 + \
                self.R7*p1*p2 + self.R9*p1*p3 + self.R8*p1 - self.R*p4
        y[2] = 2.0*p1*p2*p2 + self.R7*p1*p2 + 2.0*self.R5*p2*p2 + self.R6*p2 \
                - 8.0*p4
        y[3] = self.R9*p1*p3 + 2.0*p3*p3 - 4.0*self.R*p4
        y[4] = p0*p1 + p0 + self.R10*p1*p1 + p1*p2*p2 + \
                self.R7*p1*p2 + self.R9*p1*p3 + self.R8*p1 + self.R5*p2*p2 + \
                self.R6*p2 + p3*p3 - 1.0
        return y

    def jac_combust(self, p):
        p0, p1, p2, p3, p4 = p
        jac = empty((5,5))
        jac[0,0] = p1 + 1.0
        jac[0,1] = p0
        jac[0,2] = 0.0
        jac[0,3] = 0.0
        jac[0,4] = -3.0
        jac[1,0] = 2.0*p1 + 1.0
        jac[1,1] = 2.0*p0 + 6.0*self.R10*p1 + p2*p2 + self.R7*p2 + self.R9*p3 + self.R8
        jac[1,2] = 2.0*p1*p2 + self.R7*p1
        jac[1,3] = self.R9*p1
        jac[1,4] = -self.R
        jac[2,0] = 0.0
        jac[2,1] = 2.0*p2*p2 + self.R7*p2
        jac[2,2] = 4.0*p1*p2 + self.R7*p1 + 4*self.R5*p2 + self.R6
        jac[2,3] = -8.0
        jac[2,4] = 0.0
        jac[3,0] = 0.0
        jac[3,1] = self.R9*p3
        jac[3,2] = 0.0
        jac[3,3] = self.R9*p1 + 4.0*p3
        jac[3,4] = -4.0*self.R
        jac[4,0] = p1 + 1.0
        jac[4,1] = p0 + 2.0*self.R10*p1 + p2*p2 + self.R7*p2 + self.R9*p3 + self.R8
        jac[4,2] = 2.0*p1*p2 + self.R7*p1 + 2.0*self.R5*p2 + self.R6
        jac[4,3] = self.R9*p1 + 2.0*p3
        jac[4,4] = 0
        return jac

    def test_der(self):
        kw = dict(jacf=self.jac_combust, bounds=self.bounds, **OPTS)
        ret = levmar(self.combust, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=1)

    def test_diff(self):
        kw = dict(bounds=self.bounds, **OPTS)
        ret = levmar(self.combust, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=1)

    def test_cdiff(self):
        kw = dict(bounds=self.bounds, cdif=True, **OPTS)
        ret = levmar(self.combust, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=1)


class TestMods235(TestCase):
    __test__ = True

    def setUp(self):
        self.y = zeros(2, float64)
        self.p0 = (-2.0, 3.0, 1.0)
        self.pt = (-1.725, 2.9, 0.725)
        self.A = asarray([[1.0, 0.0, 1.0], [0.0, 1.0, -4.0]])
        self.b = asarray([-1.0, 0.0])
        self.bounds = ((None, None), (0.1, 2.9), (0.7, None))

    def mods235(self, p):
        y = empty(2)
        y[0] = 0.1*(p[0] - 1.0);
        y[1] = p[1] - p[0]*p[0];
        return y

    def jac_mods235(self, p):
        jac = empty((2,3))
        jac[0,0] = 0.1
        jac[0,1] = 0.0
        jac[0,2] = 0.0
        jac[1,0] = -2.0*p[0]
        jac[1,1] = 1.0
        jac[1,2] = 0.0
        return jac

    def test_der(self):
        kw = dict(jacf=self.jac_mods235, bounds=self.bounds, A=self.A, b=self.b, **OPTS)
        ret = levmar(self.mods235, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)

    def test_diff(self):
        kw = dict(bounds=self.bounds, A=self.A, b=self.b, **OPTS)
        ret = levmar(self.mods235, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)

    def test_cdiff(self):
        kw = dict(bounds=self.bounds, A=self.A, b=self.b, cdif=True, **OPTS)
        ret = levmar(self.mods235, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)


class TestMod1HS52(TestCase):
    __test__ = True

    def setUp(self):
        self.y = zeros(4, float64)
        self.p0 = (2.0, 2.0, 2.0, 2.0, 2.0)
        self.pt = (-0.09, 0.03, 0.25, -0.19, 0.03)
        self.A = asarray([[1.0, 3.0, 0.0, 0.0,  0.0],
                          [0.0, 0.0, 1.0, 1.0, -2.0],
                          [0.0, 1.0, 0.0, 0.0, -1.0]])
        self.b = asarray([0.0, 0.0, 0.0])
        self.bounds = ((-0.09, None), (0.0, 0.3), (None, 0.25), (-0.2, 0.3), (0.0, 0.3))

    def mod1hs52(self, p):
        y = empty(4)
        y[0] = 4.0*p[0] - p[1]
        y[1] = p[1] + p[2] - 2.0
        y[2] = p[3] - 1.0
        y[3] = p[4] - 1.0
        return y

    def jac_mod1hs52(self, p):
        jac = empty((4,5))
        jac[0,0] = 4.0
        jac[0,1] = -1.0
        jac[0,2] = 0.0
        jac[0,3] = 0.0
        jac[0,4] = 0.0
        jac[1,0] = 0.0
        jac[1,1] = 1.0
        jac[1,2] = 1.0
        jac[1,3] = 0.0
        jac[1,4] = 0.0
        jac[2,0] = 0.0
        jac[2,1] = 0.0
        jac[2,2] = 0.0
        jac[2,3] = 1.0
        jac[2,4] = 0.0
        jac[3,0] = 0.0
        jac[3,1] = 0.0
        jac[3,2] = 0.0
        jac[3,3] = 0.0
        jac[3,4] = 1.0
        return jac

    def test_der(self):
        kw = dict(jacf=self.jac_mod1hs52, bounds=self.bounds, A=self.A, b=self.b, **OPTS)
        ret = levmar(self.mod1hs52, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)

    def test_diff(self):
        kw = dict(bounds=self.bounds, A=self.A, b=self.b, **OPTS)
        ret = levmar(self.mod1hs52, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt, decimal=3)

    def test_cdiff(self):
        kw = dict(bounds=self.bounds, A=self.A, b=self.b, cdif=True, **OPTS)
        ret = levmar(self.mod1hs52, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)


class TestMod2HS52(TestCase):
    __test__ = True

    def setUp(self):
        self.y = zeros(5, float64)
        self.p0 = (2, 2, 2, 2, 2)
        self.pt = (0.5, 2.0, 0.0, 1.0, 1.0)
        self.C = asarray([[1.0,  3.0, 0.0, 0.0,  0.0],
                          [0.0,  0.0, 1.0, 1.0, -2.0],
                          [0.0, -1.0, 0.0, 0.0, 1.0]])
        self.d = asarray([-1.0, -2.0, -7.0])

    def mod2hs52(self, p):
        y = empty(5)
        y[0] = 4*p[0] - p[1]
        y[1] = p[1] + p[2] - 2
        y[2] = p[3] - 1
        y[3] = p[4] - 1
        y[4] = p[0] - 0.5
        return y

    def jac_mod2hs52(self, p):
        jac = empty((5,5))
        jac[0,0] = 4.0
        jac[0,1] = -1.0
        jac[0,2] = 0.0
        jac[0,3] = 0.0
        jac[0,4] = 0.0
        jac[1,0] = 0.0
        jac[1,1] = 1.0
        jac[1,2] = 1.0
        jac[1,3] = 0.0
        jac[1,4] = 0.0
        jac[2,0] = 0.0
        jac[2,1] = 0.0
        jac[2,2] = 0.0
        jac[2,3] = 1.0
        jac[2,4] = 0.0
        jac[3,0] = 0.0
        jac[3,1] = 0.0
        jac[3,2] = 0.0
        jac[3,3] = 0.0
        jac[3,4] = 1.0
        jac[4,0] = 1.0
        jac[4,1] = 0.0
        jac[4,2] = 0.0
        jac[4,3] = 0.0
        jac[4,4] = 0.0
        return jac

    def test_der(self):
        kw = dict(jacf=self.jac_mod2hs52, C=self.C, d=self.d, **OPTS)
        ret = levmar(self.mod2hs52, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)

    def test_diff(self):
        kw = dict(C=self.C, d=self.d, **OPTS)
        ret = levmar(self.mod2hs52, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)

    def test_cdiff(self):
        kw = dict(C=self.C, d=self.d, cdif=True, **OPTS)
        ret = levmar(self.mod2hs52, self.p0, self.y, **kw)
        assert_array_almost_equal(ret[0], self.pt)


if __name__ == '__main__':
    run_module_suite()
