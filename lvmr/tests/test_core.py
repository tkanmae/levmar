#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
from __future__ import division
from math import (atan, pi, sqrt)
import numpy as np
from numpy.testing import *

from lvmr import (_lvmr, LMUserFuncError)
run_levmar = _lvmr._run_levmar


_OPTS = {
    'eps1': 1E-15,
    'eps2': 1E-15,
    'eps3': 1E-20,
}


class TestRosen(TestCase):
    __test__ = True

    def setUp(self):
        self.x = np.zeros(2, dtype=np.float64)
        self.p0 = [-1.2, 1]
        self.pt = [1, 1]

    def rosen(self, p):
        x = np.empty(2, dtype=np.float64)
        x[0] = (1-p[0])
        x[1] = 10*(p[1]-p[0]*p[0])
        return x

    def jac_rosen(self, p):
        jac = np.empty((4,1))
        jac[0] = -1
        jac[1] = 0
        jac[2] = -20*p[0]
        jac[3] = 10
        return jac

    def test_der(self):
        kw = {'jacf': self.jac_rosen}
        kw.update(_OPTS)
        ret = run_levmar(self.rosen, self.p0, self.x, **kw)
        assert_array_almost_equal(ret.p, self.pt)

    def test_diff(self):
        ret = run_levmar(self.rosen, self.p0, self.x, **_OPTS)
        assert_array_almost_equal(ret.p, self.pt)

    def test_cdiff(self):
        kw = {'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.rosen, self.p0, self.x, **kw)
        assert_array_almost_equal(ret.p, self.pt)


class TestPowell(TestCase):
    __test__ = False

    def setUp(self):
        self.x = np.zeros(2, dtype=np.float64)
        self.p0 = [3, 1]
        self.pt = [0, 0]

    def powell(self, p):
        x = np.empty(2, dtype=np.float64)
        x[0] = p[0]
        x[1] = 10.0*p[0]/(p[0]+0.1) + 2*p[1]*p[1]
        return x

    def jac_powell(self, p):
        jac = np.empty(4, dtype=np.float64)

        jac[0] = 1.0
        jac[1] = 0.0

        jac[2] = 1.0/((p[0]+0.1)*(p[0]+0.1))
        jac[3] = 4.0*p[1]

        jac.shape = (2,2)
        return jac

    @decorators.knownfailureif(True)
    def test_der(self):
        kw = {'jacf': self.jac_powell}
        kw.update(_OPTS)
        ret = run_levmar(self.powell, self.p0, self.x, **kw)
        assert_array_almost_equal(ret.p, self.pt)

    @decorators.knownfailureif(True)
    def test_diff(self):
        ret = run_levmar(self.powell, self.p0, self.x, **_OPTS)
        assert_array_almost_equal(ret.p, self.pt)

    @decorators.knownfailureif(True)
    def test_cdiff(self):
        kw = {'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.powell, self.p0, self.x, **kw)
        assert_array_almost_equal(ret.p, self.pt)


class TestModRos(TestCase):
    __test__ = True
    _MODROSLAM = 1E+02

    def setUp(self):
        self.x = np.zeros(3, dtype=np.float64)
        self.p0 = [-1.2, 2]
        self.pt = [1, 1]

    def modros(self, p):
        x = np.empty(3)
        x[0] = 10*(p[1]-p[0]*p[0])
        x[1] = 1.0-p[0]
        x[2] = self._MODROSLAM
        return x

    def jac_modros(self, p):
        jac = np.empty((3,2), dtype=np.float64)

        jac[0,0] = -20.0*p[0]
        jac[0,1] = 10.0

        jac[1,0] = -1.0
        jac[1,1] = 0.0

        jac[2,0] = 0.0
        jac[2,1] = 0.0

        return jac

    def test_der(self):
        kw = {'jacf': self.jac_modros}
        kw.update(_OPTS)
        ret = run_levmar(self.modros, self.p0, self.x, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=5)

    def test_diff(self):
        ret = run_levmar(self.modros, self.p0, self.x, **_OPTS)
        assert_array_almost_equal(ret.p, self.pt)

    def test_cdiff(self):
        kw = {'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.modros, self.p0, self.x, **kw)
        assert_array_almost_equal(ret.p, self.pt)


class TestWood(TestCase):
    __test__ = True

    def setUp(self):
        self.x = np.zeros(6, dtype=np.float64)
        self.p0 = [-3, -1, -3, -1]
        self.pt = [1, 1, 1, 1]

    def wood(self, p):
        x = np.empty(6)
        x[0] = 10.0*(p[1] - p[0]*p[0])
        x[1] = 1.0 - p[0]
        x[2] = sqrt(90.0)*(p[3] - p[2]*p[2])
        x[3] = 1.0 - p[2]
        x[4] = sqrt(10.0)*(p[1]+p[3] - 2.0)
        x[5] = (p[1] - p[3])/sqrt(10.0)
        return x

    def test_diff(self):
        ret = run_levmar(self.wood, self.p0, self.x, **_OPTS)
        assert_array_almost_equal(ret.p, self.pt)

    def test_cdiff(self):
        kw = {'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.wood, self.p0, self.x, **kw)
        assert_array_almost_equal(ret.p, self.pt)


class TestMeyer(TestCase):
    __test__ = True

    def setUp(self):
        self.x = 0.45 + 0.05*(np.arange(16))
        self.y = np.asarray([34.780, 28.610, 23.650, 19.630, 16.370,
                             13.720, 11.540, 9.744, 8.261, 7.030,
                             6.005, 5.147, 4.427, 3.820, 3.307, 2.872])
        self.p0 = [8.85, 4.0, 2.5]
        self.pt = [2.48, 6.18, 3.45]

    def meyer(self, p, x):
        y = p[0]*np.exp(10.0*p[1]/(x+p[2]) - 13.0)
        return y

    def jac_meyer(self, p, x):
        jac = np.empty((16,3))
        tmp = np.exp(10.0*p[1]/(x+p[2]) - 13.0);
        jac[:,0] = tmp
        jac[:,1] = 10*p[0]*tmp/(x+p[2])
        jac[:,2] = -10*p[0]*p[1]*tmp/((x+p[2])*(x+p[2]))
        return jac

    def test_der(self):
        kw = {'jacf': self.jac_meyer}
        kw.update(_OPTS)
        ret = run_levmar(self.meyer, self.p0, self.y, args=(self.x,), **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=1)

    def test_diff(self):
        ret = run_levmar(self.meyer, self.p0, self.y, args=(self.x,), **_OPTS)
        assert_array_almost_equal(ret.p, self.pt, decimal=1)

    def test_cdiff(self):
        kw = {'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.meyer, self.p0, self.y, args=(self.x,), **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=1)


class TestOsborne(TestCase):
    __test__ = True

    def setUp(self):
        self.x = 10*np.arange(33)
        self.y = np.asarray([8.44E-1, 9.08E-1, 9.32E-1, 9.36E-1, 9.25E-1,
                             9.08E-1, 8.81E-1, 8.50E-1, 8.18E-1, 7.84E-1,
                             7.51E-1, 7.18E-1, 6.85E-1, 6.58E-1, 6.28E-1,
                             6.03E-1, 5.80E-1, 5.58E-1, 5.38E-1, 5.22E-1,
                             5.06E-1, 4.90E-1, 4.78E-1, 4.67E-1, 4.57E-1,
                             4.48E-1, 4.38E-1, 4.31E-1, 4.24E-1, 4.20E-1,
                             4.14E-1, 4.11E-1, 4.06E-1])
        self.p0 = [0.5, 1.5, -1.0, 1E-2, 2E-2]
        self.pt = [0.3754, 1.9358, -1.4647, 0.0129, 0.0221]

    def osborne(self, p, x):
        y = p[0] + p[1]*np.exp(-p[3]*x) + p[2]*np.exp(-p[4]*x);
        return y

    def jac_osborne(self, p, x):
        jac = np.empty((33,5))
        tmp1 = np.exp(-p[3]*x)
        tmp2 = np.exp(-p[4]*x)
        jac[:,0] = 1
        jac[:,1] = tmp1
        jac[:,2] = tmp2
        jac[:,3] = -p[1]*x*tmp1
        jac[:,4] = -p[2]*x*tmp2
        return jac

    def test_der(self):
        kw = {'jacf': self.jac_osborne}
        kw.update(_OPTS)
        ret = run_levmar(self.osborne, self.p0, self.y, args=(self.x,), **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=4)

    def test_diff(self):
        ret = run_levmar(self.osborne, self.p0, self.y, args=(self.x,), **_OPTS)
        assert_array_almost_equal(ret.p, self.pt, decimal=4)

    def test_cdiff(self):
        kw = {'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.osborne, self.p0, self.y, args=(self.x,), **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=4)


class TestHelVal(TestCase):
    __test__ = True

    def setUp(self):
        self.y = np.zeros(3, np.float64)
        self.p0 = [-1, 2, 2]
        self.pt = [1, 0, 0]

    def helval(self, p):
        y = np.empty(3)

        if p[0] < 0:
            theta = atan(p[1]/p[0])/(2.0*pi) + 0.5
        elif p[0] > 0:
            theta = atan(p[1]/p[0])/(2.0*pi)
        else:
            theta = 0.25 if p[1] >= 0 else -0.25

        y[0] = 10*(p[2] - 10*theta)
        y[1] = 10*(sqrt(p[0]*p[0] + p[1]*p[1]) - 1)
        y[2] = p[2]
        return y

    def jac_helval(self, p):
        jac = np.empty((3,3))
        tmp = p[0]*p[0] + p[1]*p[1]

        jac[0,0] = 50.0*p[1]/(pi*tmp)
        jac[0,1] = -50.0*p[0]/(pi*tmp)
        jac[0,2] = 10.0

        jac[1,0] = 10.0*p[0]/sqrt(tmp)
        jac[1,1] = 10.0*p[1]/sqrt(tmp)
        jac[1,2] = 0.0

        jac[2,0] = 0.0
        jac[2,1] = 0.0
        jac[2,2] = 1.0

        return jac

    def test_der(self):
        kw = {'jacf': self.jac_helval}
        kw.update(_OPTS)
        ret = run_levmar(self.helval, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt)

    def test_diff(self):
        ret = run_levmar(self.helval, self.p0, self.y, **_OPTS)
        assert_array_almost_equal(ret.p, self.pt)

    def test_cdiff(self):
        kw = {'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.helval, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt)


class TestBt3(TestCase):
    __test__ = True

    def setUp(self):
        self.y = np.zeros(5, np.float64)
        self.p0 = [2, 2, 2, 2, 2]
        self.pt = [-0.76744, 0.25581, 0.62791, -0.11628, 0.25581]
        self.A = np.array([[1.0, 3.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 1.0, -2.0],
                           [0.0, 1.0, 0.0, 0.0, -1.0]])
        self.b = np.array([0.0, 0.0, 0.0])

    def bt3(self, p):
        y = np.empty(5)
        t1 = p[0] - p[1]
        t2 = p[1] + p[2] - 2
        t3 = p[3] - 1
        t4 = p[4] - 1
        y[:] = t1*t1 + t2*t2 + t3*t3 + t4*t4
        return y

    def jac_bt3(self, p):
        jac = np.empty((5,5))
        t1 = p[0] - p[1]
        t2 = p[1] + p[2] - 2
        t3 = p[3] - 1
        t4 = p[4] - 1

        jac[:,0] = 2.0*t1
        jac[:,1] = 2.0*(t2 - t1)
        jac[:,2] = 2.0*t2
        jac[:,3] = 2.0*t3
        jac[:,4] = 2.0*t4

        return jac

    def test_der(self):
        kw = {'jacf': self.jac_bt3, 'A': self.A, 'b': self.b}
        kw.update(_OPTS)
        ret = run_levmar(self.bt3, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=4)

    def test_diff(self):
        kw = {'A': self.A, 'b': self.b}
        kw.update(_OPTS)
        ret = run_levmar(self.bt3, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=4)

    def test_cdiff(self):
        kw = {'A': self.A, 'b': self.b, 'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.bt3, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=4)


class TestHS28(TestCase):
    __test__ = True

    def setUp(self):
        self.y = np.zeros(3, np.float64)
        self.p0 = [-4, 1, 1]
        self.pt = [0.5, -0.5, 0.5]
        self.A = np.array([1.0, 2.0, 3.0]).reshape(1,-1)
        self.b = np.array([1.0])

    def hs28(self, p):
        y = np.empty(3)
        t1 = p[0] + p[1]
        t2 = p[1] + p[2]
        y[:] = t1*t1 + t2*t2
        return y

    def jac_hs28(self, p):
        jac = np.empty((3,3))
        t1 = p[0] + p[1]
        t2 = p[1] + p[2]

        jac[:,0] = 2*t1
        jac[:,1] = 2*(t1 + t2)
        jac[:,2] = 2*t2

        return jac

    def test_der(self):
        kw = {'jacf': self.jac_hs28, 'A': self.A, 'b': self.b}
        kw.update(_OPTS)
        ret = run_levmar(self.hs28, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=5)

    def test_diff(self):
        kw = {'A': self.A, 'b': self.b}
        kw.update(_OPTS)
        ret = run_levmar(self.hs28, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=3)

    def test_cdiff(self):
        kw = {'A': self.A, 'b': self.b, 'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.hs28, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=4)


class TestHS48(TestCase):
    __test__ = True

    def setUp(self):
        self.y = np.zeros(5, np.float64)
        self.p0 = [3, 5, -3, 2, -2]
        self.pt = [1, 1, 1, 1, 1]
        self.A = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                           [0.0, 0.0, 1.0, -2.0, -2.0]])
        self.b = np.array([5.0, -3.0])

    def hs48(self, p):
        y = np.empty(5)
        t1 = p[0] - 1
        t2 = p[1] - p[2]
        t3 = p[3] - p[4]
        y[:] = t1*t1 + t2*t2 + t3*t3
        return y

    def jac_hs48(self, p):
        jac = np.empty((5,5))
        t1 = p[0] - 1
        t2 = p[1] - p[2]
        t3 = p[3] - p[4]

        jac[:,0] = 2*t1
        jac[:,1] = 2*t2
        jac[:,2] = -2*t2
        jac[:,3] = 2*t3
        jac[:,4] = -2*t3

        return jac

    def test_der(self):
        kw = {'jacf': self.jac_hs48, 'A': self.A, 'b': self.b}
        kw.update(_OPTS)
        ret = run_levmar(self.hs48, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=5)

    def test_diff(self):
        kw = {'A': self.A, 'b': self.b}
        kw.update(_OPTS)
        ret = run_levmar(self.hs48, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=5)

    def test_cdiff(self):
        kw = {'A': self.A, 'b': self.b, 'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.hs48, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=5)


class TestHS51(TestCase):
    __test__ = True

    def setUp(self):
        self.y = np.zeros(5, np.float64)
        self.p0 = [2.5, 0.5, 2.0, -1.0, 0.5]
        self.pt = [1, 1, 1, 1, 1]
        self.A = np.array([[1.0, 3.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 1.0, -2.0],
                           [0.0, 1.0, 0.0, 0.0, -1.0]])
        self.b = np.array([4.0, 0.0, 0.0])

    def hs51(self, p):
        y = np.empty(5)
        t1 = p[0] - p[1]
        t2 = p[1] + p[2] - 2
        t3 = p[3] - 1
        t4 = p[4] - 1
        y[:] = t1*t1 + t2*t2 + t3*t3 + t4*t4
        return y

    def jac_hs51(self, p):
        jac = np.empty((5,5))
        t1 = p[0] - p[1]
        t2 = p[1] + p[2] - 2
        t3 = p[3] - 1
        t4 = p[4] - 1

        jac[:,0] = 2*t1
        jac[:,1] = 2*(t2 - t1)
        jac[:,2] = 2*t2
        jac[:,3] = 2*t3
        jac[:,4] = 2*t4

        return jac

    def test_der(self):
        kw = {'jacf': self.jac_hs51, 'A': self.A, 'b': self.b}
        kw.update(_OPTS)
        ret = run_levmar(self.hs51, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=4)

    def test_diff(self):
        kw = {'A': self.A, 'b': self.b}
        kw.update(_OPTS)
        ret = run_levmar(self.hs51, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=3)

    def test_cdiff(self):
        kw = {'A': self.A, 'b': self.b, 'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.hs51, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=3)


class TestHS01(TestCase):
    __test__ = True

    def setUp(self):
        self.y = np.zeros(2, np.float64)
        self.p0 = [-2, 1]
        self.pt = [1, 1]
        self.bc = (None, (-1.5, None))

        self.jacf_buf = np.empty(4, dtype=np.float64)

    def hs01(self, p):
        y = np.empty(2)
        y[0] = 10*(p[1]-p[0]*p[0])
        y[1] = 1 - p[0]
        return y

    def jac_hs01(self, p):
        jac = np.empty((2,2))

        jac[0,0]=-20.0*p[0]
        jac[0,1]=10.0
        jac[1,0]=-1.0
        jac[1,1]=0.0

        return jac

    def test_der(self):
        kw = {'jacf': self.jac_hs01}
        kw.update(_OPTS)
        ret = run_levmar(self.hs01, self.p0, self.y, bounds=self.bc, **kw)
        assert_array_almost_equal(ret.p, self.pt)

    def test_diff(self):
        ret = run_levmar(self.hs01, self.p0, self.y, bounds=self.bc, **_OPTS)
        assert_array_almost_equal(ret.p, self.pt)

    def test_cdiff(self):
        kw = {'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.hs01, self.p0, self.y, bounds=self.bc, **kw)
        assert_array_almost_equal(ret.p, self.pt)


class TestHS21(TestCase):
    __test__ = True

    def setUp(self):
        self.y = np.zeros(2, np.float64)
        self.p0 = [4, -1]
        self.pt = [2, 0]
        self.bc = ((2, 50), (-50, 50))

    def hs21(self, p):
        y = np.empty(2)
        y[0] = p[0]/10
        y[1] = p[1]
        return y

    def jac_hs21(self, p):
        jac = np.empty((2,2))

        jac[0,0]=0.1
        jac[0,1]=0.0

        jac[1,0]=0.0
        jac[1,1]=1.0

        return jac

    def test_der(self):
        kw = {'jacf': self.jac_hs21}
        kw.update(_OPTS)
        ret = run_levmar(self.hs21, self.p0, self.y, bounds=self.bc, **kw)
        assert_array_almost_equal(ret.p, self.pt)

    def test_diff(self):
        ret = run_levmar(self.hs21, self.p0, self.y, bounds=self.bc, **_OPTS)
        assert_array_almost_equal(ret.p, self.pt)

    def test_cdiff(self):
        kw = {'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.hs21, self.p0, self.y, bounds=self.bc, **kw)
        assert_array_almost_equal(ret.p, self.pt)


class TestHatfldB(TestCase):
    __test__ = True

    def setUp(self):
        self.y = np.zeros(4, np.float64)
        self.p0 = [0.1, 0.1, 0.1, 0.1]
        self.pt = [0.947214, 0.8, 0.64, 0.4096]
        self.bc = ((0, None), (0, 0.8), (0, None), (0, None))

    def hatfldb(self, p):
        y = np.empty(4)
        y[0] = p[0] - 1
        for i in range(1,4):
            y[i] = p[i-1] - sqrt(p[i])
        return y

    def jac_hatfldb(self, p):
        jac = np.empty((4,4))

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
        kw = {'jacf': self.jac_hatfldb}
        kw.update(_OPTS)
        ret = run_levmar(self.hatfldb, self.p0, self.y, bounds=self.bc, **kw)
        assert_array_almost_equal(ret.p, self.pt)

    def test_diff(self):
        ret = run_levmar(self.hatfldb, self.p0, self.y, bounds=self.bc, **_OPTS)
        assert_array_almost_equal(ret.p, self.pt)

    def test_cdiff(self):
        kw = {'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.hatfldb, self.p0, self.y, bounds=self.bc, **kw)
        assert_array_almost_equal(ret.p, self.pt)


class TestHatfldC(TestCase):
    __test__ = True

    def setUp(self):
        self.y = np.zeros(4, np.float64)
        self.p0 = [0.9, 0.9, 0.9, 0.9]
        self.pt = [1, 1, 1, 1]
        self.bc = ((0, 10), (0, 10), (0, 10), (0, 10))

        self.func_buf = np.empty(4, dtype=np.float64)
        self.jacf_buf = np.empty(16, dtype=np.float64)

    def hatfldc(self, p):
        y = np.empty(4)
        y[0] = p[0] - 1
        for i in range(1,3):
            y[i] = p[i-1] - sqrt(p[i])
        y[3] = p[3] - 1
        return y

    def jac_hatfldc(self, p):
        jac = np.empty((4,4))

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
        kw = {'jacf': self.jac_hatfldc}
        kw.update(_OPTS)
        ret = run_levmar(self.hatfldc, self.p0, self.y, bounds=self.bc, **kw)
        assert_array_almost_equal(ret.p, self.pt)

    def test_diff(self):
        ret = run_levmar(self.hatfldc, self.p0, self.y, bounds=self.bc, **_OPTS)
        assert_array_almost_equal(ret.p, self.pt)

    def test_cdiff(self):
        kw = {'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.hatfldc, self.p0, self.y, bounds=self.bc, **kw)
        assert_array_almost_equal(ret.p, self.pt)


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
        self.y = np.zeros(5, np.float64)
        self.p0 = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
        self.pt = [0.0034, 31.3265, 0.0684, 0.8595, 0.0370]
        self.bc = ((1e-3, 100), (1e-3, 100), (1e-3, 100), (1e-3, 100), (1e-3, 100))

    def combust(self, p):
        y = np.empty(5)
        R, R5, R6, R7, R8, R9, R10 = \
                self.R, self.R5, self.R6, self.R7, self.R8, self.R9, self.R10

        y[0] = p[0]*p[1] + p[0] - 3*p[4]
        y[1] = 2*p[0]*p[1] + p[0] + 3*R10*p[1]*p[1] + p[1]*p[2]*p[2] + \
                R7*p[1]*p[2] + R9*p[1]*p[3] + R8*p[1] - R*p[4]
        y[2] = 2*p[1]*p[2]*p[2] + R7*p[1]*p[2] + 2*R5*p[2]*p[2] + R6*p[2] - 8*p[4]
        y[3] = R9*p[1]*p[3] + 2*p[3]*p[3] - 4*R*p[4]
        y[4] = p[0]*p[1] + p[0] + R10*p[1]*p[1] + p[1]*p[2]*p[2] + \
                R7*p[1]*p[2] + R9*p[1]*p[3] + R8*p[1] + R5*p[2]*p[2] + \
                R6*p[2] + p[3]*p[3] - 1.0
        return y

    def jac_combust(self, p):
        jac = np.empty((5,5))
        R, R5, R6, R7, R8, R9, R10 = \
                self.R, self.R5, self.R6, self.R7, self.R8, self.R9, self.R10

        jac[0,0] = p[1]+1
        jac[0,1] = p[0]
        jac[0,2] = 0
        jac[0,3] = 0
        jac[0,4] = -3

        jac[1,0] = 2*p[1]+1
        jac[1,1] = 2*p[0]+6*R10*p[1] + p[2]*p[2] + R7*p[2] + R9*p[3] + R8
        jac[1,2] = 2*p[1]*p[2] + R7*p[1]
        jac[1,3] = R9*p[1]
        jac[1,4] = -R

        jac[2,0] = 0
        jac[2,1] = 2*p[2]*p[2] + R7*p[2]
        jac[2,2] = 4*p[1]*p[2] + R7*p[1] + 4*R5*p[2] + R6
        jac[2,3] = -8
        jac[2,4] = 0

        jac[3,0] = 0
        jac[3,1] = R9*p[3]
        jac[3,2] = 0
        jac[3,3] = R9*p[1] + 4*p[3]
        jac[3,4] = -4*R

        jac[4,0] = p[1] + 1
        jac[4,1] = p[0] + 2*R10*p[1] + p[2]*p[2] + R7*p[2] + R9*p[3] + R8
        jac[4,2] = 2*p[1]*p[2] + R7*p[1] + 2*R5*p[2] + R6
        jac[4,3] = R9*p[1] + 2*p[3]
        jac[4,4] = 0

        return jac

    def test_der(self):
        kw = {'jacf': self.jac_combust}
        kw.update(_OPTS)
        ret = run_levmar(self.combust, self.p0, self.y, bounds=self.bc, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=1)

    def test_diff(self):
        ret = run_levmar(self.combust, self.p0, self.y, bounds=self.bc, **_OPTS)
        assert_array_almost_equal(ret.p, self.pt, decimal=1)

    def test_cdiff(self):
        kw = {'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.combust, self.p0, self.y, bounds=self.bc, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=1)


class TestMods235(TestCase):
    __test__ = True

    def setUp(self):
        self.y = np.zeros(2, np.float64)
        self.p0 = [-2, 3, 1]
        self.pt = [-1.725, 2.9, 0.725]
        self.A = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, -4.0]])
        self.b = np.array([-1.0, 0.0])
        self.bounds = ((None, None), (0.1, 2.9), (0.7, None))

    def mods235(self, p):
        y = np.empty(2)
        y[0] = 0.1*(p[0] - 1.0);
        y[1] = p[1] - p[0]*p[0];
        return y

    def jac_mods235(self, p):
        jac = np.empty(2*3)

        jac[0] = 0.1
        jac[1] = 0.0
        jac[2] = 0.0

        jac[3] = -2.0*p[0]
        jac[4] = 1.0
        jac[5] = 0.0

        return jac.reshape(2,3)

    def test_der(self):
        kw = {'jacf': self.jac_mods235,
              'A': self.A, 'b': self.b, 'bounds': self.bounds}
        kw.update(_OPTS)
        ret = run_levmar(self.mods235, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt)

    def test_diff(self):
        kw = {'A': self.A, 'b': self.b, 'bounds': self.bounds}
        kw.update(_OPTS)
        ret = run_levmar(self.mods235, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt)

    def test_cdiff(self):
        kw = {'A': self.A, 'b': self.b, 'bounds': self.bounds, 'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.mods235, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt)


class TestMod1HS52(TestCase):
    __test__ = True

    def setUp(self):
        self.y = np.zeros(4, np.float64)
        self.p0 = [2, 2, 2, 2, 2]
        self.pt = [-0.09, 0.03, 0.25, -0.19, 0.03]
        self.A = np.array([[1.0, 3.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 1.0, -2.0],
                           [0.0, 1.0, 0.0, 0.0, -1.0]])
        self.b = np.array([0.0, 0.0, 0.0])
        self.bounds = ((-0.09, None), (0.0, 0.3), (None, 0.25),
                       (-0.2, 0.3), (0.0, 0.3))

    def mod1hs52(self, p):
        y = np.empty(4)
        y[0] = 4*p[0] - p[1]
        y[1] = p[1] + p[2] - 2
        y[2] = p[3] - 1
        y[3] = p[4] - 1
        return y

    def jac_mod1hs52(self, p):
        jac = np.empty(4*5)

        jac[0] = 4.0
        jac[1] = -1.0
        jac[2] = 0.0
        jac[3] = 0.0
        jac[4] = 0.0

        jac[5] = 0.0
        jac[6] = 1.0
        jac[7] = 1.0
        jac[8] = 0.0
        jac[9] = 0.0

        jac[10] = 0.0
        jac[11] = 0.0
        jac[12] = 0.0
        jac[13] = 1.0
        jac[14] = 0.0

        jac[15] = 0.0
        jac[16] = 0.0
        jac[17] = 0.0
        jac[18] = 0.0
        jac[19] = 1.0

        return jac.reshape(4,5)

    def test_der(self):
        kw = {'jacf': self.jac_mod1hs52,
              'A': self.A, 'b': self.b, 'bounds': self.bounds}
        kw.update(_OPTS)
        ret = run_levmar(self.mod1hs52, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt)

    def test_diff(self):
        kw = {'A': self.A, 'b': self.b, 'bounds': self.bounds}
        kw.update(_OPTS)
        ret = run_levmar(self.mod1hs52, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt, decimal=3)

    def test_cdiff(self):
        kw = {'A': self.A, 'b': self.b, 'bounds': self.bounds, 'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.mod1hs52, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt)


class TestMod2HS52(TestCase):
    __test__ = True

    def setUp(self):
        self.y = np.zeros(5, np.float64)
        self.p0 = [2, 2, 2, 2, 2]
        self.pt = [0.5, 2.0, 0.0, 1.0, 1.0]
        self.C = np.array([[1.0, 3.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 1.0, -2.0],
                           [0.0, -1.0, 0.0, 0.0, 1.0]])
        self.d = np.array([-1.0, -2.0, -7.0])

    def mod2hs52(self, p):
        y = np.empty(5)
        y[0] = 4*p[0] - p[1]
        y[1] = p[1] + p[2] - 2
        y[2] = p[3] - 1
        y[3] = p[4] - 1
        y[4] = p[0] - 0.5
        return y

    def jac_mod2hs52(self, p):
        jac = np.empty(5*5)

        jac[0] = 4.0
        jac[1] = -1.0
        jac[2] = 0.0
        jac[3] = 0.0
        jac[4] = 0.0

        jac[5] = 0.0
        jac[6] = 1.0
        jac[7] = 1.0
        jac[8] = 0.0
        jac[9] = 0.0

        jac[10] = 0.0
        jac[11] = 0.0
        jac[12] = 0.0
        jac[13] = 1.0
        jac[14] = 0.0

        jac[15] = 0.0
        jac[16] = 0.0
        jac[17] = 0.0
        jac[18] = 0.0
        jac[19] = 1.0

        jac[20] = 1.0
        jac[21] = 0.0
        jac[22] = 0.0
        jac[23] = 0.0
        jac[24] = 0.0

        return jac.reshape(5,5)

    def test_der(self):
        kw = {'jacf': self.jac_mod2hs52, 'C': self.C, 'd': self.d}
        kw.update(_OPTS)
        ret = run_levmar(self.mod2hs52, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt)

    def test_diff(self):
        kw = {'C': self.C, 'd': self.d}
        kw.update(_OPTS)
        ret = run_levmar(self.mod2hs52, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt)

    def test_cdiff(self):
        kw = {'C': self.C, 'd': self.d, 'cdif': True}
        kw.update(_OPTS)
        ret = run_levmar(self.mod2hs52, self.p0, self.y, **kw)
        assert_array_almost_equal(ret.p, self.pt)


if __name__ == '__main__':
    run_module_suite()
