#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
import numpy as np
from scipy.odr import odr
from scipy.optimize import leastsq

from lvmr import levmar


class BenchBase(object):
    def __init__(self):
        self.setup()
        np.random.seed(1)
        self.yt = self.fcn(self.p0, self.x)

    def setup(self):
        """Define self.x and self.p0 here."""
        raise NotImplementedError

    def fcn(self, p, x):
        """Define a fitting function here."""
        raise NotImplementedError

    def _y(self):
        return self.yt + 0.1 * np.random.randn(self.x.size)

    def levmar(self):
        ret = levmar(self.fcn, self.p0, self._y(), args=(self.x,))

    def leastsq(self):
        fcn = lambda p, x, y : y - self.fcn(p, x)
        ret = leastsq(fcn, self.p0, args=(self.x, self._y()), full_output=1)

    def odr(self):
        ret = odr(self.fcn, self.p0, self._y(), self.x, full_output=1)


class BenchExp(BenchBase):

    def setup(self):
        self.x = np.arange(40, dtype=np.float64)
        self.p0 = 4.0, 0.2, 2.0

    def fcn(self, p, x):
        return p[0] * np.exp(-p[1]*x) + p[2]


class BenchGauss(BenchBase):

    def setup(self):
        self.x = np.linspace(-3, 3)
        self.p0 = 4.0, 0.2, 2.0

    def fcn(self, p, x):
        return p[0] * np.exp(-p[1]*x**2) + p[2]


class BenchLorentz(BenchBase):

    def setup(self):
        self.x = np.linspace(-3, 3)
        self.p0 = 4.0, 0.2, 2.0

    def fcn(self, p, x):
        return p[0] * np.exp(-p[1]*x**2) + p[2]


def run_timing(bench):
    from timeit import Timer

    funcs = "levmar", "leastsq", "odr"

    for f in funcs:
        print("--- {0}()".format(f))
        t = Timer("Bench{0}().{1}()".format(bench, f),
                  "from __main__ import Bench{0}".format(bench))
        res = min(t.repeat(3, number=300))
        print ("  {0:g} sec".format(res))


def main():
    benches = ['Exp', 'Gauss', 'Lorentz']
    for b in benches:
        print (" Bench{0} ".format(b)).center(50, '*')
        run_timing(b)


if __name__ == '__main__':
    main()

