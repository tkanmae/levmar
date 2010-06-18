#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
from timeit import Timer
import numpy as np
from scipy.odr import odr
from scipy.optimize import leastsq

from lvmr import levmar


class BenchBase(object):
    def __init__(self, x, p0):
        self.x = x
        self.p0 = p0

    def fcn(self, p, x):
        """Define a fitting function here."""
        raise NotImplementedError

    def get_data(self):
        y = self.fcn(self.p0, self.x) + \
                0.1 * np.random.randn(self.x.size)
        return self.x, y, self.p0

    def set_seed(self, seed):
        np.random.seed(seed)

    def levmar(self):
        x, y, p0 = self.get_data()
        ret = levmar(self.fcn, p0, y, args=(x,))

    def leastsq(self):
        fcn = lambda p, x, y : y - self.fcn(p, x)
        x, y, p0 = self.get_data()
        ret = leastsq(fcn, p0, args=(x, y), full_output=1)

    def odr(self):
        x, y, p0 = self.get_data()
        ret = odr(self.fcn, p0, y, x, full_output=1)


class BenchExp(BenchBase):
    def fcn(self, p, x):
        return p[0] * np.exp(-p[1]*x) + p[2]


class BenchGauss(BenchBase):
    def fcn(self, p, x):
        return p[0] * np.exp(-((x-p[1])/p[2])**2) + p[3]


def timing_exp(size, cycle):
    for f in ("levmar", "leastsq", "odr"):
        s1 = "BenchExp(x, p0).{0}()".format(f)
        s2 = ("import numpy as np\n"
              "from __main__ import BenchExp\n"
              "x = np.linspace(0, 10, {0})\n"
              "p0 = 4.0, 0.2, 2.0\n"
              .format(size))
        t = min(Timer(s1, s2).repeat(3, number=cycle))
        print("  {0:<8}: {1:g}".format(f, t))


def timing_gauss(size, cycle):
    for f in ("levmar", "leastsq", "odr"):
        s1 = "BenchGauss(x, p0).{0}()".format(f)
        s2 = ("import numpy as np\n"
              "from __main__ import BenchGauss\n"
              "x = np.linspace(-3, 3, {0})\n"
              "p0 = 4.0, 0.5, 0.5, 0.1\n"
              .format(size))
        t = min(Timer(s1, s2).repeat(3, number=cycle))
        print("  {0:<8}: {1:g}".format(f, t))


def main():
    sizes = 100, 300
    cycles = 100, 300

    print(" BenchExp ".center(50, '*'))
    for size in sizes:
        for cycle in cycles:
            print("-- size={0}, cycle={1}".format(size, cycle))
            timing_exp(size, cycle)

    print(" BenchGauss ".center(50, '*'))
    for size in sizes:
        for cycle in cycles:
            print("-- size={0}, cycle={1}".format(size, cycle))
            timing_gauss(size, cycle)

if __name__ == '__main__':
    main()

