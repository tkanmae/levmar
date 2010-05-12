#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
import numpy as np
import levmar


def expfunc(p, x):
    ret = p[0] * np.exp(-p[1]*x) + p[2]
    return ret


def jac_expfunc(p, x):
    jac = np.empty((x.shape[0], 3))
    jac[:,0] = np.exp(-p[1]*x)
    jac[:,1] = -p[0] * x * np.exp(-p[1]*x)
    jac[:,2] = 1 * np.ones(x.size)
    return jac


np.random.seed(1)

x = np.arange(40, dtype=np.float64)
yt = 5 * np.exp(-0.1*x) + 1.0
y = yt + 0.1 * np.random.randn(x.size)


p0 = 1.0, 0.0, 0.0

for i in range(10):
    ret = levmar.bc(expfunc, p0, y, args=(x,))
    ret = levmar.bc(expfunc, p0, y, jacf=jac_expfunc, args=(x,))

