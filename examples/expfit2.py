#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
import numpy as np
import lvmr


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

## Set data
data = lvmr.Data(x, y)

## Set model (analytic Jacobian)
model_der = lvmr.Model(expfunc, jacf=jac_expfunc)
## Run
levmar = lvmr.Levmar(data, model_der)
ret_der = levmar.run(p0)

## Set model (approximated Jacobian)
model_dif = lvmr.Model(expfunc)
## Run
levmar = lvmr.Levmar(data, model_dif)
ret_dif = levmar.run(p0)
