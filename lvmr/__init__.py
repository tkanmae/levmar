#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
from _levmar import (_levmar, LMError, LMRuntimeError, LMUserFuncError,
                     LMWarning, _LM_EPS1, _LM_EPS2, _LM_EPS3)


def levmar(func, p0, y, args=(), jacf=None,
           bounds=None, A=None, b=None, C=None, d=None,
           mu=1e-3, eps1=_LM_EPS1, eps2=_LM_EPS2, eps3=_LM_EPS3,
           maxit=1000, cdif=False):
    return _levmar(func, p0,  y, args, jacf, bounds,
                   A, b, C, d, mu, eps1, eps2, eps3, maxit, cdif)
levmar.__doc__ = _levmar.__doc__


## Add test function to the package.
from numpy.testing import Tester as __Tester
test = __Tester().test


