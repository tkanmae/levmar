#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
import _levmar
from _levmar import (Output, LMError, LMRuntimeError, LMUserFuncError,
                     LMWarning)

def levmar(func, p0, y, args=(), jacf=None,
           bounds=None, A=None, b=None, C=None, d=None,
           maxiter=1000, mu=1e-3, eps1=1e-17, eps2=1e-17, eps3=1e-17, cdif=False):
    return _levmar.levmar(func, p0,  y, args, jacf,
                          bounds, A, b, C, d,
                          maxiter, mu, eps1, eps2, eps3, cdif)
levmar.__doc__ = _levmar.levmar.__doc__
