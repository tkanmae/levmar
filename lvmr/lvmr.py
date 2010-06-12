#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
import _lvmr
from _lvmr import (Output, LMError, LMRuntimeError, LMUserFuncError,
                   LMWarning, _LM_MAXITER, _LM_EPS1, _LM_EPS2, _LM_EPS3)

__Data = _lvmr._Data
__Model = _lvmr._Model
__Levmar = _lvmr._Levmar


class Data(__Data):
    __slots__ = ['x', 'y', 'wt']
    """The Data class stores the data to fit.

    Attributes
    ----------
    x : array_like, shape (n,)
    y : array_like, shape (n,)
    """
    def __init__(self, x, y, wt=None):
        __Data.__init__(self, x, t, wt)


class Model(__Model):
    """The Model class stores information about the model.

    Attributes
    ----------
    func : callable
        A function or method taking, at least, one length of n vector
        and returning a length of m vector.  The signature must be like
        `func(p, x, args) -> y`.
    jacf : callable, optional
        A function or method to compute the Jacobian of `func`.  The
        signature must be like `jacf(p, x, args)`.  If this is None, a
        approximated Jacobian will be used.
    extra_args : tuple, optional
        Extra arguments passed to `func` (and `jacf`) in this tuple.
    """
    __slot__ = ['func', 'jacf', 'extra_args']
    def __init__(self, func, jacf=None, extra_args=()):
        __Model.__init__(self, func, jacf, extra_args)


class Levmar(__Levmar):
    __slots__ = ['data', 'model']
    def __init__(self, data, model):
        __Levmar.__init__(self, data, model)

    def run(self, p0, bounds=None, A=None, b=None, C=None, d=None,
            mu=1e-3, eps1=_LM_EPS1, eps2=_LM_EPS2, eps3=_LM_EPS3,
            maxiter=1000, cntdif=False):
        """Run the fitting.

        Parameters
        ----------
        p0 : array_like, shape (m,)
            The initial estimate of the parameters.
        bounds : tuple/list, length m
            Box-constraints. Each constraint can be a None or a tuple of two
            float/Nones.  None in the first case means no constraint, and
            None in the second case means -Inf/+Inf.
        A : array_like, shape (k1,m), optional
            A linear equation constraints matrix
        b : array_like, shape (k1,), optional
            A right-hand equation linear constraint vector
        C : array_like, shape (k2,m), optional
            A linear inequality constraints matrix
        d : array_like, shape (k2,), optional
            A right-hand linear inequality constraint vector
        mu : float, optional
            The scale factor for initial \mu
        eps1 : float, optional
            The stopping threshold for ||J^T e||_inf
        eps2 : float, optional
            The stopping threshold for ||Dp||_2
        eps3 : float, optional
            The stopping threshold for ||e||_2
        maxiter : int, optional
            The maximum number of iterations.
        cntdif : {True, False}, optional
            If this is True, the Jacobian is approximated with central
            differentiation.

        Returns
        -------
        output : lvmr.Output
            The output of the minimization
        """
        __Levmar.run(p0, bounds, A, b, C, d,
                     mu, eps1, eps2, eps3, maxiter, cntdif)


def levmar(func, p0, y, args=(), jacf=None,
           bounds=None, A=None, b=None, C=None, d=None,
           mu=1e-3, eps1=_LM_EPS1, eps2=_LM_EPS2, eps3=_LM_EPS3,
           maxiter=1000, cntdif=False):
    return _lvmr.levmar(func, p0,  y, args, jacf,
                        bounds, A, b, C, d,
                        mu, eps1, eps2, eps3, maxiter, cntdif)
levmar.__doc__ = _lvmr.levmar.__doc__
