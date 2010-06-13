#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
import _lvmr
from _lvmr import (Output, LMError, LMRuntimeError, LMUserFuncError,
                   LMWarning, _LM_MAXITER, _LM_EPS1, _LM_EPS2, _LM_EPS3)

__run_levmar = _lvmr._run_levmar


class Data(object):
    """The Data class stores the data to fit.

    Attributes
    ----------
    x : array_like, shape (n,)
    y : array_like, shape (n,)
    """
    __slots__ = ['x', 'y', 'wt']
    def __init__(self, x, y, wt=None):
        x = np.array(x, dtype=np.float64, order='C', copy=False, ndim=1)
        y = np.array(y, dtype=np.float64, order='C', copy=False, ndim=1)
        if x.size != y.size:
            raise ValueError("`x` and `y` must have the same size")
        if wt is not None:
            wt = np.array(x, dtype=np.float64, order='C', copy=False, ndim=1)
            if wt.size != y.size:
                raise ValueError("`wt` and `y` must have the same size")
        self.x = x
        self.y = y
        self.wt = wt


class Model(object):
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
        if not callable(func):
            raise TypeError("`func` must be callable")
        argc_func = func.func_code.co_argcount
        if jacf is not None:
            if not callable(jacf):
                raise TypeError("`jacf` must be callable")
            argc_jacf = jacf.func_code.co_argcount
            if argc_func != argc_jacf:
                ValueError("`func` and `jacf` must have the same number of arguments")
        if not isinstance(args, tuple): args = args,
        if argc_func - 1 != len(extra_args):
            ValueError("{0} arguments expected in `extra_args`: "
                       "{1} given".format(argc_func, len(extra_args)))
        self.func = func
        self.jacf = jacf
        self.extra_args = extra_args


class Levmar(object):
    __slots__ = ['data', 'model']
    def __init__(self, data, model):
        if isinstance(data, _Data):
            self.data = data
        else:
            raise TypeError("`data` must be a instance of `lvmr._Data`")
        if isinstance(model, _Model):
            self.model = model
        else:
            raise TypeError("`model` must be a instance of `lvmr._Model`")

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
        args = (self.data.x) + self.model.extra_args
        return __run_levmar(
            self.model.func, p0, self.data.y, args, self.model.jacf,
            bounds, A, b, C, d,
            mu, eps1, eps2, eps3, maxiter, cntdif)


def levmar(func, p0, y, args=(), jacf=None,
           bounds=None, A=None, b=None, C=None, d=None,
           mu=1e-3, eps1=_LM_EPS1, eps2=_LM_EPS2, eps3=_LM_EPS3,
           maxiter=1000, cntdif=False):
    if not callable(func):
        raise TypeError("`func` must be callable")
    argc_func = func.func_code.co_argcount
    if jacf is not None:
        if not callable(jacf):
            raise TypeError("`jacf` must be callable")
        argc_jacf = jacf.func_code.co_argcount
        if argc_func != argc_jacf:
            ValueError("`func` and `jacf` must have the same number of arguments")
    if not isinstance(args, tuple): args = args,
    if argc_func - 1 != len(args):
        ValueError("{0} arguments expected in `args`: "
                   "{1} given".format(argc_func, len(args)))

    return __run_levmar(func, p0,  y, args, jacf,
                        bounds, A, b, C, d,
                        mu, eps1, eps2, eps3, maxiter, cntdif)
levmar.__doc__ = __run_levmar.__doc__
