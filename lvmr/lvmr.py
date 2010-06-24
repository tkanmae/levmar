#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
import warnings
from numpy import (array, float64)
import _lvmr
from _lvmr import (_run_levmar, Output,
                   LMError, LMRuntimeError, LMUserFuncError, LMWarning,
                   _LM_EPS1, _LM_EPS2, _LM_EPS3)


__all__ = ['Data', 'Model', 'Levmar', 'levmar', 'Output',
           'LMError', 'LMRuntimeError', 'LMUserFuncError', 'LMWarning']


class Data(object):
    """The Data class stores the data to fit.

    Parameters
    ----------
    x : array_like, shape (n,)
        The independent data.
    y : array_like, shape (n,)
        The dependent data, or the observation.

    Attributes
    ----------
    x : ndarray, shape (n,)
    y : ndarray, shape (n,)

    Raises
    ------
    ValueError
        If the sizes of the vectors do not match.
    """
    __slots__ = ['x', 'y', 'wt']
    def __init__(self, x, y, wt=None):
        x = array(x, dtype=float64, order='C', copy=False, ndmin=1)
        y = array(y, dtype=float64, order='C', copy=False, ndmin=1)
        if x.size != y.size:
            raise ValueError("`x` and `y` must have the same size")
        if wt is not None:
            wt = array(wt, dtype=float64, order='C', copy=False, ndmin=1)
            if wt.size != y.size:
                raise ValueError("`wt` and `y` must have the same size")
            warnings.warn("Sorry, but weighted least square is NOT "
                          "implemented yet.", LMWarning)
        self.x = x
        self.y = y
        self.wt = wt


class Model(object):
    """The Model class stores information about the model.

    Parameters
    ----------
    func : callable
        A function or method computing the model function.  It takes, at
        least, one length of m vector and returns a length of n vector.
    jacf : callable, optional
        A function or method computing the Jacobian of `func`.  It
        takes, at least, one length of m vector and returns the (nxm)
        Jacobian matrix or a campatible C-contiguous vector.  If it is
        None, the Jacobian will be approximated.
    extra_args : tuple, optional
        Extra arguments passed to `func` (and `jacf`).

    Attributes
    ----------
    func : callable
    jacf : callable or None
    extra_args : tuple

    Raises
    ------
    TypeError
        If the functions are not callable.
    """
    __slot__ = ['func', 'jacf', 'extra_args']
    def __init__(self, func, jacf=None, extra_args=()):
        if not callable(func):
            raise TypeError("`func` must be callable")
        if jacf is not None and not callable(jacf):
            raise TypeError("`jacf` must be callable")
        if not isinstance(extra_args, tuple): extra_args= extra_args,
        self.func = func
        self.jacf = jacf
        self.extra_args = extra_args


class Levmar(object):
    __slots__ = ['data', 'model']
    """
    Parameters
    ----------
    data : lvmr.Data
        An instance of `lvmr.Data`
    model : lvmr.Model
        An instance of `lvmr.Model`

    Attributes
    ----------
    data : lvmr.Data
    model : lvmr.Model


    Raises
    ------
    TypeError
        If `data` and `model` are not instances of `lvmr.Data` and
        `lvmr.Model` respectively.
    """
    def __init__(self, data, model):
        if isinstance(data, Data):
            self.data = data
        else:
            raise TypeError("`data` must be a instance of `lvmr.Data`")
        if isinstance(model, Model):
            self.model = model
        else:
            raise TypeError("`model` must be a instance of `lvmr.Model`")

    def run(self, p0, bounds=None, A=None, b=None, C=None, d=None,
            mu=1e-3, eps1=_LM_EPS1, eps2=_LM_EPS2, eps3=_LM_EPS3,
            maxit=1000, cdif=False):
        """Run the fitting.

        Parameters
        ----------
        p0 : array_like, shape (m,)
            The initial estimate of the parameters.
        bounds : tuple/list, length m
            Box constraints.  Each constraint can be a tuple of two
            floats/Nones or None.  A tuple determines the (inclusive)
            lower and upper bound, and None means no constraint.  If one
            of two values in a tuple is None, then the bound is
            semi-definite.
        A : array_like, shape (k1,m), optional
            A linear equation constraints matrix
        b : array_like, shape (k1,), optional
            A right-hand equation linear constraint vector
        C : array_like, shape (k2,m), optional
            A linear *inequality* constraints matrix
        d : array_like, shape (k2,), optional
            A right-hand linear *inequality* constraint vector
        mu : float, optional
            The scale factor for initial \mu
        eps1 : float, optional
            The stopping threshold for ||J^T e||_inf
        eps2 : float, optional
            The stopping threshold for ||Dp||_2
        eps3 : float, optional
            The stopping threshold for ||e||_2
        maxit : int, optional
            The maximum number of iterations.
        cdif : {True, False}, optional
            If this is True, the Jacobian is approximated with central
            differentiation.

        Returns
        -------
        output : lvmr.Output
            The output of the minimization

        Notes
        -----
        * Linear equation constraints are specified as A*p=b where A is
        k1xm matrix and b is k1x1  vector (See comments in
        src/levmar-2.5/lmlec_core.c).

        * Linear *inequality* constraints are defined as C*p>=d where C
        is k2xm matrix and d is k2x1 vector (See comments in
        src/levmar-2.5/lmbleic_core.c).

        See Also
        --------
        levmar.Output
        """
        args = (self.data.x,) + self.model.extra_args
        return _run_levmar(
            self.model.func, p0, self.data.y, args, self.model.jacf,
            bounds, A, b, C, d, mu, eps1, eps2, eps3, maxit, cntdif)


def levmar(func, p0, y, args=(), jacf=None,
           bounds=None, A=None, b=None, C=None, d=None,
           mu=1e-3, eps1=_LM_EPS1, eps2=_LM_EPS2, eps3=_LM_EPS3,
           maxit=1000, cdif=False):
    return _run_levmar(func, p0,  y, args, jacf, bounds,
                       A, b, C, d, mu, eps1, eps2, eps3, maxit, cdif)
levmar.__doc__ = _run_levmar.__doc__
