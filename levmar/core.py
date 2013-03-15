#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np

import _levmar


def levmar(func, p0, y, args=(), jacf=None,
           bounds=None, A=None, b=None, C=None, d=None,
           mu=1e-3, eps1=_levmar._LM_EPS1, eps2=_levmar._LM_EPS2, eps3=_levmar._LM_EPS3,
           maxit=1000, cdif=False, full_output=False):
    """
    Parameters
    ----------
    func : callable
        A function or method computing the model function.  It takes, at
        least, one length of m vector and returns a length of n vector.
    p0 : array_like, shape (m,)
        The initial estimate of the parameters.
    y : array_like, shape (n,)
        The dependent data, or the observation.
    args : tuple, optional
        Extra arguments passed to `func` (and `jacf`).
    jacf : callable, optional
        A function or method computing the Jacobian of `func`.  It
        takes, at least, one length of m vector and returns the (nxm)
        Jacobian matrix or a campatible C-contiguous vector.  If it is
        None, the Jacobian will be approximated.
    bounds : tuple/list, length m, optional
        Box constraints.  Each constraint can be a tuple of two
        floats/Nones or None.  A tuple determines the (inclusive) lower
        and upper bound, and None means no constraint.  If one of two
        values in a tuple is None, then the bound is semi-definite.
    A : array_like, shape (k1,m), optional
        A linear equation constraints matrix.
    b : array_like, shape (k1,), optional
        A right-hand linear equation constraint vector.
    C : array_like, shape (k2,m), optional
        A linear *inequality* constraints matrix.
    d : array_like, shape (k2,), optional
        A right-hand linear *inequality* constraint vector.
    mu : float, optional
        The scale factor for initial mu.
    eps1 : float, optional
        The stopping threshold for ||J^T e||_inf.
    eps2 : float, optional
        The stopping threshold for ||Dp||_2.
    eps3 : float, optional
        The stopping threshold for ||e||_2.
    maxit : int, optional
        The maximum number of iterations.
    cdif : {True, False}, optional
        If this is True, the Jacobian is approximated with central
        differentiation.

    Returns
    -------
    p : ndarray, shape=(m,)
        The best-fit parameters.
    covr : ndarray, shape=(m,m)
        The covariance of the best-fit parameters.
    info : tuple
        Various information regarding the minimization.
            0: ||e||_2 at `p0`
            1:
                0: 2-norm of e
                1: infinity-norm of J^T.e
                2: 2-norm of Dp
                3: mu / max{(J^T.J)_ii}
            2: The number of the iterations
            3: The reason for the termination
            4: The number of `func` evaluations
            5: The number of `jacf` evaluations
            6: The number of the linear system solved

    Notes
    -----
    * Linear equation constraints are specified as A*p=b where A is
    (k1xm) matrix and b is (k1x1) vector.

    * Linear *inequality* constraints are defined as C*p>=d where C
    is (k2xm) matrix and d is (k2x1) vector.
    """
    p, covr, info =  _levmar.levmar(func, p0,  y, args, jacf, bounds,
                                    A, b, C, d, mu, eps1, eps2, eps3, maxit, cdif)
    return p, covr, info
