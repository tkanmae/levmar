#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

from numpy.testing import Tester as __Tester
import _levmar


__version__ = '0.2.0'


# Add test function to the package.
test = __Tester().test


def levmar(func, p0, y, args=(), jacf=None,
           mu=1.0e-03, eps1=1.5e-08, eps2=1.5e-08, eps3=1.5e-08,
           maxit=1000, cdiff=False):
    """
    Parameters
    ----------
    func: callable
        Function or method computing the model function.
    p0: array_like, shape (m,)
        Initial estimate of the parameters.
    y: array_like, shape (n,)
        Dependent data, or the observation.
    args: tuple, optional
        Extra arguments passed to `func` (and `jacf`).
    jacf: callable, optional
        Function or method computing the Jacobian of `func`.  If it is None, the
        Jacobian will be approximated.
    mu: float, optional
        Scale factor for initial mu.
    eps1: float, optional
        Stopping threshold for ||J^T e||_inf.
    eps2: float, optional
        Stopping threshold for ||Dp||_2.
    eps3: float, optional
        Stopping threshold for ||e||_2.
    maxit: int, optional
        The maximum number of iterations.
    cdiff: {True, False}, optional
        If this is True, the Jacobian is approximated with central
        differentiation.

    Returns
    -------
    p: ndarray, shape=(m,)
        Best-fit parameters.
    pcov: ndarray, shape=(m,m)
        Covariance of the best-fit parameters.
    info: tuple
        Information regarding minimization.
            0: ||e||_2 at `p0`
            1:
                0: 2-norm of e
                1: infinity-norm of J^T.e
                2: 2-norm of Dp
                3: mu / max{(J^T.J)_ii}
            2: The number of iterations
            3: The reason for termination
            4: The number of `func` evaluations
            5: The number of `jacf` evaluations
            6: The number of the linear system solved
    """
    return _levmar.levmar(func, p0, y, args, jacf,
                          mu, eps1, eps2, eps3, maxit, cdiff)


def levmar_bc(func, p0, y, bc, args=(), jacf=None,
              mu=1.0e-03, eps1=1.5e-08, eps2=1.5e-08, eps3=1.5e-08,
              maxit=1000, cdiff=False):
    """
    Parameters
    ----------
    func: callable
        Function or method computing the model function, `y = func(p, *args)`.
    p0: array_like, shape (m,)
        Initial estimate of the parameters.
    y: array_like, shape (n,)
        Dependent data, or the observation.
    bc: sequence of 2-tuples
        `(min, max)` pairs for each element of the parameters, specifying the
        (inclusive) upper and lower bounds.  Use None for one of `min` or `max`
        for specifying no bound in that direction.
    args: tuple, optional
        Extra arguments passed to `func` (and `jacf`).
    jacf: callable, optional
        Function or method computing the Jacobian of `func`.  If it is None, the
        Jacobian will be approximated.
    mu: float, optional
        Scale factor for initial mu.
    eps1: float, optional
        Stopping threshold for ||J^T e||_inf.
    eps2: float, optional
        Stopping threshold for ||Dp||_2.
    eps3: float, optional
        Stopping threshold for ||e||_2.
    maxit: int, optional
        The maximum number of iterations.
    cdiff: {True, False}, optional
        If this is True, the Jacobian is approximated with central
        differentiation.

    Returns
    -------
    p: ndarray, shape=(m,)
        Best-fit parameters.
    pcov: ndarray, shape=(m,m)
        Covariance of the best-fit parameters.
    info: tuple
        Information regarding minimization.
            0: ||e||_2 at `p0`
            1:
                0: 2-norm of e
                1: infinity-norm of J^T.e
                2: 2-norm of Dp
                3: mu / max{(J^T.J)_ii}
            2: The number of iterations
            3: The reason for termination
            4: The number of `func` evaluations
            5: The number of `jacf` evaluations
            6: The number of the linear system solved
    """
    return _levmar.levmar_bc(func, p0, y, bc, args, jacf,
                             mu, eps1, eps2, eps3, maxit, cdiff)


def levmar_lec(func, p0, y, lec, args=(), jacf=None,
               mu=1.0e-03, eps1=1.5e-08, eps2=1.5e-08, eps3=1.5e-08,
               maxit=1000, cdiff=False):
    """
    Parameters
    ----------
    func: callable
        Function or method computing the model function.
    p0: array_like, shape (m,)
        Initial estimate of the parameters.
    y: array_like, shape (n,)
        Dependent data, or the observation.
    lec: 2-tuple of ndarray
        `(A, b)` pair specifying a linear equation constraint, where `A` and `b`
        are a matrix of shape (k1, m) and a vector of shape (k1,) respectively.
    args: tuple, optional
        Extra arguments passed to `func` (and `jacf`).
    jacf: callable, optional
        Function or method computing the Jacobian of `func`.  If it is None, the
        Jacobian will be approximated.
    mu: float, optional
        Scale factor for initial mu.
    eps1: float, optional
        Stopping threshold for ||J^T e||_inf.
    eps2: float, optional
        Stopping threshold for ||Dp||_2.
    eps3: float, optional
        Stopping threshold for ||e||_2.
    maxit: int, optional
        The maximum number of iterations.
    cdiff: {True, False}, optional
        If this is True, the Jacobian is approximated with central
        differentiation.

    Returns
    -------
    p: ndarray, shape=(m,)
        Best-fit parameters.
    pcov: ndarray, shape=(m,m)
        Covariance of the best-fit parameters.
    info: tuple
        Information regarding minimization.
            0: ||e||_2 at `p0`
            1:
                0: 2-norm of e
                1: infinity-norm of J^T.e
                2: 2-norm of Dp
                3: mu / max{(J^T.J)_ii}
            2: The number of iterations
            3: The reason for termination
            4: The number of `func` evaluations
            5: The number of `jacf` evaluations
            6: The number of the linear system solved
    """
    return _levmar.levmar_lec(func, p0, y, lec, args, jacf,
                              mu, eps1, eps2, eps3, maxit, cdiff)


def levmar_blec(func, p0, y, bc, lec, args=(), jacf=None,
                mu=1.0e-03, eps1=1.5e-08, eps2=1.5e-08, eps3=1.5e-08,
                maxit=1000, cdiff=False):
    """
    Parameters
    ----------
    func: callable
        Function or method computing the model function.
    p0: array_like, shape (m,)
        Initial estimate of the parameters.
    y: array_like, shape (n,)
        Dependent data, or the observation.
    bc: sequence of 2-tuples
        `(min, max)` pairs for each element of the parameters, specifying the
        (inclusive) upper and lower bounds.  Use None for one of `min` or `max`
        for specifying no bound in that direction.
    lec: 2-tuple of ndarray
        `(A, b)` pair specifying a linear equation constraint, where `A` and `b`
        are a matrix of shape (k1, m) and a vector of shape (k1,) respectively.
    args: tuple, optional
        Extra arguments passed to `func` (and `jacf`).
    jacf: callable, optional
        Function or method computing the Jacobian of `func`.  If it is None, the
        Jacobian will be approximated.
    mu: float, optional
        Scale factor for initial mu.
    eps1: float, optional
        Stopping threshold for ||J^T e||_inf.
    eps2: float, optional
        Stopping threshold for ||Dp||_2.
    eps3: float, optional
        Stopping threshold for ||e||_2.
    maxit: int, optional
        The maximum number of iterations.
    cdiff: {True, False}, optional
        If this is True, the Jacobian is approximated with central
        differentiation.

    Returns
    -------
    p: ndarray, shape=(m,)
        Best-fit parameters.
    pcov: ndarray, shape=(m,m)
        Covariance of the best-fit parameters.
    info: tuple
        Information regarding minimization.
            0: ||e||_2 at `p0`
            1:
                0: 2-norm of e
                1: infinity-norm of J^T.e
                2: 2-norm of Dp
                3: mu / max{(J^T.J)_ii}
            2: The number of iterations
            3: The reason for termination
            4: The number of `func` evaluations
            5: The number of `jacf` evaluations
            6: The number of the linear system solved
    """
    return _levmar.levmar_blec(func, p0, y, bc, lec, args, jacf,
                               mu, eps1, eps2, eps3, maxit, cdiff)


def levmar_bleic(func, p0, y, bc, lec, lic, args=(), jacf=None,
                 mu=1.0e-03, eps1=1.5e-08, eps2=1.5e-08, eps3=1.5e-08,
                 maxit=1000, cdiff=False):
    """
    Parameters
    ----------
    func: callable
        Function or method computing the model function.
    p0: array_like, shape (m,)
        Initial estimate of the parameters.
    y: array_like, shape (n,)
        Dependent data, or the observation.
    bc: sequence of 2-tuples
        `(min, max)` pairs for each element of the parameters, specifying the
        (inclusive) upper and lower bounds.  Use None for one of `min` or `max`
        for specifying no bound in that direction.
    lec: 2-tuple of ndarray
        `(A, b)` pair specifying a linear equation constraint, where `A` and `b`
        are a matrix of shape (k1, m) and a vector of shape (k1,) respectively.
    lic: 2-tuple of ndarray
        `(C, d)` pair specifying a linear inequality constraint, where `C` and
        `d` are a matrix of shape (k2, m) and a vector of shape (k2,)
        respectively.
    args: tuple, optional
        Extra arguments passed to `func` (and `jacf`).
    jacf: callable, optional
        Function or method computing the Jacobian of `func`.  If it is None, the
        Jacobian will be approximated.
    mu: float, optional
        Scale factor for initial mu.
    eps1: float, optional
        Stopping threshold for ||J^T e||_inf.
    eps2: float, optional
        Stopping threshold for ||Dp||_2.
    eps3: float, optional
        Stopping threshold for ||e||_2.
    maxit: int, optional
        The maximum number of iterations.
    cdiff: {True, False}, optional
        If this is True, the Jacobian is approximated with central
        differentiation.

    Returns
    -------
    p: ndarray, shape=(m,)
        Best-fit parameters.
    pcov: ndarray, shape=(m,m)
        Covariance of the best-fit parameters.
    info: tuple
        Information regarding minimization.
            0: ||e||_2 at `p0`
            1:
                0: 2-norm of e
                1: infinity-norm of J^T.e
                2: 2-norm of Dp
                3: mu / max{(J^T.J)_ii}
            2: The number of iterations
            3: The reason for termination
            4: The number of `func` evaluations
            5: The number of `jacf` evaluations
            6: The number of the linear system solved
    """
    return _levmar.levmar_bleic(func, p0, y, bc, lec, lic, args, jacf,
                                mu, eps1, eps2, eps3, maxit, cdiff)


def levmar_blic(func, p0, y, bc, lic, args=(), jacf=None,
                mu=1.0e-03, eps1=1.5e-08, eps2=1.5e-08, eps3=1.5e-08,
                maxit=1000, cdiff=False):
    """
    Parameters
    ----------
    func: callable
        Function or method computing the model function.
    p0: array_like, shape (m,)
        Initial estimate of the parameters.
    y: array_like, shape (n,)
        Dependent data, or the observation.
    bc: sequence of 2-tuples
        `(min, max)` pairs for each element of the parameters, specifying the
        (inclusive) upper and lower bounds.  Use None for one of `min` or `max`
        for specifying no bound in that direction.
    lic: 2-tuple of ndarray
        `(C, d)` pair specifying a linear inequality constraint, where `C` and
        `d` are a matrix of shape (k2, m) and a vector of shape (k2,)
        respectively.
    args: tuple, optional
        Extra arguments passed to `func` (and `jacf`).
    jacf: callable, optional
        Function or method computing the Jacobian of `func`.  If it is None, the
        Jacobian will be approximated.
    mu: float, optional
        Scale factor for initial mu.
    eps1: float, optional
        Stopping threshold for ||J^T e||_inf.
    eps2: float, optional
        Stopping threshold for ||Dp||_2.
    eps3: float, optional
        Stopping threshold for ||e||_2.
    maxit: int, optional
        The maximum number of iterations.
    cdiff: {True, False}, optional
        If this is True, the Jacobian is approximated with central
        differentiation.

    Returns
    -------
    p: ndarray, shape=(m,)
        Best-fit parameters.
    pcov: ndarray, shape=(m,m)
        Covariance of the best-fit parameters.
    info: tuple
        Information regarding minimization.
            0: ||e||_2 at `p0`
            1:
                0: 2-norm of e
                1: infinity-norm of J^T.e
                2: 2-norm of Dp
                3: mu / max{(J^T.J)_ii}
            2: The number of iterations
            3: The reason for termination
            4: The number of `func` evaluations
            5: The number of `jacf` evaluations
            6: The number of the linear system solved
    """
    return _levmar.levmar_blic(func, p0, y, bc, lic, args, jacf,
                               mu, eps1, eps2, eps3, maxit, cdiff)


def levmar_leic(func, p0, y, lec, lic, args=(), jacf=None,
                mu=1.0e-03, eps1=1.5e-08, eps2=1.5e-08, eps3=1.5e-08,
                maxit=1000, cdiff=False):
    """
    Parameters
    ----------
    func: callable
        Function or method computing the model function.
    p0: array_like, shape (m,)
        Initial estimate of the parameters.
    y: array_like, shape (n,)
        Dependent data, or the observation.
    lec: 2-tuple of ndarray
        `(A, b)` pair specifying a linear equation constraint, where `A` and `b`
        are a matrix of shape (k1, m) and a vector of shape (k1,) respectively.
    lic: 2-tuple of ndarray
        `(C, d)` pair specifying a linear inequality constraint, where `C` and
        `d` are a matrix of shape (k2, m) and a vector of shape (k2,)
        respectively.
    args: tuple, optional
        Extra arguments passed to `func` (and `jacf`).
    jacf: callable, optional
        Function or method computing the Jacobian of `func`.  If it is None, the
        Jacobian will be approximated.
    mu: float, optional
        Scale factor for initial mu.
    eps1: float, optional
        Stopping threshold for ||J^T e||_inf.
    eps2: float, optional
        Stopping threshold for ||Dp||_2.
    eps3: float, optional
        Stopping threshold for ||e||_2.
    maxit: int, optional
        The maximum number of iterations.
    cdiff: {True, False}, optional
        If this is True, the Jacobian is approximated with central
        differentiation.

    Returns
    -------
    p: ndarray, shape=(m,)
        Best-fit parameters.
    pcov: ndarray, shape=(m,m)
        Covariance of the best-fit parameters.
    info: tuple
        Information regarding minimization.
            0: ||e||_2 at `p0`
            1:
                0: 2-norm of e
                1: infinity-norm of J^T.e
                2: 2-norm of Dp
                3: mu / max{(J^T.J)_ii}
            2: The number of iterations
            3: The reason for termination
            4: The number of `func` evaluations
            5: The number of `jacf` evaluations
            6: The number of the linear system solved
    """
    return _levmar.levmar_leic(func, p0, y, lec, lic, args, jacf,
                               mu, eps1, eps2, eps3, maxit, cdiff)


def levmar_lic(func, p0, y, lic, args=(), jacf=None,
               mu=1.0e-03, eps1=1.5e-08, eps2=1.5e-08, eps3=1.5e-08,
               maxit=1000, cdiff=False):
    """
    Parameters
    ----------
    func: callable
        Function or method computing the model function.
    p0: array_like, shape (m,)
        Initial estimate of the parameters.
    y: array_like, shape (n,)
        Dependent data, or the observation.
    lic: 2-tuple of ndarray
        `(C, d)` pair specifying a linear inequality constraint, where `C` and
        `d` are a matrix of shape (k2, m) and a vector of shape (k2,)
        respectively.
    args: tuple, optional
        Extra arguments passed to `func` (and `jacf`).
    jacf: callable, optional
        Function or method computing the Jacobian of `func`.  If it is None, the
        Jacobian will be approximated.
    mu: float, optional
        Scale factor for initial mu.
    eps1: float, optional
        Stopping threshold for ||J^T e||_inf.
    eps2: float, optional
        Stopping threshold for ||Dp||_2.
    eps3: float, optional
        Stopping threshold for ||e||_2.
    maxit: int, optional
        The maximum number of iterations.
    cdiff: {True, False}, optional
        If this is True, the Jacobian is approximated with central
        differentiation.

    Returns
    -------
    p: ndarray, shape=(m,)
        Best-fit parameters.
    pcov: ndarray, shape=(m,m)
        Covariance of the best-fit parameters.
    info: tuple
        Information regarding minimization.
            0: ||e||_2 at `p0`
            1:
                0: 2-norm of e
                1: infinity-norm of J^T.e
                2: 2-norm of Dp
                3: mu / max{(J^T.J)_ii}
            2: The number of iterations
            3: The reason for termination
            4: The number of `func` evaluations
            5: The number of `jacf` evaluations
            6: The number of the linear system solved
    """
    return _levmar.levmar_lic(func, p0, y, lic, args, jacf,
                              mu, eps1, eps2, eps3, maxit, cdiff)
