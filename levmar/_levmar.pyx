# -*- coding: utf-8 -*-
from __future__ import division

cimport cython
from numpy cimport *

import numpy as np
import warnings


cdef extern from "stdlib.h":
    void *memcpy(void *dest, void *src, size_t n)


cdef extern from "float.h":
    double DBL_MAX
    double DBL_EPSILON


cdef extern from "Python.h":
    object PyObject_CallObject(object obj, object args)
    object PySequence_Concat(object obj1, object obj2)
    PyObject* PyErr_NoMemory() except NULL


cdef extern from "numpy/npy_math.h":
    double NPY_NAN
    double NPY_INFINITY


import_array()


_LM_STOP_REASONS = {
    1: "Stopped by small gradient J^T e",
    2: "Stop by small Dp",
    3: "Stop by `maxit`",
    4: "Singular matrix.  Restart from current `p` with increased mu",
    5: "No further error reduction is possible. Restart with increased mu",
    6: "Stopped by small ||e||_2",
}

_LM_STOP_REASONS_WARNED = (3, 4, 5)


cdef class _LMFunc:
    """
    Parameters
    ----------
    func : callable
        Function or method taking, at least, one length of m vector
        and returning a length of n vector.
    args : tuple
        Extra arguments passed to `func` (and `jacf`).
    jacf : callable, optional
        Function or method computing the Jacobian of `func`.  It
        takes, at least, one length of m vector and returns the (nxm)
        Jacobian matrix or a campatible C-contiguous vector.  If it is
        None, the Jacobian will be approximated.

    Attributes
    ----------
    func : callable
    args : tuple
    jacf : callable or None
    """
    cdef:
        object func
        object jacf
        object args

    def __init__(self, func, args, jacf):
        self.func = func
        self.args = args
        self.jacf = jacf

    cdef void eval_func(self, double *p, double *y, int m, int n):
        cdef npy_intp m_ = m
        cdef ndarray p_ = PyArray_SimpleNewFromData(1, &m_, NPY_DOUBLE, <void*>p)
        cdef ndarray y_ = PyObject_CallObject(self.func, PySequence_Concat((p_,), self.args))
        memcpy(y, y_.data, n * sizeof(double))

    cdef void eval_jacf(self, double *p, double *j, int m, int n):
        cdef npy_intp m_ = m
        cdef ndarray p_ = PyArray_SimpleNewFromData(1, &m_, NPY_DOUBLE, <void*>p)
        cdef ndarray y_ = PyObject_CallObject(self.jacf, PySequence_Concat((p_,), self.args))
        memcpy(j, y_.data, m * n * sizeof(double))


cdef inline void callback_func(double *p, double *y, int m, int n, void *ctx):
    (<_LMFunc>ctx).eval_func(p, y, m, n)


cdef inline void callback_jacf(double *p, double *j, int m, int n, void *ctx):
    (<_LMFunc>ctx).eval_jacf(p, j, m, n)


cdef int set_opts(double mu, double eps1, double eps2, double eps3, bint cdiff,
                  double *opts) except -1:
    opts[0] = mu
    if not 0 < eps1 < 1:
        raise ValueError('eps1 must be less than 1')
    opts[1] = eps1
    if not 0 < eps2 < 1:
        raise ValueError('eps2 must be less than 1')
    opts[2] = eps2
    if not 0 < eps3 < 1:
        raise ValueError('eps3 must be less than 1')
    opts[3] = eps3
    opts[4] = -LM_DIFF_DELTA if cdiff else LM_DIFF_DELTA


cdef object prepare_bc(bc, m):
    if not isinstance(bc, (list, tuple)):
        raise TypeError('bc must be a tuple/list')
    if len(bc) != m:
        raise ValueError('bounds must be the length of {0} ({1} is given)'
                         .format(m, len(bc)))
    # The levmar library returns LM_ERROR when lb[i] > ub[i].  Hence, we do not
    # check if lb[i] < ub[i].
    lb = np.zeros(m, dtype=np.float64)
    ub = np.zeros(m, dtype=np.float64)
    for i, b in enumerate(bc):
        if isinstance(b, tuple):
            if len(b) == 2:
                if b[0] in (None, NPY_NAN, -NPY_INFINITY):
                    lb[i] = -DBL_MAX
                else:
                    lb[i] = float(b[0])
                if b[1] in (None, NPY_NAN, NPY_INFINITY):
                    ub[i] = DBL_MAX
                else:
                    ub[i] = float(b[1])
            else:
                raise ValueError('Element of bc must be a 2-tuple of floats/Nones')
        else:
            raise ValueError('Element of bc must be a 2-tuple of floats/Nones')

    return lb, ub


cdef object prepare_lc(lc, m):
    if not isinstance(lc, (list, tuple)):
        raise TypeError('lc must be a tuple/list')
    if len(lc) != 2:
        raise TypeError('lc must be the length 2 ({0} given)'
                        .format(len(lc)))

    mat = np.atleast_2d(np.ascontiguousarray(lc[0], dtype=np.float64))
    vec = np.ascontiguousarray(lc[1], dtype=np.float64)

    n_rows = m
    n_cols = mat.size // m

    if mat.size % m != 0:
        raise ValueError('The size of a linear constraint matrix '
                         'must be consistent with (kx{0})'.format(m))
    if len(vec) != n_cols:
        raise ValueError('The size of a linear constraint vector '
                         'must be {0} ({1} given)'.format(n_cols, len(vec)))
    return mat, vec


cdef object return_result(p, pcov, double *c_info):
    info = (
        c_info[0],       # ||e||_2 at `p0`
        (
            c_info[1],   # ||e||_2 at `p`
            c_info[2],   # ||J^T.e||_inf
            c_info[3],   # ||Dp||_2
            c_info[4],   # mu / max[J^T.J]_ii
        ),
        <int>c_info[5],  # number of iterations
        _LM_STOP_REASONS[<int>c_info[6]],  # reason for terminating
        <int>c_info[7],  # number of `func` evaluations
        <int>c_info[8],  # number of `jacf` evaluations
        <int>c_info[9],  # number of linear system solved
    )

    if int(c_info[6]) in _LM_STOP_REASONS_WARNED:
        # Issue warning for unsuccessful termination.
        warnings.warn(_LM_STOP_REASONS[info[6]], UserWarning)
    return p, pcov, info


def levmar(func, p, x, args, jacf,
           double mu, double eps1, double eps2, double eps3,
           int maxit, bint cdiff):
    cdef:
        ndarray p_, x_, pcov
        int m, n, niter
        double c_opts[LM_OPTS_SZ], c_info[LM_INFO_SZ]
        _LMFunc lm_func

    p_ = PyArray_FROMANY(p, NPY_DOUBLE, 1, 1, NPY_ENSURECOPY)
    x_ = PyArray_ContiguousFromAny(x, NPY_DOUBLE, 1, 1)
    if p_.ndim != 1:
        raise ValueError('p must be a 1d array-like')
    if x_.ndim != 1:
        raise ValueError('x must be a 1d array-like')
    m = p_.shape[0]
    n = x_.shape[0]
    pcov = PyArray_ZEROS(2, [m, m], NPY_DOUBLE, 0)

    set_opts(mu, eps1, eps2, eps3, cdiff, c_opts)
    if maxit <= 0:
        raise ValueError('maxit must be a positive int')

    if not callable(func):
        raise TypeError('func must be callable')
    ret = func(*((p,) + args))
    if ret.ndim != 1 or ret.shape[0] != n:
        raise RuntimeError(
            'func must return a 1d array-like of length {0}'.format(n))

    if jacf is not None:
        if not callable(jacf):
            raise TypeError('jacf must be callable')
        ret = jacf(*((p,) + args))
        if ret.ndim != 2 or ret.shape != (n, m):
            raise RuntimeError(
                'jacf must return a 2d array-like of shape ({0}, {1})'
                .format(n, m))

    lm_func = _LMFunc(func, args, jacf)

    niter = 0
    if jacf is not None:
        niter = dlevmar_der(
            callback_func, callback_jacf,
            <double*>p_.data, <double*>x_.data, m, n,
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)
    else:
        niter = dlevmar_dif(
            callback_func,
            <double*>p_.data, <double*>x_.data, m, n,
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)

    if niter == LM_ERROR:
        if c_info[6] == 7:
            raise RuntimeError(
                'Stopped by invalid values (NaN or Inf) returned by func')
        else:
            raise RuntimeError

    return return_result(p_, pcov, c_info)


def levmar_bc(func, p, x, bc, args, jacf,
              double mu, double eps1, double eps2, double eps3,
              int maxit, bint cdiff):
    cdef:
        ndarray p_, x_, lb, ub, pcov
        int m, n, niter
        double c_opts[LM_OPTS_SZ], c_info[LM_INFO_SZ]
        _LMFunc lm_func

    p_ = PyArray_FROMANY(p, NPY_DOUBLE, 1, 1, NPY_ENSURECOPY)
    x_ = PyArray_ContiguousFromAny(x, NPY_DOUBLE, 1, 1)
    if p_.ndim != 1:
        raise ValueError('p must be a 1d array-like')
    if x_.ndim != 1:
        raise ValueError('x must be a 1d array-like')
    m = p_.shape[0]
    n = x_.shape[0]
    pcov = PyArray_ZEROS(2, [m, m], NPY_DOUBLE, 0)

    lb, ub = prepare_bc(bc, m)

    set_opts(mu, eps1, eps2, eps3, cdiff, c_opts)
    if maxit <= 0:
        raise ValueError('maxit must be a positive int')

    if not callable(func):
        raise TypeError('func must be callable')
    ret = func(*((p,) + args))
    if ret.ndim != 1 or ret.shape[0] != n:
        raise RuntimeError(
            'func must return a 1d array-like of length {0}'.format(n))

    if jacf is not None:
        if not callable(jacf):
            raise TypeError('jacf must be callable')
        ret = jacf(*((p,) + args))
        if ret.ndim != 2 or ret.shape != (n, m):
            raise RuntimeError(
                'jacf must return a 2d array-like of shape ({0}, {1})'
                .format(n, m))

    lm_func = _LMFunc(func, args, jacf)

    niter = 0
    if jacf is not None:
        niter = dlevmar_bc_der(
            callback_func, callback_jacf,
            <double*>p_.data, <double*>x_.data, m, n,
            <double*>lb.data, <double*>ub.data, NULL,
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)
    else:
        niter = dlevmar_bc_dif(
            callback_func,
            <double*>p_.data, <double*>x_.data, m, n,
            <double*>lb.data, <double*>ub.data, NULL,
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)

    if niter == LM_ERROR:
        if c_info[6] == 7:
            raise RuntimeError(
                'Stopped by invalid values (NaN or Inf) returned by func')
        else:
            raise RuntimeError

    return return_result(p_, pcov, c_info)


def levmar_lec(func, p, x, lec, args, jacf,
               double mu, double eps1, double eps2, double eps3,
               int maxit, bint cdiff):
    cdef:
        ndarray p_, x_, A, b, pcov
        int m, n, niter
        double c_opts[LM_OPTS_SZ], c_info[LM_INFO_SZ]
        _LMFunc lm_func

    p_ = PyArray_FROMANY(p, NPY_DOUBLE, 1, 1, NPY_ENSURECOPY)
    x_ = PyArray_ContiguousFromAny(x, NPY_DOUBLE, 1, 1)
    if p_.ndim != 1:
        raise ValueError('p must be a 1d array-like')
    if x_.ndim != 1:
        raise ValueError('x must be a 1d array-like')
    m = p_.shape[0]
    n = x_.shape[0]
    pcov = PyArray_ZEROS(2, [m, m], NPY_DOUBLE, 0)

    A, b = prepare_lc(lec, m)

    set_opts(mu, eps1, eps2, eps3, cdiff, c_opts)
    if maxit <= 0:
        raise ValueError('maxit must be a positive int')

    if not callable(func):
        raise TypeError('func must be callable')
    ret = func(*((p,) + args))
    if ret.ndim != 1 or ret.shape[0] != n:
        raise RuntimeError(
            'func must return a 1d array-like of length {0}'.format(n))

    if jacf is not None:
        if not callable(jacf):
            raise TypeError('jacf must be callable')
        ret = jacf(*((p,) + args))
        if ret.ndim != 2 or ret.shape != (n, m):
            raise RuntimeError(
                'jacf must return a 2d array-like of shape ({0}, {1})'
                .format(n, m))

    lm_func = _LMFunc(func, args, jacf)

    niter = 0
    if jacf is not None:
        niter = dlevmar_lec_der(
            callback_func, callback_jacf,
            <double*>p_.data, <double*>x_.data, m, n,
            <double*>A.data, <double*>b.data, A.shape[0],
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)
    else:
        niter = dlevmar_lec_dif(
            callback_func,
            <double*>p_.data, <double*>x_.data, m, n,
            <double*>A.data, <double*>b.data, A.shape[0],
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)

    if niter == LM_ERROR:
        if c_info[6] == 7:
            raise RuntimeError(
                'Stopped by invalid values (NaN or Inf) returned by func')
        else:
            raise RuntimeError

    return return_result(p_, pcov, c_info)


def levmar_blec(func, p, x, bc, lec, args, jacf,
                double mu, double eps1, double eps2, double eps3,
                int maxit, bint cdiff):
    cdef:
        ndarray p_, x_, lb, ub, A, b, pcov
        int m, n, niter
        double c_opts[LM_OPTS_SZ], c_info[LM_INFO_SZ]
        _LMFunc lm_func

    p_ = PyArray_FROMANY(p, NPY_DOUBLE, 1, 1, NPY_ENSURECOPY)
    x_ = PyArray_ContiguousFromAny(x, NPY_DOUBLE, 1, 1)
    if p_.ndim != 1:
        raise ValueError('p must be a 1d array-like')
    if x_.ndim != 1:
        raise ValueError('x must be a 1d array-like')
    m = p_.shape[0]
    n = x_.shape[0]
    pcov = PyArray_ZEROS(2, [m, m], NPY_DOUBLE, 0)

    lb, ub = prepare_bc(bc, m)
    A, b = prepare_lc(lec, m)

    set_opts(mu, eps1, eps2, eps3, cdiff, c_opts)
    if maxit <= 0:
        raise ValueError('maxit must be a positive int')

    if not callable(func):
        raise TypeError('func must be callable')
    ret = func(*((p,) + args))
    if ret.ndim != 1 or ret.shape[0] != n:
        raise RuntimeError(
            'func must return a 1d array-like of length {0}'.format(n))

    if jacf is not None:
        if not callable(jacf):
            raise TypeError('jacf must be callable')
        ret = jacf(*((p,) + args))
        if ret.ndim != 2 or ret.shape != (n, m):
            raise RuntimeError(
                'jacf must return a 2d array-like of shape ({0}, {1})'
                .format(n, m))

    lm_func = _LMFunc(func, args, jacf)

    niter = 0
    if jacf is not None:
        niter = dlevmar_blec_der(
            callback_func, callback_jacf,
            <double*>p_.data, <double*>x_.data, m, n,
            <double*>lb.data, <double*>ub.data,
            <double*>A.data, <double*>b.data, A.shape[0], NULL,
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)
    else:
        niter = dlevmar_blec_dif(
            callback_func,
            <double*>p_.data, <double*>x_.data, m, n,
            <double*>lb.data, <double*>ub.data,
            <double*>A.data, <double*>b.data, A.shape[0], NULL,
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)

    if niter == LM_ERROR:
        if c_info[6] == 7:
            raise RuntimeError(
                'Stopped by invalid values (NaN or Inf) returned by func')
        else:
            raise RuntimeError

    return return_result(p_, pcov, c_info)


def levmar_bleic(func, p, x, bc, lec, lic, args, jacf,
                 double mu, double eps1, double eps2, double eps3,
                 int maxit, bint cdiff):
    cdef:
        ndarray p_, x_, lb, ub, A, b, C, d, pcov
        int m, n, niter
        double c_opts[LM_OPTS_SZ], c_info[LM_INFO_SZ]
        _LMFunc lm_func

    p_ = PyArray_FROMANY(p, NPY_DOUBLE, 1, 1, NPY_ENSURECOPY)
    x_ = PyArray_ContiguousFromAny(x, NPY_DOUBLE, 1, 1)
    if p_.ndim != 1:
        raise ValueError('p must be a 1d array-like')
    if x_.ndim != 1:
        raise ValueError('x must be a 1d array-like')
    m = p_.shape[0]
    n = x_.shape[0]
    pcov = PyArray_ZEROS(2, [m, m], NPY_DOUBLE, 0)

    lb, ub = prepare_bc(bc, m)
    A, b = prepare_lc(lec, m)
    C, d = prepare_lc(lic, m)

    set_opts(mu, eps1, eps2, eps3, cdiff, c_opts)
    if maxit <= 0:
        raise ValueError('maxit must be a positive int')

    if not callable(func):
        raise TypeError('func must be callable')
    ret = func(*((p,) + args))
    if ret.ndim != 1 or ret.shape[0] != n:
        raise RuntimeError(
            'func must return a 1d array-like of length {0}'.format(n))

    if jacf is not None:
        if not callable(jacf):
            raise TypeError('jacf must be callable')
        ret = jacf(*((p,) + args))
        if ret.ndim != 2 or ret.shape != (n, m):
            raise RuntimeError(
                'jacf must return a 2d array-like of shape ({0}, {1})'
                .format(n, m))

    lm_func = _LMFunc(func, args, jacf)

    niter = 0
    if jacf is not None:
        niter = dlevmar_bleic_der(
            callback_func, callback_jacf,
            <double*>p_.data, <double*>x_.data, m, n,
            <double*>lb.data, <double*>ub.data,
            <double*>A.data, <double*>b.data, A.shape[0],
            <double*>C.data, <double*>d.data, C.shape[0],
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)
    else:
        niter = dlevmar_bleic_dif(
            callback_func,
            <double*>p_.data, <double*>x_.data, m, n,
            <double*>lb.data, <double*>ub.data,
            <double*>A.data, <double*>b.data, A.shape[0],
            <double*>C.data, <double*>d.data, C.shape[0],
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)

    if niter == LM_ERROR:
        if c_info[6] == 7:
            raise RuntimeError(
                'Stopped by invalid values (NaN or Inf) returned by func')
        else:
            raise RuntimeError

    return return_result(p_, pcov, c_info)


def levmar_blic(func, p, x, bc, lic, args, jacf,
               double mu, double eps1, double eps2, double eps3,
               int maxit, bint cdiff):
    cdef:
        ndarray p_, x_, lb, ub, C, d, pcov
        int m, n, niter
        double c_opts[LM_OPTS_SZ], c_info[LM_INFO_SZ]
        _LMFunc lm_func

    p_ = PyArray_FROMANY(p, NPY_DOUBLE, 1, 1, NPY_ENSURECOPY)
    x_ = PyArray_ContiguousFromAny(x, NPY_DOUBLE, 1, 1)
    if p_.ndim != 1:
        raise ValueError('p must be a 1d array-like')
    if x_.ndim != 1:
        raise ValueError('x must be a 1d array-like')
    m = p_.shape[0]
    n = x_.shape[0]
    pcov = PyArray_ZEROS(2, [m, m], NPY_DOUBLE, 0)

    lb, ub = prepare_bc(bc, m)
    C, d = prepare_lc(lic, m)

    set_opts(mu, eps1, eps2, eps3, cdiff, c_opts)
    if maxit <= 0:
        raise ValueError('maxit must be a positive int')

    if not callable(func):
        raise TypeError('func must be callable')
    ret = func(*((p,) + args))
    if ret.ndim != 1 or ret.shape[0] != n:
        raise RuntimeError(
            'func must return a 1d array-like of length {0}'.format(n))

    if jacf is not None:
        if not callable(jacf):
            raise TypeError('jacf must be callable')
        ret = jacf(*((p,) + args))
        if ret.ndim != 2 or ret.shape != (n, m):
            raise RuntimeError(
                'jacf must return a 2d array-like of shape ({0}, {1})'
                .format(n, m))

    lm_func = _LMFunc(func, args, jacf)

    niter = 0
    if jacf is not None:
        niter = dlevmar_blic_der(
            callback_func, callback_jacf,
            <double*>p_.data, <double*>x_.data, m, n,
            <double*>lb.data, <double*>ub.data,
            <double*>C.data, <double*>d.data, C.shape[0],
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)
    else:
        niter = dlevmar_blic_dif(
            callback_func,
            <double*>p_.data, <double*>x_.data, m, n,
            <double*>lb.data, <double*>ub.data,
            <double*>C.data, <double*>d.data, C.shape[0],
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)

    if niter == LM_ERROR:
        if c_info[6] == 7:
            raise RuntimeError(
                'Stopped by invalid values (NaN or Inf) returned by func')
        else:
            raise RuntimeError

    return return_result(p_, pcov, c_info)


def levmar_leic(func, p, x, lec, lic, args, jacf,
                double mu, double eps1, double eps2, double eps3,
                int maxit, bint cdiff):
    cdef:
        ndarray p_, x_, A, b, C, d, pcov
        int m, n, niter
        double c_opts[LM_OPTS_SZ], c_info[LM_INFO_SZ]
        _LMFunc lm_func

    p_ = PyArray_FROMANY(p, NPY_DOUBLE, 1, 1, NPY_ENSURECOPY)
    x_ = PyArray_ContiguousFromAny(x, NPY_DOUBLE, 1, 1)
    if p_.ndim != 1:
        raise ValueError('p must be a 1d array-like')
    if x_.ndim != 1:
        raise ValueError('x must be a 1d array-like')
    m = p_.shape[0]
    n = x_.shape[0]
    pcov = PyArray_ZEROS(2, [m, m], NPY_DOUBLE, 0)

    A, b = prepare_lc(lec, m)
    C, d = prepare_lc(lic, m)

    set_opts(mu, eps1, eps2, eps3, cdiff, c_opts)
    if maxit <= 0:
        raise ValueError('maxit must be a positive int')

    if not callable(func):
        raise TypeError('func must be callable')
    ret = func(*((p,) + args))
    if ret.ndim != 1 or ret.shape[0] != n:
        raise RuntimeError(
            'func must return a 1d array-like of length {0}'.format(n))

    if jacf is not None:
        if not callable(jacf):
            raise TypeError('jacf must be callable')
        ret = jacf(*((p,) + args))
        if ret.ndim != 2 or ret.shape != (n, m):
            raise RuntimeError(
                'jacf must return a 2d array-like of shape ({0}, {1})'
                .format(n, m))

    lm_func = _LMFunc(func, args, jacf)

    niter = 0
    if jacf is not None:
        niter = dlevmar_leic_der(
            callback_func, callback_jacf,
            <double*>p_.data, <double*>x_.data, m, n,
            <double*>A.data, <double*>b.data, A.shape[0],
            <double*>C.data, <double*>d.data, C.shape[0],
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)
    else:
        niter = dlevmar_leic_dif(
            callback_func,
            <double*>p_.data, <double*>x_.data, m, n,
            <double*>A.data, <double*>b.data, A.shape[0],
            <double*>C.data, <double*>d.data, C.shape[0],
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)

    if niter == LM_ERROR:
        if c_info[6] == 7:
            raise RuntimeError(
                'Stopped by invalid values (NaN or Inf) returned by func')
        else:
            raise RuntimeError

    return return_result(p_, pcov, c_info)


def levmar_lic(func, p, x, lic, args, jacf,
               double mu, double eps1, double eps2, double eps3,
               int maxit, bint cdiff):
    cdef:
        ndarray p_, x_, C, d, pcov
        int m, n, niter
        double c_opts[LM_OPTS_SZ], c_info[LM_INFO_SZ]
        _LMFunc lm_func

    p_ = PyArray_FROMANY(p, NPY_DOUBLE, 1, 1, NPY_ENSURECOPY)
    x_ = PyArray_ContiguousFromAny(x, NPY_DOUBLE, 1, 1)
    if p_.ndim != 1:
        raise ValueError('p must be a 1d array-like')
    if x_.ndim != 1:
        raise ValueError('x must be a 1d array-like')
    m = p_.shape[0]
    n = x_.shape[0]
    pcov = PyArray_ZEROS(2, [m, m], NPY_DOUBLE, 0)

    C, d = prepare_lc(lic, m)

    set_opts(mu, eps1, eps2, eps3, cdiff, c_opts)
    if maxit <= 0:
        raise ValueError('maxit must be a positive int')

    if not callable(func):
        raise TypeError('func must be callable')
    ret = func(*((p,) + args))
    if ret.ndim != 1 or ret.shape[0] != n:
        raise RuntimeError(
            'func must return a 1d array-like of length {0}'.format(n))

    if jacf is not None:
        if not callable(jacf):
            raise TypeError('jacf must be callable')
        ret = jacf(*((p,) + args))
        if ret.ndim != 2 or ret.shape != (n, m):
            raise RuntimeError(
                'jacf must return a 2d array-like of shape ({0}, {1})'
                .format(n, m))

    lm_func = _LMFunc(func, args, jacf)

    niter = 0
    if jacf is not None:
        niter = dlevmar_lic_der(
            callback_func, callback_jacf,
            <double*>p_.data, <double*>x_.data, m, n,
            <double*>C.data, <double*>d.data, C.shape[0],
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)
    else:
        niter = dlevmar_lic_dif(
            callback_func,
            <double*>p_.data, <double*>x_.data, m, n,
            <double*>C.data, <double*>d.data, C.shape[0],
            maxit, c_opts, c_info, NULL, <double*>pcov.data, <void*>lm_func)

    if niter == LM_ERROR:
        if c_info[6] == 7:
            raise RuntimeError(
                'Stopped by invalid values (NaN or Inf) returned by func')
        else:
            raise RuntimeError

    return return_result(p_, pcov, c_info)
