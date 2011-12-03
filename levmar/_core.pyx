# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
"""
TODO:
    * Implement a weighted least square method.
"""
from __future__ import division
cimport cython
from numpy cimport *
import warnings


cdef extern from "stdlib.h":
    void free(void *ptr)
    void *malloc(size_t size)
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


ctypedef float64_t dtype_t


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

# The stopping threshold for ||J^T e||_inf
_LM_EPS1 = DBL_EPSILON**(1/2)
# The stopping threshold for ||Dp||_2
_LM_EPS2 = DBL_EPSILON**(1/2)
# The stopping threshold for ||e||_2
_LM_EPS3 = DBL_EPSILON**(1/2)


class LMError(Exception):
    pass

class LMRuntimeError(LMError, RuntimeError):
    pass

class LMUserFuncError(LMError):
    pass

class LMWarning(UserWarning):
    pass


cdef class _LMFunction:
    """
    Parameters
    ----------
    func : callable
        A function or method taking, at least, one length of m vector
        and returning a length of n vector.
    args : tuple
        Extra arguments passed to `func` (and `jacf`).
    jacf : callable, optional
        A function or method computing the Jacobian of `func`.  It
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

    def __init__(self, func, args, jacf=None):
        self.func = func
        self.args = args
        self.jacf = jacf

    cdef void eval_func(self, double *p, double *y, int m, int n):
        cdef:
            npy_intp m_ = m
            object py_p = PyArray_SimpleNewFromData(1, &m_, NPY_DOUBLE, <void*>p)
            ndarray py_y

        args = PySequence_Concat((py_p,), self.args)
        py_y = PyObject_CallObject(self.func, args)
        # Copy the result to `x`
        memcpy(y, py_y.data, n*sizeof(double))

    cdef void eval_jacf(self, double *p, double *jacf, int m, int n):
        cdef:
            npy_intp m_ = m
            object py_p = PyArray_SimpleNewFromData(1, &m_, NPY_DOUBLE, <void*>p)
            ndarray py_jacf

        args = PySequence_Concat((py_p,), self.args)
        py_jacf = PyObject_CallObject(self.jacf, args)
        # Copy the result to `jacf`
        memcpy(jacf, py_jacf.data, m*n*sizeof(double))

    cdef int _check_funcs(self, ndarray p, int m, int n) except -1:
        cdef Py_ssize_t size
        args = (p,) + self.args
        try:
            ret = self.func(*args)
            size = PyArray_SIZE(ret)
        except Exception, e:
            raise LMUserFuncError(e)
        if size != n:
            msg = ("`func` returned a invalid size vector: "
                   "{0} expected but {1} returned".format(n, size))
            raise LMUserFuncError(msg)
        if self.jacf is not None:
            try:
                ret = self.jacf(*args)
                size = PyArray_SIZE(ret)
            except Exception, e:
                raise LMUserFuncError(e)
            if size != m*n:
                msg = ("`jacf` returned a invalid size vector: "
                       "{0} expected but {1} returned".format(m*n, size))
                raise LMUserFuncError(msg)
            if not self._check_jacf(<double*>p.data, m, n):
                msg = "`jacf` may not be correct"
                warnings.warn(msg, LMWarning)
        return 1

    cdef bint _check_jacf(self, double *p, int m, int n):
        cdef:
            bint is_jacf_correct = True
            double *err = NULL

        err = <double*>malloc(n*sizeof(double))
        if err == NULL:
            PyErr_NoMemory()
        dlevmar_chkjac(callback_func, callback_jacf, p, m, n, <void*>self, err)

        cdef int i
        for i in range(n):
            if err[i] < 0.5:
                is_jacf_correct = False
                break
        free(err)
        return is_jacf_correct


cdef inline void callback_func(double *p, double *y, int m, int n, void *ctx):
    (<_LMFunction>ctx).eval_func(p, y, m, n)


cdef inline void callback_jacf(double *p, double *jacf, int m, int n, void *ctx):
    (<_LMFunction>ctx).eval_jacf(p, jacf, m, n)


cdef class _LMWorkSpace:
    """A class managing the working array used internally in the levmar
    libraray."""
    cdef:
        double* ptr
        int prev_size

    cdef double* allocate(self, size_t size):
        if size != self.prev_size:
            if self.ptr != NULL:
                free(self.ptr)
            self.ptr = <double*>malloc(size*sizeof(double))
            if self.ptr == NULL:
                PyErr_NoMemory()
            self.prev_size = size
        return self.ptr

    def __dealloc__(self):
        if self.ptr != NULL:
            free(self.ptr)


# This global variable holds a working array used internally in the
# levmar library.  The necessarily size of the array depends on
# specification of problems.
cdef _LMWorkSpace _LMWork = _LMWorkSpace()


cdef class _LMConstraints:
    "A base class for the constraints"
    pass


cdef class _LMBoxConstraints(_LMConstraints):
    """Box constraints class

    Parameters
    ----------
    bounds : tuple/list, length m
        Box constraints.  Each constraint can be a tuple of two
        floats/Nones or None.  A tuple determines the (inclusive) lower
        and upper bound, and None means no constraint.  If one of two
        values in a tuple is None, then the bound is semi-definite.
    m : int
        The size of parameters

    Attributes
    ----------
    lb : double*, length m
        The (inclusive) lower bounds
    ub : double*, length m
        The (inclusive) upper bounds
    """
    cdef:
        double* lb
        double* ub

    def __init__(self, object bounds, int m):
        if not isinstance(bounds, (list, tuple)):
            raise TypeError("`bounds` must be a tuple/list")
        if len(bounds) != m:
            msg = ("`bounds` must be length of {0} ({1} is given)"
                   .format(m, len(bounds)))
            raise ValueError(msg)
        # Obviously, a lower bound cannot be larger than a upper bound.
        # Since the levmar library returns LM_ERROR when the requirement
        # is not satisfied, the requirement is not checked here.
        for i, b in enumerate(bounds):
            if isinstance(b, tuple):
                if len(b) == 2:
                    if b[0] in (None, NPY_NAN, -NPY_INFINITY):
                        self.lb[i] = -DBL_MAX
                    else:
                        self.lb[i] = float(b[0])
                    if b[1] in (None, NPY_NAN, NPY_INFINITY):
                        self.ub[i] = DBL_MAX
                    else:
                        self.ub[i] = float(b[1])
                else:
                    raise ValueError("A component of `bounds` must be a None "
                                     "or a tuple of 2 floats/Nones")
            elif b is None:
                self.lb[i] = -DBL_MAX
                self.ub[i] =  DBL_MAX
            else:
                raise ValueError("A component of `bounds` must be a None "
                                 "or a tuple of 2 floats/Nones")

    def __cinit__(self, object bounds, int m):
        self.lb = <double*>malloc(m*sizeof(double))
        if self.lb == NULL:
            PyErr_NoMemory()
        self.ub = <double*>malloc(m*sizeof(double))
        if self.ub == NULL:
            free(self.lb)
            PyErr_NoMemory()

    def __dealloc__(self):
        if self.lb != NULL:
            free(self.lb)
        if self.ub != NULL:
            free(self.ub)


cdef class _LMLinConstraints(_LMConstraints):
    cdef:
        double* mat
        double* vec
        int k

    def __cinit__(self, object mat, object vec, int m):
        cdef:
            ndarray A = PyArray_ContiguousFromAny(mat, NPY_DOUBLE, 1, 2)
            ndarray b = PyArray_ContiguousFromAny(vec, NPY_DOUBLE, 1, 1)
            Py_ssize_t size1 = PyArray_SIZE(A)
            Py_ssize_t size2 = PyArray_SIZE(b)
            Py_ssize_t k
        # In case of linear equation constraints, obviously a constraint
        # matrix cannot have more rows than columns for solving a problems.
        # Since the levmar library returns LM_ERROR when this requirement
        # is not satisfied, the requirement is not checked here.
        if A.ndim == 2:
            if A.shape[1] != m :
                raise ValueError("The shape of the constraint matrix must be "
                                 "(kx{0})".format(m))
        else:
            if size1 % m != 0:
                raise ValueError("The shape of the constraint matrix must be "
                                 "(kx{0})".format(m))
        # the number of equations/inequalities
        k = size1 // m

        if size2 != k:
            raise ValueError("The shape of the RH constraint vector must be "
                             "consistent with ({0}x1)".format(k))

        self.mat = <double*> malloc(m*k*sizeof(double))
        if self.mat == NULL:
            PyErr_NoMemory()
        memcpy(self.mat, A.data, m*k*sizeof(double))
        self.vec = <double*> malloc(k*sizeof(double))
        if self.vec == NULL:
            free(self.mat)
            PyErr_NoMemory()
        memcpy(self.vec, b.data, k*sizeof(double))
        self.k = k

    def __dealloc__(self):
        if self.mat != NULL:
            free(self.mat)
        if self.vec != NULL:
            free(self.vec)


cdef object py_info(double *c_info):
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
    return info


cdef inline int verify_funcs(_LMFunction func, ndarray p, int m, int n) except -1:
    func._check_funcs(p, m, n)
    return 1


cdef inline int set_iter_params(double mu, double eps1, double eps2, double eps3,
                                int maxit, bint cdif, double *opts) except -1:
    opts[0] = mu
    if eps1 >= 1: raise ValueError("`eps1` must be less than 1.")
    opts[1] = eps1
    if eps2 >= 1: raise ValueError("`eps2` must be less than 1.")
    opts[2] = eps2
    if eps2 >= 1: raise ValueError("`eps3` must be less than 1.")
    opts[3] = eps3
    opts[4] = LM_DIFF_DELTA if cdif else -LM_DIFF_DELTA
    if maxit <= 0:
        raise ValueError("`maxit` must be a positive.")


@cython.wraparound(False)
@cython.boundscheck(False)
def levmar(func, p0, y, args=(), jacf=None,
           bounds=None, A=None, b=None, C=None, d=None,
           double mu=1e-3, double eps1=_LM_EPS1, double eps2=_LM_EPS2,
           double eps3=_LM_EPS3, int maxit=1000, bint cdif=False):
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
    cdef:
        # Make a copy of `p0`
        ndarray p = PyArray_FROMANY(p0, NPY_DOUBLE, 1, 1, NPY_ENSURECOPY)
        ndarray x = PyArray_ContiguousFromAny(y, NPY_DOUBLE, 1, 1)

        int n = x.shape[0]
        int m = p.shape[0]
        double opts[LM_OPTS_SZ]
        double info[LM_INFO_SZ]
        double* work = NULL

        _LMFunction lm_func
        # Box constraints
        _LMBoxConstraints bc
        # Linear equation/inequality constraints
        _LMLinConstraints lec, lic
        # Output
        int niter = 0
        npy_intp* dims = [m, m]
        ndarray covr = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0)

    # Set the functions and their extra arguments, and verify them.
    lm_func = _LMFunction(func, args, jacf)
    verify_funcs(lm_func, p, m, n)
    # Set the iteration parameters: `opts` and `maxit`
    set_iter_params(mu, eps1, eps2, eps3, maxit, cdif, opts)

    if C is not None:
        lic = _LMLinConstraints(C, d, m)
        if A is not None:
            lec = _LMLinConstraints(A, b, m)
            if bounds is not None:
                # Box, linear equations & inequalities constrained minimization
                bc = _LMBoxConstraints(bounds, m)
                if jacf is not None:
                    work = _LMWork.allocate(LM_BLEIC_DER_WORKSZ(m, n, lec.k, lic.k))
                    niter = dlevmar_bleic_der(
                        callback_func, callback_jacf,
                        <double*>p.data, <double*>x.data, m, n,
                        bc.lb, bc.ub,
                        lec.mat, lec.vec, lec.k, lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>lm_func)
                else:
                    work = _LMWork.allocate( LM_BLEIC_DIF_WORKSZ(m, n, lec.k, lic.k))
                    niter = dlevmar_bleic_dif(
                        callback_func,
                        <double*>p.data, <double*>x.data, m, n,
                        bc.lb, bc.ub,
                        lec.mat, lec.vec, lec.k, lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>lm_func)
            else:
                # Linear equation & inequality constraints
                if jacf is not None:
                    work = _LMWork.allocate(LM_BLEIC_DER_WORKSZ(m, n, lec.k, lic.k))
                    niter = dlevmar_leic_der(
                        callback_func, callback_jacf,
                        <double*>p.data, <double*>x.data, m, n,
                        lec.mat, lec.vec, lec.k, lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>lm_func)
                else:
                    work = _LMWork.allocate(LM_BLEIC_DIF_WORKSZ(m, n, lec.k, lic.k))
                    niter = dlevmar_leic_dif(
                        callback_func,
                        <double*>p.data, <double*>x.data, m, n,
                        lec.mat, lec.vec, lec.k, lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>lm_func)
        else:
            if bounds is not None:
                # Box & linear inequality constraints
                bc = _LMBoxConstraints(bounds, m)
                if jacf is not None:
                    work = _LMWork.allocate(LM_BLEIC_DER_WORKSZ(m, n, 0, lic.k))
                    niter = dlevmar_blic_der(
                        callback_func, callback_jacf,
                        <double*>p.data, <double*>x.data, m, n,
                        bc.lb, bc.ub, lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>lm_func)
                else:
                    work = _LMWork.allocate(LM_BLEIC_DIF_WORKSZ(m, n, 0, lic.k))
                    niter = dlevmar_blic_dif(
                        callback_func,
                        <double*>p.data, <double*>x.data, m, n,
                        bc.lb, bc.ub, lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>lm_func)
            else:
                # Linear inequality constraints
                if jacf is not None:
                    work = _LMWork.allocate(LM_BLEIC_DER_WORKSZ(m, n, 0, lic.k))
                    niter = dlevmar_lic_der(
                        callback_func, callback_jacf,
                        <double*>p.data, <double*>x.data, m, n,
                        lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>lm_func)
                else:
                    work = _LMWork.allocate(LM_BLEIC_DIF_WORKSZ(m, n, 0, lic.k))
                    niter = dlevmar_lic_dif(
                        callback_func,
                        <double*>p.data, <double*>x.data, m, n,
                        lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>lm_func)
    elif A is not None:
        lec = _LMLinConstraints(A, b, m)
        if bounds is not None:
            # Box & linear equation constrained minimization
            bc = _LMBoxConstraints(bounds, m)
            if jacf is not None:
                work = _LMWork.allocate(LM_BLEC_DER_WORKSZ(m, n, lec.k))
                niter = dlevmar_blec_der(
                    callback_func, callback_jacf,
                    <double*>p.data, <double*>x.data, m, n,
                    bc.lb, bc.ub, lec.mat, lec.vec, lec.k, NULL,
                    maxit, opts, info, work,
                    <double*>covr.data, <void*>lm_func)
            else:
                work = _LMWork.allocate(LM_BLEC_DIF_WORKSZ(m, n, lec.k))
                niter = dlevmar_blec_dif(
                    callback_func,
                    <double*>p.data, <double*>x.data, m, n,
                    bc.lb, bc.ub, lec.mat, lec.vec, lec.k, NULL,
                    maxit, opts, info, work,
                    <double*>covr.data, <void*>lm_func)
        else:
            # Linear equation constrained minimization
            if jacf is not None:
                work = _LMWork.allocate(LM_LEC_DER_WORKSZ(m, n, lec.k))
                niter = dlevmar_lec_der(
                    callback_func, callback_jacf,
                    <double*>p.data, <double*>x.data, m, n,
                    lec.mat, lec.vec, lec.k,
                    maxit, opts, info, work,
                    <double*>covr.data, <void*>lm_func)
            else:
                work = _LMWork.allocate(LM_LEC_DIF_WORKSZ(m, n, lec.k))
                niter = dlevmar_lec_dif(
                    callback_func,
                    <double*>p.data, <double*>x.data, m, n,
                    lec.mat, lec.vec, lec.k,
                    maxit, opts, info, work,
                    <double*>covr.data, <void*>lm_func)
    elif bounds is not None:
        # Box-constrained minimization
        bc = _LMBoxConstraints(bounds, m)
        if jacf is not None:
            work = _LMWork.allocate(LM_BC_DER_WORKSZ(m, n))
            niter = dlevmar_bc_der(
                callback_func, callback_jacf,
                <double*>p.data, <double*>x.data, m, n,
                bc.lb, bc.ub,
                maxit, opts, info, work,
                <double*>covr.data, <void*>lm_func)
        else:
            work = _LMWork.allocate(LM_BC_DIF_WORKSZ(m, n))
            niter = dlevmar_bc_dif(
                callback_func,
                <double*>p.data, <double*>x.data, m, n,
                bc.lb, bc.ub,
                maxit, opts, info, work,
                <double*>covr.data, <void*>lm_func)
    else:
        # Unconstrained minimization
        if jacf is not None:
            work = _LMWork.allocate(LM_DER_WORKSZ(m, n))
            niter = dlevmar_der(
                callback_func, callback_jacf,
                <double*>p.data, <double*>x.data, m, n,
                maxit, opts, info, work, <double*>covr.data, <void*>lm_func)
        else:
            work = _LMWork.allocate(LM_DIF_WORKSZ(m, n))
            niter = dlevmar_dif(
                callback_func,
                <double*>p.data, <double*>x.data, m, n,
                maxit, opts, info, work, <double*>covr.data, <void*>lm_func)

    reason_id = <int>info[6]
    if reason_id in _LM_STOP_REASONS_WARNED:
        # Issue warning for unsuccessful termination.
        warnings.warn(_LM_STOP_REASONS[reason_id], LMWarning)
    if niter == LM_ERROR:
        if reason_id == 7:
            raise LMUserFuncError(
                "Stopped by invalid values (NaN or Inf) returned by `func`")
        else:
            raise LMRuntimeError

    return p, covr, py_info(info)


def __py_verify_funcs(_LMFunction func, ndarray p, int m, int n):
    """Test porpose only"""
    func._check_funcs(p, m, n)
    return True
