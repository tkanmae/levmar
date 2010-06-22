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
from cStringIO import StringIO
from numpy import (array_str, finfo)


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


_LM_MAXITER = 1000

_LM_STOP_REASONS = {
    1: "Stopped by small gradient J^T e",
    2: "Stop by small Dp",
    3: "Stop by `maxit`",
    4: "Singular matrix.  Restart from current `p` with increased mu",
    5: "No further error reduction is possible. Restart with increased mu",
    6: "Stopped by small ||e||_2",
}

_LM_STOP_REASONS_WARNED = (3, 4, 5)

## The stopping threshold for ||J^T e||_inf
_LM_EPS1 = DBL_EPSILON**(1/2)
## The stopping threshold for ||Dp||_2
_LM_EPS2 = DBL_EPSILON**(1/2)
## The stopping threshold for ||e||_2
_LM_EPS3 = DBL_EPSILON**(1/2)


class LMError(Exception):
    pass

class LMRuntimeError(LMError, RuntimeError):
    pass

class LMUserFuncError(LMError):
    pass

class LMWarning(UserWarning):
    pass


cdef class LMFunction:
    cdef void eval_func(self, double *p, double *x, int m, int n):
        raise NotImplementedError()

    cdef void eval_jacf(self, double *p, double *jacf, int m, int n):
        raise NotImplementedError()

    cdef int _check_funcs(self, double *p, int m, int n) except -1:
        raise NotImplementedError()

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


cdef class LMPyFunction(LMFunction):
    cdef:
        object func
        object jacf
        object args

    def __init__(self, func, args, jacf=None):
        self.func = func
        self.args = args
        self.jacf = jacf

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void eval_func(self, double *p, double *y, int m, int n):
        cdef:
            object py_p = \
               PyArray_SimpleNewFromData(1, <npy_intp*>&m, NPY_DOUBLE, <void*>p)
            ndarray[dtype_t, ndim=1, mode='c'] py_y

        args = PySequence_Concat((py_p,), self.args)
        py_y = PyObject_CallObject(<object>self.func, args)
        ## Copy the result to `x`
        memcpy(y, py_y.data, sizeof(double)*n)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void eval_jacf(self, double *p, double *jacf, int m, int n):
        cdef:
            object py_p = \
               PyArray_SimpleNewFromData(1, <npy_intp*>&m, NPY_DOUBLE, <void*>p)
            ndarray[dtype_t, ndim=2, mode='c'] py_jacf

        args = PySequence_Concat((py_p,), self.args)
        py_jacf = PyObject_CallObject(<object>self.jacf, args)
        ## Copy the result to `jacf`
        memcpy(jacf, py_jacf.data, sizeof(double)*n*m)

    cdef int _check_funcs(self, double *p, int m, int n) except -1:
        cdef object py_p = \
                PyArray_SimpleNewFromData(1, <npy_intp*>&m, NPY_DOUBLE, <void*>p)

        args = (py_p,) + self.args
        try:
            ret = self.func(*args)
        except Exception, e:
            raise LMUserFuncError(e)
        if ret.size != n:
            msg = ("`{0.__name__}()` returned a invalid size vector: "
                   "{1} expected but {2} given".format(self.func, n, ret.size))
            raise LMUserFuncError(msg)
        if self.jacf is not None:
            try:
                ret = self.jacf(*args)
            except Exception, e:
                raise LMUserFuncError(e)
            if ret.size != m*n:
                msg = ("`{0.__name__}()` returned a invalid size vector: "
                       "{1} expected but {2} given".format(self.jacf, m*n, ret.size))
                raise LMUserFuncError(msg)
            if not self._check_jacf(p, m, n):
                msg = ("`{0.__name__}()` may not be a correct Jacobian of "
                       "`{1.__name__}()`".format(self.jacf, self.func))
                warnings.warn(msg, LMWarning)
        return 1


cdef inline void callback_func(double *p, double *y, int m, int n, void *ctx):
    (<LMFunction>ctx).eval_func(p, y, m, n)


cdef inline void callback_jacf(double *p, double *jacf, int m, int n, void *ctx):
    (<LMFunction>ctx).eval_jacf(p, jacf, m, n)


cdef class LMConstraint:
    pass


cdef class LMBoxConstraint(LMConstraint):
    cdef:
        double* lb
        double* ub

    def __init__(self, object bounds, int m):
        if not isinstance(bounds, (list, tuple)):
            raise TypeError("`bounds` must be a tuple/list")
        if len(bounds) != m:
            raise ValueError("`bounds` must be length of {0} "
                             "(given length is {1})".format(m, len(bounds)))

        for i, b in enumerate(bounds):
            if b is None:
                self.lb[i] = -DBL_MAX
                self.ub[i] =  DBL_MAX
            elif len(b) == 2:
                if b[0] in (None, NPY_NAN, -NPY_INFINITY):
                    self.lb[i] = -DBL_MAX
                else:
                    self.lb[i] = float(b[0])
                if b[1] in (None, NPY_NAN, NPY_INFINITY):
                    self.ub[i] = DBL_MAX
                else:
                    self.ub[i] = float(b[1])
            else:
                raise ValueError("A component of `bounds` must be as "
                                 "a None or a sequence of 2 floats/Nones")

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


cdef class LMLinEqnLikeConstraint(LMConstraint):
    cdef:
        double* mat
        double* vec
        int k

    def __cinit__(self, object mat, object vec, int m):
        if m < 2:
            raise ValueError("Linear equation constraints can not be defined.")
        cdef:
            ndarray[dtype_t,ndim=2,mode='c'] A = \
                    PyArray_ContiguousFromAny(mat, NPY_DOUBLE, 2, 2)
            ndarray[dtype_t,ndim=1,mode='c'] b = \
                    PyArray_ContiguousFromAny(vec, NPY_DOUBLE, 1, 1)
            int k

        if A.shape[1] != m:
            raise ValueError("The shape of the constraint matrix "
                             "must be (kx{0})".format(m))
        ## the number of equations/inequalities
        k = A.size // m

        if b.size != k:
            raise ValueError("The shape of the RH constraint vector "
                             "must be consistent with ({0}x1)".format(k))

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


cdef class LMLinEqnConstraint(LMLinEqnLikeConstraint):
    pass

cdef class LMInequalConstraint(LMLinEqnLikeConstraint):
    pass


cdef class LMWorkSpace:
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


## This global variable holds a working array used internally in the
## levmar library.  The necessarily size of the array depends on
## specification of problems.
cdef LMWorkSpace _LMWork = LMWorkSpace()


class Output(object):
    """A class stores output from `levmar`.

    Attributes
    ----------
    p : ndarray, shape=(m,)
        The best-fit parameters.
    p_stdv : ndarray, shape=(m,)
        The standard deviation of the best-fit parameters.  The standard
        deviation is computed as square root of diagonal elements of the
        covariance matrix.
    r2 : float
        The coefficient of determination.
    covar : ndarray, shape=(m,m)
        The covariance matrix corresponding to the least square fit.
    corr : ndarray, shape=(m,m)
        Pearson's correlation coefficient of the best-fit parameters.
    ndf : int
        The degrees of freedom.
    niter : int
        The number of the iterations
    reason : str
        The reason for the termination
    info : list
        Various information regarding the minimization.
            0: ||e||_2 at `p0`.
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

    Methods
    -------
    pprint()
        Print a summary of the fit.
    """
    __slots__ = ('_p', '_p_stdv', '_r2', '_covar', '_corr', '_m', '_n', '_info',
                 '_cache')
    def __init__(self, p, p_stdv, r2, covar, corr, m, n, info):
        self._p = p
        self._p_stdv = p_stdv
        self._r2 = r2
        self._covar = covar
        self._corr = corr
        self._m = m
        self._n = n
        self._info = info
        self._cache = None

        ## These arrays must be read-only.
        self._p.setflags(False)
        self._p_stdv.setflags(False)
        self._covar.setflags(False)
        self._corr.setflags(False)

    def pprint(self):
        print(self.__str__())

    def __str__(self):
        if self._cache is None:
            buf = StringIO()

            buf.write("Iteration: {0}\n".format(self.niter))
            buf.write("Reason: {0}\n\n".format(self.reason))
            buf.write("Degrees of freedom: {0}\n\n".format(self.ndf))
            buf.write("Parameters:\n")
            for i, (p, p_stdv) in enumerate(zip(self._p, self._p_stdv)):
                buf.write("p[{0}] = {1:<+12.6g} +/- {2:<12.6g} "
                      "({3:.1f}%)\n".format(i, p, p_stdv, 100*abs(p_stdv/p)))
            buf.write("\n")
            buf.write("Covariance:\n")
            buf.write(array_str(self._covar, precision=4))
            buf.write("\n\n")
            buf.write("Correlation:\n")
            buf.write(array_str(self._corr, precision=4))
            buf.write("\n\n")
            buf.write("R2: {0:.5f}".format(self._r2))

            self._cache = buf.getvalue()
            buf.close()

        return self._cache

    @property
    def p(self):
        return self._p

    @property
    def p_stdv(self):
        return self._p_stdv

    @property
    def r2(self):
        return self._r2

    @property
    def covar(self):
        return self._covar

    @property
    def corr(self):
        return self._corr

    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n

    @property
    def info(self):
        return self._info

    @property
    def ndf(self):
        """The degrees of freedom"""
        return self._n - self._m

    @property
    def niter(self):
        """The number of the iterations"""
        return self._info[2]

    @property
    def reason(self):
        return self._info[3]


cdef int __check_funcs(LMFunction func, object p, object y) except -1:
    cdef:
        ndarray[dtype_t,ndim=1,mode='c'] p_ = p
        ndarray[dtype_t,ndim=1,mode='c'] y_ = y
        int m = p_.shape[0]
        int n = y_.shape[0]
    func._check_funcs(<double*>p_.data, m, n)
    return 1


cdef object py_info(double *c_info):
    info = [[] for i in range(7)]
    info[0] = c_info[0]         # ||e||_2 at `p0`
    info[1] = (c_info[1],       # ||e||_2 at `p`
               c_info[2],       # ||J^T.e||_inf
               c_info[3],       # ||Dp||_2
               c_info[4])       # mu / max[J^T.J]_ii
    info[2] = <int>c_info[5]    # number of iterations

    ## Issue warning for unsuccessful termination.
    reason = <int>c_info[6]     # reason for terminating
    msg = _LM_STOP_REASONS[reason]
    if reason in _LM_STOP_REASONS_WARNED:
        warnings.warn(msg, LMWarning)
    info[3] = msg
    info[4] = <int>c_info[7]    # number of `func` evaluations
    info[5] = <int>c_info[8]    # number of `jacf` evaluations
    info[6] = <int>c_info[9]    # number of linear system solved

    return tuple(info)


@cython.wraparound(False)
@cython.boundscheck(False)
def _run_levmar(func, p0, ndarray[dtype_t,ndim=1,mode='c'] y, args=(), jacf=None,
                bounds=None, A=None, b=None, C=None, d=None,
                mu=1e-3, eps1=_LM_EPS1, eps2=_LM_EPS2, eps3=_LM_EPS3,
                maxit=1000, cntdif=False):
    """
    Parameters
    ----------
    func : callable
        A function or method taking, at least, one length of m vector
        and returning a length of n vector.
    y : array_like, shape (n,)
        The dependent data, or the observation.
    p0 : array_like, shape (m,)
        The initial estimate of the parameters.
    args : tuple, optional
        Extra arguments passed to `func` (and `jacf`).
    jacf : callable, optional
        A function or method computing the Jacobian of `func`.  It
        takes, at least, one length of m vector and returns a (nxm)
        matrix or a campatible C-contiguous vector.  If this is a None,
        a approximated Jacobian will be used.
    bounds : tuple/list, length m
        Box constraints.  Each constraint can be a tuple of two
        floats/Nones or None.  A tuple determines the (inclusive) lower
        and upper bound, and None means no constraint.  If one of two
        values in a tuple is None, then the bound is semi-definite.
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
    cntdif : {True, False}, optional
        If this is True, the Jacobian is approximated with central
        differentiation.

    Returns
    -------
    output : levmar.Output
        The output of the minimization

    Notes
    -----
    * The linear equation constraints are specified as A*p=b where A is
    k1xm matrix and b is k1x1  vector (See comments in
    src/levmar-2.5/lmlec_core.c).

    * The linear *inequality* constraints are defined as C*p>=d where C
    is k2xm matrix and d is k2x1 vector (See comments in
    src/levmar-2.5/lmbleic_core.c).

    See Also
    --------
    levmar.Output
    """
    cdef:
        ## Make a copy of `p0`
        ndarray[dtype_t,ndim=1,mode='c'] p = \
                PyArray_ContiguousFromAny(p0, NPY_DOUBLE, 1, 1)

        int n = y.shape[0]
        int m = p.shape[0]
        int niter = 0
        double opts[LM_OPTS_SZ]
        double info[LM_INFO_SZ]
        double* work = NULL

        ## Box constraints
        LMBoxConstraint bc
        ## Linear equation constraints
        LMLinEqnConstraint lec
        ## Inequility constraints
        LMInequalConstraint lic
        ## Output
        npy_intp* dims = [m,m]
        ndarray[dtype_t,ndim=2,mode='c'] covr = \
                PyArray_ZEROS(2, dims, NPY_DOUBLE, 0)
        ndarray[dtype_t,ndim=2,mode='c'] corr = \
                PyArray_ZEROS(2, dims, NPY_DOUBLE, 0)
        ndarray[dtype_t,ndim=1,mode='c'] p_stdv = \
                PyArray_ZEROS(1, <npy_intp*>&m, NPY_DOUBLE, 0)
        double r2

    ## Set `func` (and `jacf`)
    py_func = LMPyFunction(func, args, jacf)
    __check_funcs(py_func, p, y)
    ## Set the iteration parameters: `opts` and `maxit`
    opts[0] = mu
    if eps1 >= 1: raise ValueError("`eps1` must be less than 1.")
    opts[1] = eps1
    if eps2 >= 1: raise ValueError("`eps2` must be less than 1.")
    opts[2] = eps2
    if eps2 >= 1: raise ValueError("`eps3` must be less than 1.")
    opts[3] = eps3
    opts[4] = LM_DIFF_DELTA if cntdif else -LM_DIFF_DELTA
    if maxit <= 0:
        raise ValueError("`maxit` must be a positive.")
    maxit = int(maxit)

    if C is not None:
        lic = LMInequalConstraint(C, d, m)
        if A is not None:
            lec = LMLinEqnConstraint(A, b, m)
            if bounds is not None:
                ## Box, linear equations & inequalities constrained minimization
                bc = LMBoxConstraint(bounds, m)
                if jacf is not None:
                    work = _LMWork.allocate(LM_BLEIC_DER_WORKSZ(m, n, lec.k, lic.k))
                    niter = dlevmar_bleic_der(
                        callback_func, callback_jacf,
                        <double*>p.data, <double*>y.data, m, n,
                        bc.lb, bc.ub,
                        lec.mat, lec.vec, lec.k, lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>py_func)
                else:
                    work = _LMWork.allocate(
                        LM_BLEIC_DIF_WORKSZ(m, n, lec.k, lic.k) * sizeof(double))
                    niter = dlevmar_bleic_dif(
                        callback_func,
                        <double*>p.data, <double*>y.data, m, n,
                        bc.lb, bc.ub,
                        lec.mat, lec.vec, lec.k, lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>py_func)
            else:
                ## Linear equation & inequality constraints
                if jacf is not None:
                    work = _LMWork.allocate(LM_BLEIC_DER_WORKSZ(m, n, lec.k, lic.k))
                    niter = dlevmar_leic_der(
                        callback_func, callback_jacf,
                        <double*>p.data, <double*>y.data, m, n,
                        lec.mat, lec.vec, lec.k, lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>py_func)
                else:
                    work = _LMWork.allocate(LM_BLEIC_DIF_WORKSZ(m, n, lec.k, lic.k))
                    niter = dlevmar_leic_dif(
                        callback_func,
                        <double*>p.data, <double*>y.data, m, n,
                        lec.mat, lec.vec, lec.k, lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>py_func)
        else:
            if bounds is not None:
                ## Box & linear inequality constraints
                bc = LMBoxConstraint(bounds, m)
                if jacf is not None:
                    work = _LMWork.allocate(LM_BLEIC_DER_WORKSZ(m, n, 0, lic.k))
                    niter = dlevmar_blic_der(
                        callback_func, callback_jacf,
                        <double*>p.data, <double*>y.data, m, n,
                        bc.lb, bc.ub, lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>py_func)
                else:
                    work = _LMWork.allocate(LM_BLEIC_DIF_WORKSZ(m, n, 0, lic.k))
                    niter = dlevmar_blic_dif(
                        callback_func,
                        <double*>p.data, <double*>y.data, m, n,
                        bc.lb, bc.ub, lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>py_func)
            else:
                ## Linear inequality constraints
                if jacf is not None:
                    work = _LMWork.allocate(LM_BLEIC_DER_WORKSZ(m, n, 0, lic.k))
                    niter = dlevmar_lic_der(
                        callback_func, callback_jacf,
                        <double*>p.data, <double*>y.data, m, n,
                        lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>py_func)
                else:
                    work = _LMWork.allocate(LM_BLEIC_DIF_WORKSZ(m, n, 0, lic.k))
                    niter = dlevmar_lic_dif(
                        callback_func,
                        <double*>p.data, <double*>y.data, m, n,
                        lic.mat, lic.vec, lic.k,
                        maxit, opts, info, work,
                        <double*>covr.data, <void*>py_func)
    elif A is not None:
        lec = LMLinEqnConstraint(A, b, m)
        if bounds is not None:
            ## Box & linear equation constrained minimization
            bc = LMBoxConstraint(bounds, m)
            if jacf is not None:
                work = _LMWork.allocate(LM_BLEC_DER_WORKSZ(m, n, lec.k))
                niter = dlevmar_blec_der(
                    callback_func, callback_jacf,
                    <double*>p.data, <double*>y.data, m, n,
                    bc.lb, bc.ub, lec.mat, lec.vec, lec.k, NULL,
                    maxit, opts, info, work,
                    <double*>covr.data, <void*>py_func)
            else:
                work = _LMWork.allocate(LM_BLEC_DIF_WORKSZ(m, n, lec.k))
                niter = dlevmar_blec_dif(
                    callback_func,
                    <double*>p.data, <double*>y.data, m, n,
                    bc.lb, bc.ub, lec.mat, lec.vec, lec.k, NULL,
                    maxit, opts, info, work,
                    <double*>covr.data, <void*>py_func)
        else:
            ## Linear equation constrained minimization
            if jacf is not None:
                work = _LMWork.allocate(LM_LEC_DER_WORKSZ(m, n, lec.k))
                niter = dlevmar_lec_der(
                    callback_func, callback_jacf,
                    <double*>p.data, <double*>y.data, m, n,
                    lec.mat, lec.vec, lec.k,
                    maxit, opts, info, work,
                    <double*>covr.data, <void*>py_func)
            else:
                work = _LMWork.allocate(LM_LEC_DIF_WORKSZ(m, n, lec.k))
                niter = dlevmar_lec_dif(
                    callback_func,
                    <double*>p.data, <double*>y.data, m, n,
                    lec.mat, lec.vec, lec.k,
                    maxit, opts, info, work,
                    <double*>covr.data, <void*>py_func)
    elif bounds is not None:
        ## Box-constrained minimization
        bc = LMBoxConstraint(bounds, m)
        if jacf is not None:
            work = _LMWork.allocate(LM_BC_DER_WORKSZ(m, n))
            niter = dlevmar_bc_der(
                callback_func, callback_jacf,
                <double*>p.data, <double*>y.data, m, n,
                bc.lb, bc.ub,
                maxit, opts, info, work,
                <double*>covr.data, <void*>py_func)
        else:
            work = _LMWork.allocate(
                LM_BC_DIF_WORKSZ(m, n) * sizeof(double))
            niter = dlevmar_bc_dif(
                callback_func,
                <double*>p.data, <double*>y.data, m, n,
                bc.lb, bc.ub,
                maxit, opts, info, work,
                <double*>covr.data, <void*>py_func)
    else:
        ## Unconstrained minimization
        if jacf is not None:
            work = _LMWork.allocate(LM_DER_WORKSZ(m, n))
            niter = dlevmar_der(
                callback_func, callback_jacf,
                <double*>p.data, <double*>y.data, m, n,
                maxit, opts, info, work, <double*>covr.data, <void*>py_func)
        else:
            work = _LMWork.allocate(LM_DIF_WORKSZ(m, n))
            niter = dlevmar_dif(
                callback_func,
                <double*>p.data, <double*>y.data, m, n,
                maxit, opts, info, work, <double*>covr.data, <void*>py_func)

    cdef int i, j
    if niter != LM_ERROR:
        for i in range(m):
            p_stdv[i] = dlevmar_stddev(<double*>covr.data, m, i)
        for i in range(m):
            for j in range(m):
                corr[i][j] = dlevmar_corcoef(<double*>covr.data, m, i, j)
        r2 = dlevmar_R2(callback_func,
                        <double*>p.data, <double*>y.data, m, n, <void*>py_func)
        output = Output(p, p_stdv, r2, covr, corr, m, n, py_info(info))

    if niter == LM_ERROR:
        if <int>info[6] == 7:
            raise LMUserFuncError(
                "Stopped by invalid values (NaN or Inf) returned by `func`")
        else:
            raise LMRuntimeError

    return output
