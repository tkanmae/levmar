# -*- coding: utf-8 -*-
"""
TODO:
    * Implement a weighted least square method.
    * Follow the data structures in `scipy.odr` module.
"""
# cython: cdivision=True
from __future__ import division
cimport cython
from numpy cimport *
import warnings
from cStringIO import StringIO
import numpy as np

ctypedef float64_t dtype_t


import_array()


_LM_MAXITER = 1000

_LM_STOP_REASONS = {
    1: "Stopped by small gradient J^T e",
    2: "Stop by small Dp",
    3: "Stop by `maxiter`",
    4: "Singular matrix.  Restart from current `p` with increased `mu`",
    5: "No further error reduction is possible. Restart with increased mu",
    6: "Stopped by small ||e||_2",
}

_LM_STOP_REASONS_WARNED = (3, 4, 5)


class LMError(Exception):
    pass

class LMRuntimeError(LMError):
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


cdef class __LMPyFunctionBase(LMFunction):
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void eval_func(self, double *p, double *x, int m, int n):
        cdef:
            object py_p = \
               PyArray_SimpleNewFromData(1, <npy_intp*>&m, NPY_DOUBLE, <void*>p)
            ndarray[dtype_t, ndim=1, mode='c'] py_x

        args = PySequence_Concat((py_p,), self.args)
        py_x = PyObject_CallObject(<object>self.func, args)
        ## Copy the result to `x`
        memcpy(x, py_x.data, sizeof(double)*n)

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


class LMPyFunction(__LMPyFunctionBase):
    def __init__(self, func, p, x, args, jacf=None):
        if not isinstance(args, tuple): args = args,
        if self.is_valid_func(func, p, x, args):
            self.func = func
        if jacf is not None:
            if self.is_valid_func(jacf, p, x, args, is_jacf=True):
                self.jacf = jacf
        self.args = args

    def is_valid_func(self, func, p, x, args, is_jacf=False):
        if not callable(func):
            raise TypeError("`func` must be callable")
        try:
            args = (p,) + args
            ret = func(*args)
        except Exception, e:
            raise LMUserFuncError(e)

        if is_jacf and ret.ndim != 2:
            raise LMUserFuncError("`{0.__name__}()` must return "
                                  "an 2-dimensional ndarray".format(func))
        nret = p.shape[0]*x.shape[0] if is_jacf else x.shape[0]
        if ret.size != nret:
            raise LMUserFuncError("`{0.__name__}()` returned "
                                  "an ndarray of invalid size".format(func))
        return True


cdef void callback_func(double *p, double *x, int m, int n, void *ctx):
    (<LMFunction>ctx).eval_func(p, x, m, n)


cdef void callback_jacf(double *p, double *jacf, int m, int n, void *ctx):
    (<LMFunction>ctx).eval_jacf(p, jacf, m, n)


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
    r2: float
        The coefficient of determination.
    covar: ndarray, shape=(m,m)
        The covariance matrix corresponding to the least square fit.
    corr: ndarray, shape=(m,m)
        Pearson's correlation coefficient of the best-fit parameters.
    ndf: int
        The degrees of freedom.
    niter: int
        The number of the iterations
    reason: str
        The reason for the termination
    info: list
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
    __slots__ = ('_p', '_p_stdv', '_r2', '_covar', '_corr', '_m', '_n', '_info')
    def __init__(self, p, p_stdv, r2, covar, corr, m, n, info):
        self._p = p
        self._p_stdv = p_stdv
        self._r2 = r2
        self._covar = covar
        self._corr = corr
        self._m = m
        self._n = n
        self._info = info

        ## These arrays must be read-only.
        self._p.setflags(False)
        self._p_stdv.setflags(False)
        self._covar.setflags(False)
        self._corr.setflags(False)

    def pprint(self):
        print(self.__str__())

    def __str__(self):
        buf = StringIO()

        buf.write("Iteration: {0}\n".format(self.iter))
        buf.write("Reason: {0}\n\n".format(self.reason))
        buf.write("Degrees of freedom: {0}\n\n".format(self.ndf()))
        buf.write("Parameters:\n")
        for i, (p, p_stdv) in enumerate(zip(self._p, self._p_stdv)):
            buf.write("p[{0}] = {1:<+12.6g} +/- {2:<12.6g} "
                  "({3:.1f}%)\n".format(i, p, p_stdv, 100*abs(p_stdv/p)))
        buf.write("\n")
        buf.write("Covariance:\n")
        buf.write(np.array_str(self._covar, precision=4))
        buf.write("\n\n")
        buf.write("Correlation:\n")
        buf.write(np.array_str(self._corr, precision=4))
        buf.write("\n\n")
        buf.write("R2: {0:.5f}".format(self._r2))

        self._buf = buf.getvalue()
        buf.close()

        return self._buf

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


cdef object verify_bounds(object bounds, int m):
    if not isinstance(bounds, (list,tuple)):
            raise TypeError("`bound` must be a tuple/list")
    else:
        if len(bounds) != m:
            raise ValueError("`bounds` must be the same size as `p0`")

    lb = np.empty(len(bounds))
    ub = np.empty(len(bounds))

    for i, b in enumerate(bounds):
        if b is None:
            lb[i] = -DBL_MAX
            ub[i] =  DBL_MAX
        elif len(b) == 2:
            lb[i] = -DBL_MAX if b[0] == None else float(b[0])
            ub[i] =  DBL_MAX if b[1] == None else float(b[1])
    return lb, ub


cdef int set_iterparams(object kw, double *opts, int *maxiter) except -1:
    """Set the iteration parameters `opts` and `maxiter`"""
    if 'opts.mu' in kw:
        opts[0] = kw['opts.mu']
    else:
        opts[0] = LM_INIT_MU
    if 'opts.eps1' in kw:
        opts[1] = kw['opts.eps1']
    else:
        opts[1] = LM_STOP_THRESH
    if 'opts.eps2' in kw:
        opts[2] = kw['opts.eps2']
    else:
        opts[2] = LM_STOP_THRESH
    if 'opts.eps3' in kw:
        opts[3] = kw['opts.eps3']
    else:
        opts[3] = LM_STOP_THRESH

    if 'opts.cdif' in kw:
        opts[4] = LM_DIFF_DELTA
    else:
        opts[4] = -LM_DIFF_DELTA

    if 'maxiter' in kw and isinstance(kw['maxiter'], int):
        maxiter[0] = kw['maxiter']
    else:
        maxiter[0] = _LM_MAXITER


cdef object py_info(double *c_info):
    info = [[] for i in range(7)]
    info[0] = c_info[0]         # ||e||_2 at `p0`
    info[1] = (c_info[1],       # ||e||_2 at `p`
               c_info[2],       # ||J^T.e||_inf
               c_info[3],       # ||Dp||_2
               c_info[4])       # mu / max[J^T.J]_ii
    info[2] = <int>c_info[5]    # number of iterations

    reason = <int>c_info[6]     # reason for terminating
    msg = _LM_STOP_REASONS[reason]
    if reason in _LM_STOP_REASONS_WARNED:
        warnings.warn(msg, LMWarning)
    info[3] = msg
    info[4] = <int>c_info[7]    # number of `func` evaluations
    info[5] = <int>c_info[8]    # number of `jacf` evaluations
    info[6] = <int>c_info[9]    # number of linear system solved.

    return tuple(info)


@cython.wraparound(False)
@cython.boundscheck(False)
def bc(func, p0, ndarray[dtype_t,ndim=1] x, args=(), bounds=None, **kwargs):
    ## Make a copy of `p0`
    cdef ndarray[dtype_t,ndim=1,mode='c'] p = \
            np.array(p0, dtype=np.float64, ndmin=1)
    ## Ensure `x` is a C-contiguous array
    x = np.asarray(x, dtype=np.float64, order='C')

    cdef:
        int n = x.shape[0]
        int m = p.shape[0]
        int maxiter = 0
        int niter = 0
        double opts[LM_OPTS_SZ]
        double info[LM_INFO_SZ]
        ndarray[dtype_t, ndim=1] lb
        ndarray[dtype_t, ndim=1] ub

        bint has_jacf = False

        npy_intp* dims = [m,m]
        ndarray[dtype_t,ndim=2,mode='c'] covr = \
                PyArray_ZEROS(2, dims, NPY_DOUBLE, 0)
        ndarray[dtype_t,ndim=2,mode='c'] corr = \
                PyArray_ZEROS(2, dims, NPY_DOUBLE, 0)
        ndarray[dtype_t,ndim=1,mode='c'] p_stdv = \
                PyArray_ZEROS(1, <npy_intp*>&m, NPY_DOUBLE, 0)
        double r2

    has_jacf = True if 'jacf' in kwargs else False
    if has_jacf:
        py_func = LMPyFunction(func, p, x, args, kwargs['jacf'])
    else:
        py_func = LMPyFunction(func, p, x, args)
    ## Set the iteration parameters: `opts` and `maxiter`
    set_iterparams(kwargs, opts, &maxiter)
    ## `bounds`
    if bounds is not None:
        lb, ub = verify_bounds(bounds, m)

    ## -- Call a dlevmar_* function.
    if bounds is None:
        if has_jacf:
            niter = dlevmar_der(
                callback_func, callback_jacf,
                <double*>p.data, <double*>x.data, m, n,
                maxiter, opts, info, NULL, <double*>covr.data, <void*>py_func)
        else:
            niter = dlevmar_dif(
                callback_func,
                <double*>p.data, <double*>x.data, m, n,
                maxiter, opts, info, NULL, <double*>covr.data, <void*>py_func)
    else:
        if has_jacf:
            niter = dlevmar_bc_der(
                callback_func, callback_jacf,
                <double*>p.data, <double*>x.data, m, n,
                <double*>lb.data, <double*>ub.data,
                maxiter, opts, info, NULL, <double*>covr.data, <void*>py_func)
        else:
            niter = dlevmar_bc_dif(
                callback_func,
                <double*>p.data, <double*>x.data, m, n,
                <double*>lb.data, <double*>ub.data,
                maxiter, opts, info, NULL, <double*>covr.data, <void*>py_func)

    cdef int i, j
    if niter != LM_ERROR:
        for i in range(m):
            p_stdv[i] = dlevmar_stddev(<double*>covr.data, m, i)
        for i in range(m):
            for j in range(m):
                corr[i][j] = dlevmar_corcoef(<double*>covr.data, m, i, j)
        r2 = dlevmar_R2(callback_func,
                        <double*>p.data, <double*>x.data, m, n, <void*>py_func)
        output = Output(p, p_stdv, r2, covr, corr, m, n, py_info(info))

    if niter == LM_ERROR:
        if <int>info[6] == 7:
            raise LMUserFuncError("`func` likely returned a invalid value")
        else:
            raise LMRuntimeError

    return output
