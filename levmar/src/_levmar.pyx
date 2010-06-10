# -*- coding: utf-8 -*-
#cython: embedsignature=True
"""
TODO:
    * Implement a weighted least square method.
    * Follow the data structures in `scipy.odr` module.
"""
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

__eps = np.finfo(float).eps
## The stopping threshold for ||J^T e||_inf
_LM_EPS1 = __eps**(1/2)
## The stopping threshold for ||Dp||_2
_LM_EPS2 = __eps**(1/2)
## The stopping threshold for ||e||_2
_LM_EPS3 = __eps**(1/2)


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


class LMPyFunction(__LMPyFunctionBase):
    def __init__(self, func, p, y, args, jacf=None):
        if not isinstance(args, tuple): args = args,
        if self._is_func_valid(func, p, y, args):
            self.func = func
        if jacf is not None and self._is_jacf_valid(jacf, p, y, args):
            self.jacf = jacf
        self.args = args

    def _is_func_valid(self, func, p, y, args):
        ret = self._verify_callable(func, p, args)
        if ret.size != y.shape[0]:
            raise LMUserFuncError(
                "`{0.__name__}()` returned a invalid size vector: "
                "{1} expected but {2} given"
                .format(func, y.shape[0], ret.size))
        return True

    def _is_jacf_valid(self, jacf, p, y, args):
        ret = self._verify_callable(jacf, p, args)
        if ret.size != p.shape[0]*y.shape[0]:
            raise LMUserFuncError(
                "`{0.__name__}()` returned a invalid size vector: "
                "{1} expected but {2} given"
                .format(jacf, y.shape[0]*p.shape[0], ret.size))
        return True

    def _verify_callable(self, func, p, args):
        if not callable(func):
            raise TypeError("`func/jacf` must be callable")
        try:
            args = (p,) + args
            ret = func(*args)
        except Exception, e:
            raise LMUserFuncError(e)
        return ret


cdef void callback_func(double *p, double *y, int m, int n, void *ctx):
    (<LMFunction>ctx).eval_func(p, y, m, n)


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

        buf.write("Iteration: {0}\n".format(self.niter))
        buf.write("Reason: {0}\n\n".format(self.reason))
        buf.write("Degrees of freedom: {0}\n\n".format(self.ndf))
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


cdef object verify_bc(object bounds, int m):
    if not isinstance(bounds, (list, tuple)):
        raise TypeError("`bounds` must be a tuple/list")
    if len(bounds) != m:
        raise ValueError("`bounds` must be length of {0} "
                         "(given length is {1})".format(m, len(bounds)))
    lb = np.empty(m)
    ub = np.empty(m)

    for i, b in enumerate(bounds):
        if b is None:
            lb[i] = -DBL_MAX
            ub[i] =  DBL_MAX
        elif len(b) == 2:
            if b[0] in (None, np.nan, -np.inf):
                lb[i] = -DBL_MAX
            else:
                lb[i] = float(b[0])
            if b[1] in (None, np.nan, np.inf):
                ub[i] = DBL_MAX
            else:
                ub[i] = float(b[1])
        else:
            raise ValueError("Each in `bounds` must be given as None "
                             "or a sequence of 2 floats")

    return lb, ub


cdef object verify_lc(object A, object b, int m):
    if m < 2:
        raise ValueError("Linear equation/inequility constraints can not be defined.")

    A = np.array(A, dtype=np.float64, copy=False, order='C', ndmin=2)
    if A.shape[1] != m:
        raise ValueError("The shape of the constraint matrix "
                         "must be (kx{0})".format(m))
    if not np.alltrue(np.isfinite(A)):
        raise ValueError("The constraint matrix should not contain "
                         "non-finite values.")
    ## the number of equations/inequalities
    k = A.size // m

    b = np.array(b, dtype=np.float64, copy=False, order='C', ndmin=1)
    if b.size != k:
        raise ValueError("The shape of the RH constraint vector "
                         "must be consistent with ({0}x1)".format(k))
    if not np.alltrue(np.isfinite(b)):
        raise ValueError("The RH constraint vector should not contain "
                         "non-finite values.")
    return A, b, k


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
def levmar(func, p0, ndarray[dtype_t,ndim=1] y, args=(), jacf=None,
           bounds=None, A=None, b=None, C=None, d=None,
           mu=1e-3, eps1=_LM_EPS1, eps2=_LM_EPS2, eps3=_LM_EPS3,
           maxiter=1000, cntdif=False):
    """
    Parameters
    ----------
    func : callable
        A function or method taking, at least, one length of n vector
        and returning a length of m vector.
    p0 : array_like, shape (m,)
        The initial estimate of the parameters.
    args : tuple, optional
        Extra arguments passed to `func` (and `jacf`) in this tuple.
    jacf : callable, optional
        A function or method to compute the Jacobian of `func`.  If this
        is None, a approximated Jacobian will be used.
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
    maxiter : int, optional
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
    * The linear equation constraints are specified as A*p=b where A
    is k1xm matrix and b is k1x1  vector (See comments in
    src/levmar-2.5/lmlec_core.c).

    * The linear inequality constraints are defined as C*p>=d where C
    is k2xm matrix and d is k2x1 vector (See comments in
    src/levmar-2.5/lmbleic_core.c).

    See Also
    --------
    levmar.Output
    """
    ## Make a copy of `p0`
    cdef ndarray[dtype_t,ndim=1,mode='c'] p = \
            np.array(p0, dtype=np.float64, ndmin=1)
    ## Ensure `y` is a C-contiguous array
    y = np.asarray(y, dtype=np.float64, order='C')

    cdef:
        int n = y.shape[0]
        int m = p.shape[0]
        # int maxiter = 0
        int niter = 0
        double opts[LM_OPTS_SZ]
        double info[LM_INFO_SZ]
        ## Box constraints
        ndarray[dtype_t, ndim=1] lb
        ndarray[dtype_t, ndim=1] ub
        ## Linear equation/inequility constraints
        int k1 = 0
        int k2 = 0
        ndarray[dtype_t, ndim=2] A_
        ndarray[dtype_t, ndim=1] b_
        ndarray[dtype_t, ndim=2] C_
        ndarray[dtype_t, ndim=1] d_
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
    py_func = LMPyFunction(func, p, y, args, jacf)
    ## Set the iteration parameters: `opts` and `maxiter`
    opts[0] = mu
    if eps1 >= 1: raise ValueError("`eps1` must be less than 1.")
    opts[1] = eps1
    if eps2 >= 1: raise ValueError("`eps2` must be less than 1.")
    opts[2] = eps2
    if eps2 >= 1: raise ValueError("`eps3` must be less than 1.")
    opts[3] = eps3
    opts[4] = LM_DIFF_DELTA if cntdif else -LM_DIFF_DELTA
    if maxiter <= 0:
        raise ValueError("`maxiter` must be a positive.")
    maxiter = int(maxiter)

    if C is not None:
        C_, d_, k2 = verify_lc(C, d, m)
        if A is not None:
            A_, b_, k1 = verify_lc(A, b, m)
            if bounds is not None:
                ## Box, linear equations & inequalities constrained minimization
                lb, ub = verify_bc(bounds, m)
                if jacf is not None:
                    niter = dlevmar_bleic_der(
                        callback_func, callback_jacf,
                        <double*>p.data, <double*>y.data, m, n,
                        <double*>lb.data, <double*>ub.data,
                        <double*>A_.data, <double*>b_.data, k1,
                        <double*>C_.data, <double*>d_.data, k2,
                        maxiter, opts, info, NULL,
                        <double*>covr.data, <void*>py_func)
                else:
                    niter = dlevmar_bleic_dif(
                        callback_func,
                        <double*>p.data, <double*>y.data, m, n,
                        <double*>lb.data, <double*>ub.data,
                        <double*>A_.data, <double*>b_.data, k1,
                        <double*>C_.data, <double*>d_.data, k2,
                        maxiter, opts, info, NULL,
                        <double*>covr.data, <void*>py_func)
            else:
                ## Linear equation & inequality constraints
                if jacf is not None:
                    niter = dlevmar_leic_der(
                        callback_func, callback_jacf,
                        <double*>p.data, <double*>y.data, m, n,
                        <double*>A_.data, <double*>b_.data, k1,
                        <double*>C_.data, <double*>d_.data, k2,
                        maxiter, opts, info, NULL,
                        <double*>covr.data, <void*>py_func)
                else:
                    niter = dlevmar_leic_dif(
                        callback_func,
                        <double*>p.data, <double*>y.data, m, n,
                        <double*>A_.data, <double*>b_.data, k1,
                        <double*>C_.data, <double*>d_.data, k2,
                        maxiter, opts, info, NULL,
                        <double*>covr.data, <void*>py_func)
        else:
            if bounds is not None:
                ## Box & linear inequality constraints
                lb, ub = verify_bc(bounds, m)
                if jacf is not None:
                    niter = dlevmar_blic_der(
                        callback_func, callback_jacf,
                        <double*>p.data, <double*>y.data, m, n,
                        <double*>lb.data, <double*>ub.data,
                        <double*>C_.data, <double*>d_.data, k2,
                        maxiter, opts, info, NULL,
                        <double*>covr.data, <void*>py_func)
                else:
                    niter = dlevmar_blic_dif(
                        callback_func,
                        <double*>p.data, <double*>y.data, m, n,
                        <double*>lb.data, <double*>ub.data,
                        <double*>C_.data, <double*>d_.data, k2,
                        maxiter, opts, info, NULL,
                        <double*>covr.data, <void*>py_func)
            else:
                ## Linear inequality constraints
                if jacf is not None:
                    niter = dlevmar_lic_der(
                        callback_func, callback_jacf,
                        <double*>p.data, <double*>y.data, m, n,
                        <double*>C_.data, <double*>d_.data, k2,
                        maxiter, opts, info, NULL,
                        <double*>covr.data, <void*>py_func)
                else:
                    niter = dlevmar_lic_dif(
                        callback_func,
                        <double*>p.data, <double*>y.data, m, n,
                        <double*>C_.data, <double*>d_.data, k2,
                        maxiter, opts, info, NULL,
                        <double*>covr.data, <void*>py_func)
    elif A is not None:
        A_, b_, k1 = verify_lc(A, b, m)
        if bounds is not None:
            ## Box & linear equation constrained minimization
            lb, ub = verify_bc(bounds, m)
            if jacf is not None:
                niter = dlevmar_blec_der(
                    callback_func, callback_jacf,
                    <double*>p.data, <double*>y.data, m, n,
                    <double*>lb.data, <double*>ub.data,
                    <double*>A_.data, <double*>b_.data, k1, NULL,
                    maxiter, opts, info, NULL,
                    <double*>covr.data, <void*>py_func)
            else:
                niter = dlevmar_blec_dif(
                    callback_func,
                    <double*>p.data, <double*>y.data, m, n,
                    <double*>lb.data, <double*>ub.data,
                    <double*>A_.data, <double*>b_.data, k1, NULL,
                    maxiter, opts, info, NULL,
                    <double*>covr.data, <void*>py_func)
        else:
            ## Linear equation constrained minimization
            if jacf is not None:
                niter = dlevmar_lec_der(
                    callback_func, callback_jacf,
                    <double*>p.data, <double*>y.data, m, n,
                    <double*>A_.data, <double*>b_.data, k1,
                    maxiter, opts, info, NULL,
                    <double*>covr.data, <void*>py_func)
            else:
                niter = dlevmar_lec_dif(
                    callback_func,
                    <double*>p.data, <double*>y.data, m, n,
                    <double*>A_.data, <double*>b_.data, k1,
                    maxiter, opts, info, NULL,
                    <double*>covr.data, <void*>py_func)
    elif bounds is not None:
        ## Box-constrained minimization
        lb, ub = verify_bc(bounds, m)
        if jacf is not None:
            niter = dlevmar_bc_der(
                callback_func, callback_jacf,
                <double*>p.data, <double*>y.data, m, n,
                <double*>lb.data, <double*>ub.data,
                maxiter, opts, info, NULL,
                <double*>covr.data, <void*>py_func)
        else:
            niter = dlevmar_bc_dif(
                callback_func,
                <double*>p.data, <double*>y.data, m, n,
                <double*>lb.data, <double*>ub.data,
                maxiter, opts, info, NULL,
                <double*>covr.data, <void*>py_func)
    else:
        ## Unconstrained minimization
        if jacf is not None:
            niter = dlevmar_der(
                callback_func, callback_jacf,
                <double*>p.data, <double*>y.data, m, n,
                maxiter, opts, info, NULL, <double*>covr.data, <void*>py_func)
        else:
            niter = dlevmar_dif(
                callback_func,
                <double*>p.data, <double*>y.data, m, n,
                maxiter, opts, info, NULL, <double*>covr.data, <void*>py_func)

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
            raise LMUserFuncError("Stopped by invalid values (NaN or Inf) "
                                  "returned by `func`")
        else:
            raise LMRuntimeError

    return output
