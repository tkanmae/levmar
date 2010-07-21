#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import sys
from functools import partial
from collections import namedtuple
from cStringIO import StringIO
import numpy as np


_Output = namedtuple('Ouput', 'p, p_stdv, covr, corr, r2, niter, ndf, info')
class Output(_Output):
    """
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
    covr : ndarray, shape=(m,m)
        The covariance matrix corresponding to the least square fit.
    corr : ndarray, shape=(m,m)
        Pearson's correlation coefficient of the best-fit parameters.
    niter : int
        The number of the iterations
    ndf : int
        The degrees of freedom.
    info : tuple
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
    __slots__ =()
    def pprint(self, file=sys.stdout):
        """
        Parameters
        ----------
        file : file-like, optional
            An object with `write()` method
        """
        p = partial(print, file=file)
        ## Print summary of the fitting
        p("degree of freedom: {0}".format(self.ndf))
        p("iterations: {0}".format(self.info[2]))
        p("reason for stop: {0}".format(self.info[3]))
        p("")
        p(":parameters:")
        for i, (q, dq) in enumerate(zip(self.p, self.p_stdv)):
            rel = 100 * abs(dq/q)
            p("  p[{0}]: {1:+12.5g} +/- {2:>12.5g}  ({3:4.1f}%)".format(i, q, dq, rel))
        p("")
        p(":covariance:")
        p(np.array_str(self.covr, precision=2, max_line_width=200))
        p("")
        p(":correlation:")
        p(np.array_str(self.corr, precision=2, max_line_width=200))
        p("")
        p(":r2:")
        p("  {0:6g}".format(self.r2))

    def __str__(self):
        buf = StringIO()
        self.pprint(buf)
        s = buf.getvalue()
        buf.close()
        return s


def _full_output(func, p, y, args, covr, info):
    """
    Parameters
    ----------
    func : callable
        A function or method computing the model function.
    p : array_like, shape (m,)
        The best-fit parameters
    y : array_like, shape (n,)
        The dependent data, or the observation.
    args : tuple
        Extra arguments passed to `func`
    covr : ndarray, shape=(m,m)
        The covariance matrix corresponding to the least square fit.
    info : tuple
        Information regarding the minimization returned from `levmar()`.

    Returns
    -------
    output : Output
        An `Output` object
    """
    ## The number of the iterations
    niter = info[2]
    ## The number of the degrees of freedom
    ndf = y.size - p.size
    ## The standard deviation in the best-fit parameters
    p_stdv = np.sqrt(np.diag(covr))
    ## The correlation coefficients of the best-fit parameters
    corr = np.corrcoef(covr)
    ## The coefficient of determination
    r2 = 1 - np.sum((y-func(p, *args))**2) / np.sum((y-y.mean())**2)

    return Output(p, p_stdv, covr, corr, r2, niter, ndf, info)
