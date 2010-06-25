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
    niter = info[2]
    ## The number of the degree of freedom
    ndf = y.size - p.size
    ## The standard deviation in the best-fit parameters
    p_stdv = np.sqrt(np.diag(covr))
    ## The correlation coefficients of the best-fit parameters
    corr = np.corrcoef(covr)
    ## The coefficient of determination
    r2 = 1 - np.sum((y-func(p, *args))**2) / np.sum((y-y.mean())**2)

    return Output(p, p_stdv, covr, corr, r2, niter, ndf, info)
