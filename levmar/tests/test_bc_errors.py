#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from numpy.testing import *

from levmar import levmar
from levmar import LMUserFuncError


class TestBCErrors(TestCase):

    def test_func_not_callable(self):
        x = np.arange(10, dtype=np.float64)
        p0 = (1, 1)
        args = ()

        invalid_funcs = (1, 'foo', [1, 2], (1, 2), {'foo': 1})
        for func in invalid_funcs:
            assert_raises(TypeError, levmar.bc, func, p0, x, args)

    def test_jacf_not_callable(self):
        x = np.arange(10, dtype=np.float64)
        func = lambda p: p[0]*x + p[1]
        p0 = (1, 1)
        args = ()

        invalid_jacfs = (1, 'foo', [1, 2], (1, 2), {'foo': 1})
        for jacf in invalid_jacfs:
            kw = {'jacf': jacf}
            assert_raises(TypeError, levmar.bc, func, p0, x, args, **kw)

    def test_func_type_error(self):
        x = np.arange(10, dtype=np.float64)

        func = lambda p, x: p[0]*x + p[1]
        p0 = (1, 1)
        args = ()
        assert_raises(LMUserFuncError, levmar.bc, func, p0, x, args)

    def test_func_invalid_return(self):
        x = np.arange(10, dtype=np.float64)
        args = ()
        ## Ruturn value of `func` and `x` musr have the same size.
        func = lambda p, x: p[0] * np.arange(5) + p[1]
        p0 = (1, 1)
        assert_raises(LMUserFuncError, levmar.bc, func, p0, x, args)

    def test_bounds_not_valid_type(self):
        x = np.arange(10, dtype=np.float64)
        func = lambda p: p[0]*x + p[1]
        p0 = (1, 1)
        args = ()

        invalid_bounds = [0, (0, 2), ((0,2), 2)]
        for bounds in invalid_bounds:
            assert_raises(TypeError, levmar.bc, func, p0, x, args, bounds)

    def test_bounds_not_valid_size(self):
        x = np.arange(10, dtype=np.float64)
        func = lambda p: p[0]*x + p[1]
        p0 = (1, 1)
        args = ()

        invalid_bounds = [
            (None,),
            ((0,2), (0,2), (0,2))]
        for bounds in invalid_bounds:
            assert_raises(ValueError, levmar.bc, func, p0, x, args, bounds)


if __name__ == '__main__':
    run_module_suite()
