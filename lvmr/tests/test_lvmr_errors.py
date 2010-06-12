#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
from __future__ import division
import numpy as np
from numpy.testing import *

import lvmr
from lvmr import LMUserFuncError


class TestFuncErrors(TestCase):

    def setUp(self):
        self.x = np.arange(10, dtype=np.float64)
        self.y = np.arange(10, dtype=np.float64)
        self.func = lambda p, x: p[0]*x + p[1]
        self.p0 = (1, 1)

    def test_not_callable(self):
        invalid_funcs = (1, 'foo', [1, 2], (1, 2), {'foo': 1})
        for f in invalid_funcs:
            ## `func`
            assert_raises(TypeError,
                          lvmr.levmar, f, self.p0, self.y,
                          args=(self.x,))
            ## `jacf`
            assert_raises(TypeError,
                          lvmr.levmar, self.func, self.p0, self.y,
                          args=(self.x,), jacf=f)

    def test_invalid_args(self):
        assert_raises(LMUserFuncError,
                      lvmr.levmar, self.func, self.p0, self.y)
        assert_raises(LMUserFuncError,
                      lvmr.levmar, self.func, self.p0, self.y,
                      args=(self.x, ()))

    def test_return_invalid_size(self):
        x = np.arange(5, dtype=np.float64)
        assert_raises(LMUserFuncError,
                      lvmr.levmar, self.func, self.p0, self.y, args=(x,))


class TestBCErrors(TestCase):

    def setUp(self):
        self.x = np.arange(10, dtype=np.float64)
        self.y = np.arange(10, dtype=np.float64)
        self.func = lambda p, x: p[0]*x + p[1]
        self.p0 = (1, 1)

    def test_not_valid_type(self):
        invalid_bounds = [0, 'bounds']
        for bounds in invalid_bounds:
            assert_raises(TypeError,
                          lvmr.levmar, self.func, self.p0, self.y,
                          args=(self.x,), bounds=bounds)

    def test_not_valid_size(self):
        invalid_bounds = [
            [None,],
            [None, None, None,],
            [(0,2), (0,2), (0,2)],
            [(0,2), (0,2,3)],
            [tuple(), (0,2)]]
        for bounds in invalid_bounds:
            assert_raises(ValueError,
                          lvmr.levmar, self.func, self.p0, self.y,
                          args=(self.x,), bounds=bounds)

    def test_not_valid_value(self):
        invalid_bounds = [[None, (None,'upper')],]
        for bounds in invalid_bounds:
            assert_raises(ValueError,
                          lvmr.levmar, self.func, self.p0, self.y,
                          args=(self.x,), bounds=bounds)


class TestLCErrors(TestCase):

    def setUp(self):
        self.x = np.arange(10, dtype=np.float64)
        self.y = np.arange(10, dtype=np.float64)
        self.func = lambda p, x: p[0]*x*x + p[1]*x + p[2]
        self.p0 = (1, 1, 1)

    def test_not_defined(self):
        func = lambda p, x: p[0]*x
        p0 = (1,)
        A = [1]
        b = [1]
        assert_raises(ValueError,
                      lvmr.levmar, func, p0, self.y, args=(self.x,),
                      A=A, b=b)
        assert_raises(ValueError,
                      lvmr.levmar, func, p0, self.y, args=(self.x,),
                      C=A, d=b)
        assert_raises(ValueError,
                      lvmr.levmar, func, p0, self.y, args=(self.x,),
                      A=A, b=b, C=A, d=b)

    def test_not_valid_type(self):
        invalid_As = [0, 'A', ]
        invalid_bs = [0, 'd', ]
        for A in invalid_As:
            for b in invalid_bs:
                assert_raises(ValueError,
                              lvmr.levmar, self.func, self.p0, self.y,
                              args=(self.x,), A=A, b=b)
                assert_raises(ValueError,
                              lvmr.levmar, self.func, self.p0, self.y,
                              args=(self.x,), C=A, d=b)
                assert_raises(ValueError,
                              lvmr.levmar, self.func, self.p0, self.y,
                              args=(self.x,), A=A, b=b, C=A, d=b)

    def test_non_finite(self):
        invalid_As = [
            [1,2,np.nan], [1,2,np.inf], [1,2,None], [1,2,3], [1,2,3], [1,2,3]]
        invalid_bs = [
            [1], [1], [1], [np.nan], [np.inf], [None]]
        for A, b in zip(invalid_As, invalid_bs):
            assert_raises(ValueError,
                          lvmr.levmar, self.func, self.p0, self.y,
                          args=(self.x,), A=A, b=b)
            assert_raises(ValueError,
                          lvmr.levmar, self.func, self.p0, self.y,
                          args=(self.x,), C=A, d=b)
            assert_raises(ValueError,
                          lvmr.levmar, self.func, self.p0, self.y,
                          args=(self.x,), A=A, b=b, C=A, d=b)

    def test_not_valid_size(self):
        invalid_As = [
            [[1,2], [1,2]],
            [[1,2,3], [1,2,3]]]
        invalid_bs = [
            [[1],[1],[1]],
            [1]]
        for A, b in zip(invalid_As, invalid_bs):
            assert_raises(ValueError,
                          lvmr.levmar, self.func, self.p0, self.y,
                          args=(self.x,), A=A, b=b)
            assert_raises(ValueError,
                          lvmr.levmar, self.func, self.p0, self.y,
                          args=(self.x,), C=A, d=b)
            assert_raises(ValueError,
                          lvmr.levmar, self.func, self.p0, self.y,
                          args=(self.x,), A=A, b=b, C=A, d=b)


if __name__ == '__main__':
    run_module_suite()
