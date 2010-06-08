#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from numpy.testing import *

import levmar
from levmar import LMUserFuncError


class TestFuncErrors(TestCase):

    def setUp(self):
        self.x = np.arange(10, dtype=np.float64)
        self.y = np.arange(10, dtype=np.float64)
        self.func = lambda p, x: p[0]*x + p[1]
        self.p0 = (1, 1)

    def test_func_not_callable(self):
        invalid_funcs = (1, 'foo', [1, 2], (1, 2), {'foo': 1})
        for func in invalid_funcs:
            assert_raises(TypeError,
                          levmar.levmar, func, self.p0, self.y,
                          args=(self.x,))

    def test_jacf_not_callable(self):
        invalid_jacfs = (1, 'foo', [1, 2], (1, 2), {'foo': 1})
        for jacf in invalid_jacfs:
            assert_raises(TypeError,
                          levmar.levmar, self.func, self.p0, self.y,
                          args=(self.x,), jacf=jacf)

    def test_func_invalid_call(self):
        assert_raises(LMUserFuncError,
                      levmar.levmar, self.func, self.p0, self.y, args=())
        foo = ()
        assert_raises(LMUserFuncError,
                      levmar.levmar, self.func, self.p0, self.y,
                      args=(self.x, foo))

    def test_func_return_invalid_size(self):
        x = np.arange(5, dtype=np.float64)
        assert_raises(LMUserFuncError,
                      levmar.levmar, self.func, self.p0, self.y, args=(x,))


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
                          levmar.levmar, self.func, self.p0, self.y,
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
                          levmar.levmar, self.func, self.p0, self.y,
                          args=(self.x,), bounds=bounds)

    def test_not_valid_value(self):
        invalid_bounds = [[None, (None,'upper')],]
        for bounds in invalid_bounds:
            assert_raises(ValueError,
                          levmar.levmar, self.func, self.p0, self.y,
                          args=(self.x,), bounds=bounds)


if __name__ == '__main__':
    run_module_suite()
