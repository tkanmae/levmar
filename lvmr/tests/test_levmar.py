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


if __name__ == '__main__':
    run_module_suite()
