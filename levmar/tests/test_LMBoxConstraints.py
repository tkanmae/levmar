#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
from __future__ import division
import numpy as np
from numpy.testing import *

from levmar._core import _LMBoxConstraints


def test_valid1():
    m = 2
    bounds = [(0, 1), (2, 3)]
    assert_(isinstance(_LMBoxConstraints(bounds, m), _LMBoxConstraints))


def test_valid2():
    m = 2
    bounds = [(0, None), None]
    assert_(isinstance(_LMBoxConstraints(bounds, m), _LMBoxConstraints))


def test_valid3():
    m = 2
    bounds = [(None, 0), None]
    assert_(isinstance(_LMBoxConstraints(bounds, m), _LMBoxConstraints))


def test_valid4():
    m = 2
    bounds = [(0, np.inf), None]
    assert_(isinstance(_LMBoxConstraints(bounds, m), _LMBoxConstraints))


def test_valid5():
    m = 2
    bounds = [(-np.inf, 0), None]
    assert_(isinstance(_LMBoxConstraints(bounds, m), _LMBoxConstraints))


def test_invalid_shape1():
    m = 2
    bounds = [(0, 1), (2, 3), (3, 4)]
    assert_raises(ValueError, _LMBoxConstraints, bounds, m)


def test_invalid_shape2():
    m = 2
    bounds = [(0, 1)]
    assert_raises(ValueError, _LMBoxConstraints, bounds, m)


def test_invalid_value1():
    m = 2
    bounds = [1, 2]
    assert_raises(ValueError, _LMBoxConstraints, bounds, m)


def test_invalid_value2():
    m = 2
    bounds = [1, (2, 3)]
    assert_raises(ValueError, _LMBoxConstraints, bounds, m)


def test_invalid_value3():
    m = 2
    bounds = [(), (2, 3)]
    assert_raises(ValueError, _LMBoxConstraints, bounds, m)



if __name__ == '__main__':
    run_module_suite()
