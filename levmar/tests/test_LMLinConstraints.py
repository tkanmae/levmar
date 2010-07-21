#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
from __future__ import division
import numpy as np
from numpy.testing import *

from levmar._core import _LMLinConstraints


def test_valid_list1():
    m = 2
    A = [1, 2]
    b = [1]
    assert_(isinstance(_LMLinConstraints(A, b, m), _LMLinConstraints))


def test_valid_list2():
    m = 3
    A = [[1, 2, 3], [1, 2, 3]]
    b = [1, 2]
    assert_(isinstance(_LMLinConstraints(A, b, m), _LMLinConstraints))


def test_valid_list2():
    m = 3
    A = [1, 2, 3, 1, 2, 3]
    b = [1, 2]
    assert_(isinstance(_LMLinConstraints(A, b, m), _LMLinConstraints))


def test_valid_ndarray1():
    m = 2
    A = np.array([1, 2])
    b = np.array([1])
    assert_(isinstance(_LMLinConstraints(A, b, m), _LMLinConstraints))


def test_valid_ndarray2():
    m = 3
    A = np.array([[1, 2, 3], [1, 2, 3]])
    b = np.array([1, 2])
    assert_(isinstance(_LMLinConstraints(A, b, m), _LMLinConstraints))


def test_valid_ndarray3():
    m = 3
    A = np.array([1, 2, 3, 1, 2, 3])
    b = np.array([1, 2])
    assert_(isinstance(_LMLinConstraints(A, b, m), _LMLinConstraints))


def test_invalid_size_list1():
    m = 3
    A = [[1, 2], [1, 2]]
    b = [1, 2, 3]
    assert_raises(ValueError, _LMLinConstraints, A, b, m)


def test_invalid_size_list2():
    m = 3
    A = [[1, 2, 3], [1, 2, 3]]
    b = [1, 2, 3, 4]
    assert_raises(ValueError, _LMLinConstraints, A, b, m)


def test_invalid_shaped_list1():
    m = 3
    A = [[1, 2], [1, 2], [1, 2]]
    b = [1, 2, 3]
    assert_raises(ValueError, _LMLinConstraints, A, b, m)


def test_invalid_shaped_list2():
    m = 2
    A = [1, 2]
    b = [[1]]
    assert_raises(ValueError, _LMLinConstraints, A, b, m)


def test_invalid_shaped_list3():
    m = 2
    A = [1, [2]]
    b = [1]
    assert_raises(ValueError, _LMLinConstraints, A, b, m)


def test_invalid_size_ndarray1():
    m = 3
    A = np.array([[1, 2], [1, 2]])
    b = np.array([1, 2, 3])
    assert_raises(ValueError, _LMLinConstraints, A, b, m)


def test_invalid_size_ndarray2():
    m = 3
    A = np.array([[1, 2, 3], [1, 2, 3]])
    b = np.array([1, 2, 3, 4])
    assert_raises(ValueError, _LMLinConstraints, A, b, m)


def test_invalid_shaped_ndarray1():
    m = 3
    A = np.array([[1, 2], [1, 2], [1, 2]])
    b = np.array([1, 2, 3])
    assert_raises(ValueError, _LMLinConstraints, A, b, m)


def test_invalid_shaped_ndarray2():
    m = 2
    A = np.array([1, 2])
    b = np.array([[1]])
    assert_raises(ValueError, _LMLinConstraints, A, b, m)


def test_invalid_shaped_ndarray3():
    m = 2
    A = np.array([1, 2])
    b = np.array([[1]])
    assert_raises(ValueError, _LMLinConstraints, A, b, m)


def test_invalid_value1():
    m = 2
    A = [[], []]
    b = [1]
    assert_raises(ValueError, _LMLinConstraints, A, b, m)


def test_invalid_value2():
    m = 2
    A = [1, 2]
    b = []
    assert_raises(ValueError, _LMLinConstraints, A, b, m)


def test_invalid_value3():
    m = 2
    A = [1, 'a']
    b = [1]
    assert_raises(ValueError, _LMLinConstraints, A, b, m)


def test_invalid_value4():
    m = 2
    A = [1, 2]
    b = ['a']
    assert_raises(ValueError, _LMLinConstraints, A, b, m)


if __name__ == '__main__':
    run_module_suite()
