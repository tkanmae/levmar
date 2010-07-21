#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
from __future__ import division
import numpy as np
from numpy.testing import *

from levmar._core import (_LMFunction, __py_verify_funcs, LMUserFuncError, )


def test_valid_func1():
    m, n = 3, 50
    p = np.array([1, 2, 3])
    x = np.linspace(-5, 5, n)
    func = lambda p, x : p[0]*np.exp(-p[1]*x) + p[2]

    lm_func = _LMFunction(func, args=(x,))
    assert_(__py_verify_funcs(lm_func, p, m, n))


def test_valid_func2():
    m, n = 3, 50
    p = np.array([1, 2, 3])
    x = np.linspace(-5, 5, n)
    func = lambda p, x : p[0]*np.exp(-p[1]*x) + p[2]
    def jacf(p, x):
        y = np.empty((n,m))
        y[:,0] = np.exp(-p[1]*x)
        y[:,1] = -p[0] * x * np.exp(-p[1]*x)
        y[:,2] = np.ones(x.size)
        return y

    lm_func = _LMFunction(func, args=(x,),  jacf=jacf)
    assert_(__py_verify_funcs(lm_func, p, m, n))


def test_valid_method():
    m, n = 3, 50
    p = np.array([1, 2, 3])
    x = np.linspace(-5, 5, n)

    class Foo(object):
        def func(self, p, x):
            return p[0]*np.exp(-p[1]*x) + p[2]

    lm_func = _LMFunction(Foo().func, args=(x,))
    assert_(__py_verify_funcs(lm_func, p, m, n))


def test_valid_class_method():
    m, n = 3, 50
    p = np.array([1, 2, 3])
    x = np.linspace(-5, 5, n)

    class Foo(object):
        @classmethod
        def func(cls, p, x):
            return p[0]*np.exp(-p[1]*x) + p[2]

    lm_func = _LMFunction(Foo.func, args=(x,))
    assert_(__py_verify_funcs(lm_func, p, m, n))


def test_func_not_callable():
    m, n = 3, 50
    p = np.array([1, 2, 3])
    x = np.linspace(-5, 5, n)
    func = []

    lm_func = _LMFunction(func, args=(x,))
    assert_raises(LMUserFuncError, __py_verify_funcs, lm_func, p, m, n)


def test_func_invalid_args_given():
    m, n = 3, 50
    p = np.array([1, 2, 3])
    x = np.linspace(-5, 5, n)
    func = lambda p, x : p[0]*np.exp(-p[1]*x) + p[2]

    lm_func = _LMFunction(func, args=())
    assert_raises(LMUserFuncError, __py_verify_funcs, lm_func, p, m, n)


def test_func_returns_invalid_size():
    m, n = 3, 50
    p = np.array([1, 2, 3])
    x = np.linspace(-5, 5, n)
    func = lambda p, x : p[0]*np.exp(-p[1]*x) + p[2]

    lm_func = _LMFunction(func, args=(x,))
    assert_raises(LMUserFuncError, __py_verify_funcs, lm_func, p, m, n+1)


def test_jacf_returns_invalid_size():
    m, n = 3, 50
    p = np.array([1, 2, 3])
    x = np.linspace(-5, 5, n)
    func = lambda p, x : p[0]*np.exp(-p[1]*x) + p[2]
    def jacf(p, x):
        y = np.empty((n,m-1))
        y[:,0] = np.exp(-p[1]*x)
        y[:,1] = -p[0] * x * np.exp(-p[1]*x)
        return y

    lm_func = _LMFunction(func, args=(x,),  jacf=jacf)
    assert_raises(LMUserFuncError, __py_verify_funcs, lm_func, p, m, n)
