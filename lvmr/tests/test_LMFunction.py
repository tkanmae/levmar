#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
from __future__ import division
import numpy as np
from numpy.testing import *

from lvmr._lvmr import (_LMFunction, __py_verify_funcs, LMUserFuncError, )


def test_valid_func1():
    m, n = 2, 50
    p = np.array([1, 2])
    x = np.linspace(-5, 5, n)
    func = lambda p, x : p[0]*x + p[1]

    lm_func = _LMFunction(func, args=(x,))
    assert_(__py_verify_funcs(lm_func, p, m, n))


def test_valid_func2():
    m, n = 2, 50
    p = np.array([1, 2])
    x = np.linspace(-5, 5, n)
    func = lambda p, x : p[0]*x + p[1]
    def jacf(p, x):
        y = np.empty((n,m))
        y[:,0] = x
        y[:,1] = 1
        return y

    lm_func = _LMFunction(func, args=(x,),  jacf=jacf)
    assert_(__py_verify_funcs(lm_func, p, m, n))


def test_valid_method():
    m, n = 2, 50
    p = np.array([1, 2])
    x = np.linspace(-5, 5, n)

    class Foo(object):
        def func(self, p, x):
            return p[0]*x + p[1]

    lm_func = _LMFunction(Foo().func, args=(x,))
    assert_(__py_verify_funcs(lm_func, p, m, n))


def test_valid_class_method():
    m, n = 2, 50
    p = np.array([1, 2])
    x = np.linspace(-5, 5, n)

    class Foo(object):
        @classmethod
        def func(cls, p, x):
            return p[0]*x + p[1]

    lm_func = _LMFunction(Foo.func, args=(x,))
    assert_(__py_verify_funcs(lm_func, p, m, n))


def test_func_not_callable():
    m, n = 2, 50
    p = np.array([1, 2])
    x = np.linspace(-5, 5, n)
    func = []

    lm_func = _LMFunction(func, args=(x,))
    assert_raises(LMUserFuncError, __py_verify_funcs, lm_func, p, m, n)


def test_func_invalid_args_given():
    m, n = 2, 50
    p = np.array([1, 2])
    x = np.linspace(-5, 5, n)
    func = lambda p, x : p[0]*x + p[1]

    lm_func = _LMFunction(func, args=())
    assert_raises(LMUserFuncError, __py_verify_funcs, lm_func, p, m, n)


def test_func_returns_invalid_size():
    m, n = 2, 50
    p = np.array([1, 2])
    x = np.linspace(-5, 5, n)
    func = lambda p, x : p[0]*x + p[1]

    lm_func = _LMFunction(func, args=(x,))
    assert_raises(LMUserFuncError, __py_verify_funcs, lm_func, p, m, n+1)


def test_jacf_returns_invalid_size():
    m, n = 2, 50
    p = np.array([1, 2])
    x = np.linspace(-5, 5, n)
    func = lambda p, x : p[0]*x + p[1]
    def jacf(p, x):
        y = np.empty((n+1,m))
        y[:,0] = x
        y[:,1] = 1
        return y

    lm_func = _LMFunction(func, args=(x,),  jacf=jacf)
    assert_raises(LMUserFuncError, __py_verify_funcs, lm_func, p, m, n)
