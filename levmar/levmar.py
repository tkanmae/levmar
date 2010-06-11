#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
import _levmar
from _levmar import (Output, LMError, LMRuntimeError, LMUserFuncError,
                     LMWarning, _LM_MAXITER, _LM_EPS1, _LM_EPS2, _LM_EPS3)


class Data(object):
    """The Data class stores the data to fit.

    Attributes
    ----------
    x : array_like
    y : array_like, shape (n,)
    """
    __slots__ = ['x', 'y', 'wt']
    def __init__(self, x, y, wt=None):
        self.x = x
        self.y = y
        self.wt = wt


class Model(object):
    """The Model class stores information about the model.

    Attributes
    ----------
    func : callable
        A function or method taking, at least, one length of n vector
        and returning a length of m vector.  The signature must be like
        `func(p, x, args) -> y`.
    jacf : callable, optional
        A function or method to compute the Jacobian of `func`.  The
        signature must be like `jacf(p, x, args)`.  If this is None, a
        approximated Jacobian will be used.
    args : tuple, optional
        Extra arguments passed to `func` (and `jacf`) in this tuple.
    """
    __slot__ = ['func', 'jacf', 'args']
    def __init__(self, func, jacf=None, args=()):
        self.func = func
        self.jacf = jacf
        self.args = args


class Levmar(object):
    """
    Attributes
    ----------
    data : Data
        The Data object
    model : Model
        The Model object

    Methods
    -------
    run() : Run the fitting.
    """
    __slots__ = ['data', 'model']
    def __init__(self, data, model):
        if isinstance(data, Data):
            self.data = data
        else:
            raise TypeError("`data` must be `levmar.Data` object")
        if isinstance(model, Model):
            self.model = model
        else:
            raise TypeError("`model` must be `levmar.Model` object")

    def run(self, p0, bounds=None, A=None, b=None, C=None, d=None,
            mu=1e-3, eps1=_LM_EPS1, eps2=_LM_EPS2, eps3=_LM_EPS3,
            maxiter=1000, cntdif=False):
        """Run the fitting.

        Parameters
        ----------
        p0 : array_like, shape (m,)
            The initial estimate of the parameters.
        bounds : tuple/list, length m
            Box-constraints. Each constraint can be a None or a tuple of two
            float/Nones.  None in the first case means no constraint, and
            None in the second case means -Inf/+Inf.
        A : array_like, shape (k1,m), optional
            A linear equation constraints matrix
        b : array_like, shape (k1,), optional
            A right-hand equation linear constraint vector
        C : array_like, shape (k2,m), optional
            A linear inequality constraints matrix
        d : array_like, shape (k2,), optional
            A right-hand linear inequality constraint vector
        mu : float, optional
            The scale factor for initial \mu
        eps1 : float, optional
            The stopping threshold for ||J^T e||_inf
        eps2 : float, optional
            The stopping threshold for ||Dp||_2
        eps3 : float, optional
            The stopping threshold for ||e||_2
        maxiter : int, optional
            The maximum number of iterations.
        cntdif : {True, False}, optional
            If this is True, the Jacobian is approximated with central
            differentiation.

        Returns
        -------
        output : levmar.Output
            The output of the minimization
        """
        args = (self.data.x,) + self.model.args
        output = _levmar.levmar(
            self.model.func, p0, self.data.y, args, self.model.jacf,
            bounds, A, b, C, d, mu, eps1, eps2, eps3, maxiter, cntdif)
        return output


def levmar(func, p0, y, args=(), jacf=None,
           bounds=None, A=None, b=None, C=None, d=None,
           mu=1e-3, eps1=_LM_EPS1, eps2=_LM_EPS2, eps3=_LM_EPS3,
           maxiter=1000, cntdif=False):
    return _levmar.levmar(func, p0,  y, args, jacf,
                          bounds, A, b, C, d,
                          mu, eps1, eps2, eps3, maxiter, cntdif)
levmar.__doc__ = _levmar.levmar.__doc__
