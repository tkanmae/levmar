#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
from core import (levmar, Output, LMError, LMRuntimeError, LMUserFuncError, LMWarning,)


__version__ = '0.1.0'


## Add test function to the package.
from numpy.testing import Tester as __Tester
test = __Tester().test
