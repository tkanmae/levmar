#!/usr/bin/env python
# -*- coding: utf-8 -*-
from core import (levmar, Output)


__version__ = '0.1.0'


# Add test function to the package.
from numpy.testing import Tester as __Tester
test = __Tester().test
