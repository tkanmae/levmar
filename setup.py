#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup
from distutils.extension import Extension
import numpy as np
from numpy.distutils.system_info import get_info


levmar_sources = [
    'levmar/_levmar.c',
    'levmar-2.5/lm.c',
    'levmar-2.5/Axb.c',
    'levmar-2.5/misc.c',
    'levmar-2.5/lmlec.c',
    'levmar-2.5/lmbc.c',
    'levmar-2.5/lmblec.c',
    'levmar-2.5/lmbleic.c'
]


setup(
    name='levmar',
    version='0.1.0',
    license='GNU General Public Licence v2',
    maintainer='Takeshi Kanmae',
    maintainer_email='tkanmae@gmail.com',
    classifiers=[
        'Intentended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programing Language :: Python',
        'Licence :: OSI Approved :: MIT License',
    ],
    install_requires=[
        'numpy>=1.6.2',
    ],
    packages=[
        'levmar',
    ],
    ext_modules=[
        Extension(
            'levmar._levmar',
            sources=levmar_sources,
            include_dirs=['levmar-2.5', np.get_include()],
            **get_info('lapack_opt')
        ),
    ],
    test_suite='nose.collector',
)
