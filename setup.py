#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
import sys
import os
from os.path import join as pjoin

package_path = 'lvmr'
version = '0.10'
library_dir = 'levmar-2.5'


def get_extension_sources():
    src = ('_levmar.c',)
    return [pjoin(package_path, f) for f in src]


def get_extension_include_dirs():
    return [package_path, library_dir]


def get_library_sources():
    src = ('lm.c', 'Axb.c', 'misc.c', 'lmlec.c', 'lmbc.c', 'lmblec.c',
           'lmbleic.c',)
    src = [pjoin(library_dir, f) for f in src]
    return src


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info

    config = Configuration('lvmr',
                           parent_package,
                           top_path,
                           package_path=package_path)

    ## Add `levmar` C library
    config.add_library('levmar',
                       sources=get_library_sources())

    ## Add `levmar` extension module.
    config.add_extension('_levmar',
                         sources=get_extension_sources(),
                         include_dirs=get_extension_include_dirs(),
                         libraries=['levmar'],
                         extra_info=get_info('lapack_opt'),)

    ## Add `tests` directory.
    config.add_data_dir(('tests', pjoin(package_path, 'tests')))

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(configuration = configuration,
          version       = version,
          author        = 'Takeshi Kanmae',
          author_email  = 'tkanmae@gmail.com',
          keywords      = ['numpy', 'data', 'science'],
         )
