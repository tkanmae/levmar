#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
import os


PACKAGE_PATH = 'levmar'
LIBRARY_DIR = 'levmar-2.5'


def get_extension_sources():
    src = ('_core.c',)
    return [os.path.join(PACKAGE_PATH, f) for f in src]


def get_extension_include_dirs():
    return [PACKAGE_PATH, LIBRARY_DIR]


def get_library_sources():
    src = ('lm.c', 'Axb.c', 'misc.c', 'lmlec.c', 'lmbc.c', 'lmblec.c',
           'lmbleic.c',)
    src = [os.path.join(LIBRARY_DIR, f) for f in src]
    return src


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info

    config = Configuration('levmar',
                           parent_package,
                           top_path,
                           package_path=PACKAGE_PATH)

    ## Add `levmar` C library
    config.add_library('levmar',
                       sources=get_library_sources())

    ## Add `levmar` extension module.
    config.add_extension('_core',
                         sources=get_extension_sources(),
                         include_dirs=get_extension_include_dirs(),
                         libraries=['levmar'],
                         extra_info=get_info('lapack_opt'),)

    ## Add `tests` directory.
    config.add_data_dir(('tests', os.path.join(PACKAGE_PATH, 'tests')))

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(configuration    = configuration,
          version          = '0.1.0',
          maintainer       = 'Takeshi Kanmae',
          maintainer_email = 'tkanmae@gmail.com',
          licence          = 'GNU General Public Licence v2',)
