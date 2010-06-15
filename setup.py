#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2010 Takeshi Kanmae
# ----------------------------------------------------------------------
import sys
import os

root_dir = 'lvmr'
version = '0.10'
src_dir = os.path.join(root_dir, 'src')
lib_dir = 'levmar-2.5'


def extention_src():
    """Return a list containing the paths to the extension source
    files"""
    src = ['_lvmr.c']
    src = [os.path.join(src_dir, f) for f in src]
    include_dirs = [src_dir, lib_dir]
    return src, include_dirs


def library_src():
    """Return a list containing the paths to the C levmar library source
    files"""
    src = ('lm.c', 'Axb.c', 'misc.c', 'lmlec.c',
           'lmbc.c', 'lmblec.c', 'lmbleic.c')
    src = [os.path.join(lib_dir, f) for f in src]
    return src


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info

    config = Configuration('lvmr', parent_package, top_path,
                           package_path=root_dir)

    ## Add `levmar` C library
    config.add_library('levmar', sources=library_src())

    ## Add `levmar` extension module.
    src, inc_dirs = extention_src()
    lapack_opts = get_info('lapack_opt')
    config.add_extension('_lvmr',
                         sources=src,
                         include_dirs=inc_dirs,
                         libraries=['levmar'],
                         extra_info=lapack_opts)

    ## Add `tests` directory.
    config.add_data_dir(('tests', os.path.join(root_dir, 'tests')))

    return config


if __name__ == '__main__':
    try:
        import setuptools
    except ImportError:
        pass
    from numpy.distutils.core import setup

    setup(configuration = configuration,
          version       = version,
          author        = 'Takeshi Kanmae',
          author_email  = 'tkanmae@gmail.com',
          keywords      = ['numpy', 'data', 'science'],
          tests_require = ['nose >= 0.11'],
          test_suite    = 'nose.collector',
         )
