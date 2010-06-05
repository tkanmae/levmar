#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
"""
TODO:
    * Run `nosetest` as an Git pre-commit hook.
"""
import sys
import os

root_dir = 'levmar'
version = '0.10'
src_dir = os.path.join(root_dir, 'src')
lib_dir = os.path.join(src_dir, 'levmar-2.5')


def extention_src():
    """Return a list containing the paths to the extension source files"""
    src = ['_levmar.c']
    src = [os.path.join(src_dir, f) for f in src]
    include_dirs = [src_dir, lib_dir]
    return src, include_dirs


def library_src():
    """Return a list containing the paths to the C-library source files"""
    src = ('lm.c', 'Axb.c', 'misc.c', 'lmlec.c',
           'lmbc.c', 'lmblec.c', 'lmbleic.c')
    src = [os.path.join(lib_dir, f) for f in src]
    return src


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info

    config = Configuration('levmar', parent_package, top_path,
                           package_path=root_dir)

    # -- Add `levmar` C library
    config.add_library('levmar', sources=library_src())

    # -- Add `levmar` extension module.
    src, inc_dirs = extention_src()
    lapack_opts = get_info('lapack_opt')
    config.add_extension('_levmar',
                         sources=src,
                         include_dirs=inc_dirs,
                         libraries=['levmar'],
                         extra_info=lapack_opts)

    # -- Add `tests` directory.
    config.add_data_dir(('tests', os.path.join(root_dir, 'tests')))

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(version       = version,
          author        = 'Takeshi Kaname',
          author_email  = 'tkanmae@gmail.com',
          keywords      = ['numpy', 'data', 'science'],
          configuration = configuration
         )
