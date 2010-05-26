#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
"""
TODO:
    * Isolate the extension build from `levmar` library build in order
      to reduce compile time.
    * Run `nosetest` as an Git pre-commit hook.
"""
import sys
import os

root_dir = 'levmar'
version = '0.10'
src_dir = os.path.join(root_dir, 'src')


def levmar_config():
    levmar_dir = os.path.join(src_dir, 'levmar-2.5')
    sources = ('lm.c', 'Axb.c', 'misc.c', 'lmlec.c',
               'lmbc.c', 'lmblec.c', 'lmbleic.c')
    sources = [os.path.join(levmar_dir, f) for f in sources]
    include_dirs = [levmar_dir]
    return sources, include_dirs


def get_lapack_opts():
    from numpy.distutils.system_info import get_info
    return get_info('lapack_opt')


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('levmar', parent_package, top_path,
                           package_path=root_dir)

    # -- Add `_levmar` extension module.
    sources = ['levmar.c']
    sources = [os.path.join(src_dir, f) for f in sources]
    include_dirs = [src_dir]
    ## levmar sources and include_dirs
    levmar_sources, levmar_include_dirs = levmar_config()
    sources.extend(levmar_sources)
    include_dirs.extend(levmar_include_dirs)

    config.add_extension('levmar',
                         sources=sources,
                         include_dirs=include_dirs,
                         extra_info=get_lapack_opts())
    # config.add_extension('levmar',
    #                      sources=sources,
    #                      include_dirs=include_dirs,
    #                      extra_info=get_lapack_opts(),
    #                      extra_compile_args=['-g'],
    #                      extra_link_args=['-g'])

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
