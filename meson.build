project('levmar', ['c', 'cython'],
    default_options : ['warning_level=2'])

py = import('python').find_installation(pure: false)
numpy_dep = dependency('numpy', required: true, version: '>=2.0.0')
lapack_dep = dependency('lapack', required : true)

levmar_include_dir = include_directories('levmar-2.6')


levmar_c_sources = [
    'levmar/_levmar.pyx',
    'levmar-2.6/lm.c',
    'levmar-2.6/Axb.c',
    'levmar-2.6/misc.c',
    'levmar-2.6/lmlec.c',
    'levmar-2.6/lmbc.c',
    'levmar-2.6/lmblec.c',
    'levmar-2.6/lmbleic.c'
] 

py.extension_module(
    '_levmar',
    levmar_c_sources,
    install: true,
    dependencies : [numpy_dep, lapack_dep ],
    include_directories : levmar_include_dir,
    subdir: 'levmar'
)


py.install_sources(
    [
        'levmar/__init__.py'
    ],
    subdir: 'levmar',
    pure: false
)

