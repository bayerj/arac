import sys
import distutils.sysconfig
import numpy.distutils.misc_util

# Some settings depending on the platform.
if sys.platform == 'darwin':
    libname = 'libarac.dylib'
    linkflags = '-Wno-long-double -undefined suppress -flat_namespace'
    frameworksflags = '-flat_namespace -undefined suppress'
elif sys.platform == 'linux2':
    libname = 'libarac.so'
    frameworksflags = ''
    linkflags = '' 
else:
    raise SystemExit("Cannot build on %s." % sys.platform)


SetOption('num_jobs', 4)

TARGET = '/usr/local'

LIBPATH = ['/usr/lib', '.', '/usr/local/lib', '/sw/lib']
CPPPATH = ['/usr/local/include', '/sw/include', '/usr/include']
CCFLAGS = ['-g', '-O3']
#CCFLAGS = ['-O3']

PYTHONPATH = [distutils.sysconfig.get_python_inc()]
NUMPYPATH = numpy.distutils.misc_util.get_numpy_include_dirs()


# First compile and link the library.
libenv = Environment(LIBS=['m', 'blas'], CPPPATH=CPPPATH, LIBPATH=LIBPATH,
                     SHLIBPREFIX="", CCFLAGS=CCFLAGS)
library_globs = ['src/cpp/*.cpp', 
                 'src/cpp/common/*.cpp', 
                 'src/cpp/utilities/*.cpp', 
                 'src/cpp/datasets/*.cpp', 
                 'src/cpp/optimization/*.cpp', 
                 'src/cpp/optimization/descent/*.cpp', 
                 'src/cpp/structure/*.cpp',  
                 'src/cpp/structure/connections/*.cpp',  
                 'src/cpp/structure/modules/*.cpp',  
                 'src/cpp/structure/networks/*.cpp',
                 'src/cpp/structure/networks/mdrnns/*.cpp']
lib = libenv.SharedLibrary(libname, sum([Glob(i) for i in library_globs], []))

# Then compile the tests.
testenv = Environment(LIBS=['arac', 'gtest'], CPPPATH=CPPPATH, LIBPATH=LIBPATH)
test = testenv.Program('test-arac', Glob('src/cpp/tests/*.cpp'))


swigenv = Environment(SWIGFLAGS=['-python', '-c++', '-outdir', 'src/python/arac'],
                      CPPPATH=CPPPATH + NUMPYPATH + PYTHONPATH,
                      LIBS=['arac'],
                      FRAMEWORKSFLAGS=frameworksflags,
                      LINKFLAGS=linkflags,
                      LIBPATH=LIBPATH,
                      LDMODULEPREFIX='src/python/arac/_', 
                      LDMODULESUFFIX = '.so',
                      )
swig = swigenv.LoadableModule('cppbridge', 
                             ['src/swig/cppbridge.i'])

# Declare some dependencies.
Depends(test, lib)
Depends(swig, test)

