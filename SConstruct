import distutils.sysconfig
import numpy.distutils.misc_util

TARGET = '/usr/local'

LIBPATH = ['/usr/lib', '/usr/local/lib', '/sw/lib', '.']
CPPPATH = ['/usr/local/include', '/sw/include', '/usr/include']

PYTHONPATH = [distutils.sysconfig.get_python_inc()]
NUMPYPATH = numpy.distutils.misc_util.get_numpy_include_dirs()



# First compile and link the library.
libenv = Environment(LIBS=['m', 'blas'], CPPPATH=CPPPATH, LIBPATH=LIBPATH,
                     SHLIBPREFIX="")
library_globs = ['src/cpp/*.cpp', 
                 'src/cpp/common/*.cpp', 
                 'src/cpp/datasets/*.cpp', 
                 'src/cpp/structure/*.cpp',  
                 'src/cpp/structure/connections/*.cpp',  
                 'src/cpp/structure/modules/*.cpp',  
                 'src/cpp/structure/networks/*.cpp',
                 'src/cpp/structure/networks/mdrnns/*.cpp']
lib = libenv.SharedLibrary('libarac.dylib', sum([Glob(i) for i in library_globs], []))
libenv.Install('/usr/local/lib', lib)
libenv.Alias('install', '/usr/local/lib')

# Then compile the tests.
testenv = Environment(LIBS=['arac', 'gtest'], CPPPATH=CPPPATH, LIBPATH=LIBPATH)
test = testenv.Program('test-arac', Glob('src/cpp/tests/*.cpp'))


swigenv = Environment(SWIGFLAGS=['-python', '-c++'],
                      CPPPATH=CPPPATH + NUMPYPATH + PYTHONPATH,
                      LIBS=['arac'],
                      CCFLAGS='-Wno-long-double -bundle -undefined suppress -flat_namespace',
                      LINKFLAGS='-Wno-long-double -undefined suppress -flat_namespace',
                      LIBPATH=LIBPATH,
                      SHLIBPREFIX="")
swig = swigenv.SharedLibrary('src/python/arac/_cppbridge.so', 
                             ['src/swig/cppbridge.i'])

# Declare some dependencies.
Depends(test, lib)
Depends(swig, test)

