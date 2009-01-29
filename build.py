#! /usr/bin/env python2.6
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import glob
import os
import sys

from setuptools import setup, find_packages
from distutils.ccompiler import new_compiler


def compile_arac():
    if not (sys.platform.startswith('linux') or sys.platform == 'darwin'):
        raise AracCompileError('No support for arac on platform %s yet.' 
                               % sys.platform)
    
    globs = ('src/cpp/*.cpp', 
             'src/cpp/common/*.cpp', 
             'src/cpp/structure/*.cpp',  
             'src/cpp/structure/connections/*.cpp',  
             'src/cpp/structure/modules/*.cpp',  
             'src/cpp/structure/networks/*.cpp')
    sources = sum((glob.glob(i) for i in globs), [])

    compiler_cmd = 'g++'
    executables = {
        'preprocessor': None,
        'compiler': [compiler_cmd],
        'compiler_so': [compiler_cmd],
        'compiler_cxx': [compiler_cmd],
        'linker_so': [compiler_cmd, "-shared"],
        'linker_exe': [compiler_cmd],
        'archiver': ["ar", "-cr"],
        'ranlib': None,
    }

    compiler = new_compiler(verbose=1)
    compiler.set_executables(**executables)
    compiler.add_include_dir('/usr/local/include')
    compiler.add_include_dir('/usr/include')
    compiler.add_include_dir('/sw/include')
    compiler.add_include_dir('/sw/lib')
    compiler.add_library_dir('/usr/local/lib')
    compiler.add_library_dir('/usr/lib')
    output_dir = '.'
        
    compiler.add_library('m')
    compiler.add_library('blas')
    compiler.add_library('c')
    compiler.add_library('stdc++')
    objects = compiler.compile(sources, extra_postargs=['-g', '-O3'])
    
    extra_postargs = ['-dynamiclib'] if sys.platform == 'darwin' else []
    
    compiler.link_shared_lib(objects=objects, 
                             output_libname='arac', 
                             target_lang='c++', 
                             output_dir=output_dir,
                             extra_postargs=extra_postargs)
