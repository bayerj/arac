// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include "module.h"

using arac::structure::modules::Module;


Module::Module(int insize, int outsize, bool error_agnostic) : 
    _insize(insize),
    _outsize(outsize),
    _input(insize),
    _output(outsize),
    _inerror(insize),
    _outerror(outsize),
    _error_agnostic(error_agnostic),
    _sequential(false)
{   
}


Module::Module(int insize, int outsize) :
    _insize(insize),
    _outsize(outsize),
    _input(insize),
    _output(outsize),
    _inerror(insize),
    _outerror(outsize),
    _error_agnostic(false),
    _sequential(false)
{   
}


Module::~Module() {}