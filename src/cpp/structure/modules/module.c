// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include "module.h"


using arac::structure::modules::Module;
using arac::structure::Component;
using arac::common::Buffer;


Module::Module() : 
    _insize(0),
    _outsize(0),
    _input_p(0),
    _output_p(0),
    _inerror_p(0),
    _outerror_p(0)
{
    init_buffers();
}



Module::Module(int insize, int outsize) : 
    _insize(insize),
    _outsize(outsize),
    _input_p(0),
    _output_p(0),
    _inerror_p(0),
    _outerror_p(0)
{
    init_buffers();
}


Module::~Module() {
    free_buffers(); 
}


void
Module::expand()
{
    input().expand();
    output().expand();
    if (!(error_agnostic()))
    {
        inerror().expand();
        outerror().expand();
    }
}


void
Module::forward()
{
    Component::forward();
    if (sequential())
    {
        expand();
    }
}


void 
Module::free_buffers()
{
    if (_input_p != 0)
    {
        delete _input_p;
        _input_p = 0;
    }
    
    if (_output_p != 0)
    {
        delete _output_p;
        _output_p = 0;
    }
    
    if (!error_agnostic())
    {
        if (_inerror_p != 0)
        {
            delete _inerror_p;
            _inerror_p = 0;
        }
        if (_outerror_p != 0)
        {
            delete _outerror_p;
            _outerror_p = 0;
        }
    }
}


void
Module::init_buffers()
{
    free_buffers();
    _input_p = new Buffer(_insize);
    _output_p = new Buffer(_outsize);
    if (!error_agnostic())
    {
        _inerror_p = new Buffer(_insize);
        _outerror_p = new Buffer(_outsize);
    }
}
