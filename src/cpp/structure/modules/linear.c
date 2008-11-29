// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <cstring>

#include "linear.h"


using arac::structure::modules::LinearLayer;


void
LinearLayer::forward()
{
    int size = _insize * sizeof(double);
    void* sourcebuffer_p = (void*) _input.current();
    void* sinkbuffer_p = (void*) _output.current();
    memcpy(sinkbuffer_p, sourcebuffer_p, size);
}


void
LinearLayer::backward()
{
    int size = _outsize * sizeof(double);
    void* sourcebuffer_p = (void*) _outerror.current();
    void* sinkbuffer_p = (void*) _inerror.current();
    memcpy(sinkbuffer_p, sourcebuffer_p, size);
}