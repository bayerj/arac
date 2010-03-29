// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <cstring>
#include <cmath>

#include "softmax.h"


using arac::structure::modules::SoftmaxLayer;


void
SoftmaxLayer::_forward()
{
    double sum = 0;
    double* input_p = input()[timestep()];
    double* output_p = output()[timestep()];
    for(int i = 0; i < _insize; i++)
    {
        // Clip of input argument if its to extreme to avoid NaNs and inf as a
        // result of exp().
        double inpt;
        inpt = input_p[i] < -500 ? -500 : input_p[i];
        inpt = inpt > 500 ? 500 : inpt;
        double item = exp(inpt);

        sum += item;
        output_p[i] = item;
    }
    for(int i = 0; i < _insize; i++)
    {
        output_p[i] /= sum;
    }
}


void
SoftmaxLayer::_backward()
{
    int size = _outsize * sizeof(double);
    void* sourcebuffer_p = (void*) outerror()[timestep() - 1];
    void* sinkbuffer_p = (void*) inerror()[timestep() - 1];
    memcpy(sinkbuffer_p, sourcebuffer_p, size);
}
