// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <cstring>
#include <cmath>

#include "partialsoftmax.h"


using arac::structure::modules::PartialSoftmaxLayer;


void
PartialSoftmaxLayer::forward()
{
    double* input_p = input().current();
    double* output_p = output().current();
    int slices = _insize / _slicelength;
    for (int i = 0; i < slices; i++)
    {
        double sum = 0;
        for(int j = 0; j < _slicelength; j++)
        {
            int index = i * _slicelength + j;
            double item = exp(input_p[index]);
            item = item < -500 ? -500 : item;
            item = item > 500 ? 500 : item;
            sum += item;
            output_p[index] = item;
        }
        for(int j = 0; j < _slicelength; j++)
        {
            int index = i * _slicelength + j;
            output_p[index] /= sum;
        }
    }
}


void
PartialSoftmaxLayer::backward()
{
    int size = _outsize * sizeof(double);
    void* sourcebuffer_p = (void*) outerror().current();
    void* sinkbuffer_p = (void*) inerror().current();
    memcpy(sinkbuffer_p, sourcebuffer_p, size);
}