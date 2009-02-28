// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <iostream>
#include <cstring>

#include "errorscaling.h"


using arac::structure::modules::ErrorScalingLayer;


ErrorScalingLayer::ErrorScalingLayer(int size, std::vector<double> scale) :
    LinearLayer(size),
    _scale(scale)
{
    assert(scale.size() == size);
}

    
ErrorScalingLayer::~ErrorScalingLayer() 
{
    
}


void
ErrorScalingLayer::_backward()
{
    double* outerror_p = outerror()[timestep() - 1];
    double* output_p =  output()[timestep() - 1];
    double* inerror_p = inerror()[timestep() - 1];
    for (int i = 0; i < _insize; i++)
    {
        inerror_p[i] += _scale[i] * outerror_p[i];
    }
}