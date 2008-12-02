// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <iostream>
#include <cstring>

#include "sigmoid.h"
#include "../../common/functions.h"


using arac::structure::modules::SigmoidLayer;
using arac::common::sigmoid;
using arac::common::sigmoidprime;


void
SigmoidLayer::_forward()
{
    double* input_p = input()[_timestep];
    double* output_p = output()[_timestep];
    for (int i = 0; i < _insize; i++)
    {
        *output_p = sigmoid(*input_p);
        output_p++;
        input_p++;
    }
}


void
SigmoidLayer::_backward()
{
    double* outerror_p = outerror()[_timestep];
    double* output_p =  output()[_timestep];
    double* inerror_p = inerror()[_timestep];
    for (int i = 0; i < _insize; i++)
    {
        inerror_p[i] += output_p[i] * (1 - output_p[i]) * outerror_p[i];
    }
}