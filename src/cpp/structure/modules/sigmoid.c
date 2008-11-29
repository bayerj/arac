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
SigmoidLayer::forward()
{
    double* input_p = input().current();
    double* output_p = output().current();
    for (int i = 0; i < _insize; i++)
    {
        *output_p = sigmoid(*input_p);
        output_p++;
        input_p++;
    }
}


void
SigmoidLayer::backward()
{
    double* outerror_p = outerror().current();
    double* output_p =  output().current();
    double* inerror_p = inerror().current();
    for (int i = 0; i < _insize; i++)
    {
        inerror_p[i] += output_p[i] * (1 - output_p[i]) * outerror_p[i];
    }
}