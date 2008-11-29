// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <cstring>

#include "tanh.h"
#include "../../common/functions.h"



using arac::structure::modules::TanhLayer;
using arac::common::tanh_;
using arac::common::tanhprime;


void
TanhLayer::forward()
{
    double* input_p = input().current();
    double* output_p = output().current();
    for (int i = 0; i < _insize; i++)
    {
        *output_p = tanh_(*input_p);
        output_p++;
        input_p++;
    }
}


void
TanhLayer::backward()
{
    double* outerror_p = outerror().current();
    double* output_p =  output().current();
    double* inerror_p = inerror().current();
    for (int i = 0; i < _insize; i++)
    {
        inerror_p[i] += (1 - output_p[i] * output_p[i]) * outerror_p[i];
    }

}