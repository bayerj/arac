// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>
#include <cstring>

#include "doublegate.h"
#include "../../common/functions.h"


using arac::common::sigmoid;
using arac::common::sigmoidprime;

using arac::structure::modules::DoubleGateLayer;


void
DoubleGateLayer::_forward()
{
    // Shortcuts.
    int size = insize() / 2;
    double* input_p = input()[timestep()];
    double* output_p = output()[timestep()];
    for(int i = 0, j =  size; i < size; i++, j++)
    {
        output_p[i] = sigmoid(input_p[i]) * input_p[j];
        output_p[j] = (1 - sigmoid(input_p[i])) * input_p[j];
    }
}


void
DoubleGateLayer::_backward()
{
    // Shortcuts.
    int size = insize() / 2;
    double* inerror_p = inerror()[timestep() - 1];
    double* outerror_p = outerror()[timestep() - 1];
    double* input_p = input()[timestep() - 1];

    for (int i = 0; i < size; i++)
    {
        inerror_p[i] = sigmoidprime(input_p[i]) 
                       * input_p[i + size] 
                       * outerror_p[i];
        inerror_p[i] -= sigmoidprime(input_p[i])
                        * input_p[i + size] 
                        * outerror_p[i + size];
    }
    for(int i = 0; i < size; i++)
    {
        inerror_p[i + size] = sigmoid(input_p[i]) * outerror_p[i];
        inerror_p[i + size] += (1 - sigmoid(input_p[i])) * outerror_p[i + size];
    }
}
