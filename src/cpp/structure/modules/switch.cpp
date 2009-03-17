// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>
#include <cstring>

#include "switch.h"
#include "../../common/functions.h"


using arac::common::sigmoid;
using arac::common::sigmoidprime;

using arac::structure::modules::SwitchLayer;


void
SwitchLayer::_forward()
{
    // Shortcuts.
    int size = insize();
    double* input_p = input()[timestep()];
    double* output_p = output()[timestep()];

    for(int i = 0; i < size; i++)
    {
        output_p[i] = sigmoid(input_p[i]);
        output_p[i + size] = 1 - sigmoid(input_p[i]);
    }
}


void
SwitchLayer::_backward()
{
    // Shortcuts.
    int size = insize();
    double* inerror_p = inerror()[timestep() - 1];
    double* outerror_p = outerror()[timestep() - 1];
    double* input_p = input()[timestep() - 1];

    for (int i = 0; i < size; i++)
    {
        inerror_p[i] = sigmoidprime(input_p[i]) * outerror_p[i];
        inerror_p[i] -= sigmoidprime(input_p[i]) * outerror_p[i + size];
    }
}
