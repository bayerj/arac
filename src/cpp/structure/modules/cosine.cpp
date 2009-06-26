// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <cstring>
#include <cmath>

extern "C"
{
    #include "cblas.h"
}

#include "cosine.h"


using arac::structure::modules::CosineLayer;


CosineLayer::~CosineLayer()
{
    if (_interbuffer_p)
    {
        delete _interbuffer_p;
    }
    if (_intererror_p)
    {
        delete _intererror_p;
    }
}


void
CosineLayer::_forward()
{
   // Calculate the dot product. 
   double* input_p = input()[timestep()];
   double length = cblas_ddot(insize(), 
                              input_p, 1.0,  
                              input_p, 1.0);
   length = sqrt(length);

   double* output_p = output()[timestep()];
   for (int i = 0; i < outsize(); ++i)
   {
       output_p[i] = input_p[i] / length;
   }
}


void
CosineLayer::_backward()
{
    // Calculate the dot product. 
    double* input_p = input()[timestep() - 1];
    double dotprod = cblas_ddot(insize(), 
                                input_p, 1.0,  
                                input_p, 1.0);
    double invdotprodroot = 1 / sqrt(dotprod);
    double invdotprodrootcubed = 1 / pow(sqrt(dotprod), 3);

    double* inter_p = new double[insize()];

    for (int i = 0; i < insize(); ++i)
    {
       inter_p[i] = -pow(input_p[i], 2) * invdotprodrootcubed + invdotprodroot;
    }

    double* intererror_p = new double[insize()];
    intererror_p[insize() - 1] = 0;
    memcpy(intererror_p, outerror()[timestep() - 1], sizeof(double) * outsize());

    double* inerror_p = inerror()[timestep() - 1];
    for (int i = 0; i < insize(); ++i)
    {
        inerror_p[i] = inter_p[i] * intererror_p[i];
    }
}


void
CosineLayer::expand()
{
    _interbuffer_p->expand();
    _intererror_p->expand();
    Module::expand();
}
    
