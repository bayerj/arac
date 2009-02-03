// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>
#include <typeinfo>

#include "backprop.h"
#include "../structure/parametrized.h"


using arac::optimization::SimpleBackprop;
using arac::structure::Parametrized;


SimpleBackprop::SimpleBackprop(Network& network, 
               SupervisedDataset<double*, double*>& dataset) :
               Backprop<double*, double*>(network, dataset)
{
    
}   
           
SimpleBackprop::~SimpleBackprop()
{
    
}


void
SimpleBackprop::process_sample(double* sample_p, double* target_p)
{
    const double* output_p = network().activate(sample_p);
    for (int i = 0; i < network().outsize(); i++)
    {
        _error_p[i] = target_p[i] - output_p[i];
    }
    network().back_activate(_error_p);
}