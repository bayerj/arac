// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>
#include <typeinfo>

#include "backprop.h"
#include "../structure/parametrized.h"


using arac::optimization::Backprop;
using arac::structure::Parametrized;

    
Backprop::Backprop(Network& network, Dataset& dataset) :
    _network(network),
    _dataset(dataset),
    _learningrate(0.001)
{
    _network.sort();
    assert(_network.insize() == _dataset.inputsize());
    assert(_network.outsize() == _dataset.targetsize());
    _error_p = new double[_network.outsize()];
}


Backprop::~Backprop()
{
    delete[] _error_p;
}


void 
Backprop::process_sample(const double* input_p, const double* target_p)
{
    const double* output_p = network().activate(input_p);
    for (int i = 0; i < network().outsize(); i++)
    {
        _error_p[i] = target_p[i] - output_p[i];
    }
    network().back_activate(_error_p);
}
        

void
Backprop::train_stochastic()
{
    int index = rand() % dataset().size();
    
    double* input_p = dataset()[index];
    double* target_p = input_p + dataset().inputsize();
    network().clear();
    process_sample(input_p, target_p);
    learn();
}


void 
Backprop::learn()
{
    std::vector<Parametrized*>::const_iterator param_iter;
    for (param_iter = network().parametrizeds().begin();
         param_iter != network().parametrizeds().end();
         param_iter++)
    {
        double* params_p = (*param_iter)->get_parameters();
        double* derivs_p = (*param_iter)->get_derivatives();
        for (int i = 0; i < (*param_iter)->size(); i++)
        {
            params_p[i] += _learningrate * derivs_p[i];
        }
    }
}



