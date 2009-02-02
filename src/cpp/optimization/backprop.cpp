// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include "backprop.h"
#include "../structure/parametrized.h"


using arac::optimization::Backprop;
using arac::structure::Parametrized;

    
Backprop::Backprop(Network& network, Dataset& dataset) :
    _network(network),
    _dataset(dataset),
    _learningrate(0.001)
{
    // TODO: assert that dimensions are correct.
}


Backprop::~Backprop()
{
    
}


void 
Backprop::process_sample(const double* input_p, const double* target_p)
{
    const double* output_p = network().activate(input_p);
    double* error_p = new double[network().outsize()];
    for (int i = 0; i < network().outsize(); i++)
    {
        error_p[i] = input_p[i] - target_p[i];
    }
    network().back_activate(error_p);
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



