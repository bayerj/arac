// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>
#include <cstring>

#include "backprop.h"
#include "../structure/parametrized.h"


using arac::optimization::SimpleBackprop;
using arac::optimization::SemiSequentialBackprop;
using arac::optimization::SequentialBackprop;
using arac::structure::Parametrized;


SimpleBackprop::SimpleBackprop(BaseNetwork& network, 
               SupervisedDataset<double*, double*>& dataset) :
               Backprop<double*, double*>(network, dataset) {}   
           
SimpleBackprop::~SimpleBackprop() {}


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


void
SimpleBackprop::process_sample(double* sample_p, double* target_p,
                               double* importance_p)
{
    const double* output_p = network().activate(sample_p);
    for (int i = 0; i < network().outsize(); i++)
    {
        _error_p[i] = (target_p[i] - output_p[i]) * importance_p[i];
    }
    network().back_activate(_error_p);
}


SemiSequentialBackprop::SemiSequentialBackprop(
    BaseNetwork& network,
    SupervisedDataset<Sequence, double*>& dataset) :
    Backprop<Sequence, double*>(network, dataset) 
{
    _output_p = new double[dataset.targetsize()];
}


SemiSequentialBackprop::~SemiSequentialBackprop() {}


void
SemiSequentialBackprop::process_sample(Sequence input, double* target_p)
{
    memset(_output_p, 0, sizeof(double) * dataset().targetsize());

    // Process the sequence and sum the outputs.
    for (int i = 0; i < input.length(); i++)
    {
        const double* output_p = network().activate(input[i]);
        // Sum the outputs 
        for (int j = 0; j < dataset().targetsize(); j++)
        {
            _output_p[j] += output_p[j];
        }
    }
    // Calculate the overall error.
    for (int i = 0; i < network().outsize(); i++)
    {
        _error_p[i] = target_p[i] - _output_p[i];
    }
    // Backpropagate the error.
    for (int i = 0; i < input.length(); i++)
    {
        network().back_activate(_error_p);
    }
}


void
SemiSequentialBackprop::process_sample(Sequence input, double* target_p, 
                                       double* importance_p)
{
    memset(_output_p, 0, sizeof(double) * dataset().targetsize());

    // Process the sequence and sum the outputs.
    for (int i = 0; i < input.length(); i++)
    {
        const double* output_p = network().activate(input[i]);
        // Sum the outputs 
        for (int j = 0; j < dataset().targetsize(); j++)
        {
            _output_p[j] += output_p[j];
        }
    }
    // Calculate the overall error.
    for (int i = 0; i < network().outsize(); i++)
    {
        _error_p[i] = (target_p[i] - _output_p[i]) * importance_p[i];
    }
    // Backpropagate the error.
    for (int i = 0; i < input.length(); i++)
    {
        network().back_activate(_error_p);
    }
}


SequentialBackprop::SequentialBackprop(
    BaseNetwork& network, 
    SupervisedDataset<Sequence, Sequence>& dataset) :
    Backprop<Sequence, Sequence>(network, dataset) {}


SequentialBackprop::~SequentialBackprop() {}


void 
SequentialBackprop::process_sample(Sequence input, Sequence target)
{
    // Process the sequence and save the outputs.
    for (int i = 0; i < input.length(); i++)
    {
        const double* output_p = network().activate(input[i]);
        _outputs.push_back(output_p);
    }
    // Iterate over the outputs and targets in parallel.
    std::vector<const double*>::const_reverse_iterator output_iter;
    int i = target.length() - 1;
    for (output_iter = _outputs.rbegin(); 
         output_iter != _outputs.rend(); 
         output_iter++)
    {
        // Calculate the error for this timestep.
        const double* output_p = *output_iter;
        const double* target_p = target[i];
        for (int j = 0; j < dataset().targetsize(); j++)
        {
            _error_p[j] = target_p[j] - output_p[j];
        }
        network().back_activate(_error_p);
        i -= 1;
    }
}
 

void 
SequentialBackprop::process_sample(Sequence input, Sequence target, 
                                   Sequence importance)
{
    // Process the sequence and save the outputs.
    for (int i = 0; i < input.length(); i++)
    {
        const double* output_p = network().activate(input[i]);
        _outputs.push_back(output_p);
    }
    // Iterate over the outputs and targets in parallel.
    std::vector<const double*>::const_reverse_iterator output_iter;
    int i = target.length() - 1;
    for (output_iter = _outputs.rbegin(); 
         output_iter != _outputs.rend();
         output_iter++)
    {
        // Calculate the error for this timestep.
        const double* output_p = *output_iter;
        const double* target_p = target[i];
        for (int j = 0; j < dataset().targetsize(); j++)
        {
            _error_p[j] = (target_p[j] - output_p[j]) * importance[i][j];
        }
        network().back_activate(_error_p);
        i -= 1;
    }
}
