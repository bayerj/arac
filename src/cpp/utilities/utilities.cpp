// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include "utilities.h"
#include <iostream>

using namespace arac::structure::networks;
using namespace arac::structure;
using namespace arac::datasets;


namespace arac {
    
namespace utilities {
    
    
void
print_array(const double* array, int length)
{
    for (int i = 0; i < length; i++)
    {
        std::cout << array[i] << " ";
    }
}
    
    
void
print_parameters(arac::structure::networks::Network& net)
{
    std::vector<Parametrized*>::const_iterator param_p_iter;
    for(param_p_iter = net.parametrizeds().begin();
        param_p_iter != net.parametrizeds().end();
        param_p_iter++)
    {
        print_array((*param_p_iter)->get_parameters(), (*param_p_iter)->size());
    }
}


void
print_derivatives(arac::structure::networks::Network& net)
{
    std::vector<Parametrized*>::const_iterator param_p_iter;
    for(param_p_iter = net.parametrizeds().begin();
        param_p_iter != net.parametrizeds().end();
        param_p_iter++)
    {
        print_array((*param_p_iter)->get_derivatives(), (*param_p_iter)->size());
    }
}


void
print_activations(Network& net, SupervisedDataset<double*, double*>& ds)
{
    for (int i = 0; i < ds.size(); i++)
    {
        net.clear();
        const double* prediction_p = net.activate(ds[i].first);
        const double* input_p = ds[i].first;
        const double* target_p = ds[i].second;

        double* error_p = new double[net.outsize()];
        for (int j = 0; j < net.outsize(); j++)
        {
            error_p[j] = target_p[j] - prediction_p[j];
        }
        
        std::cout << "Input: ";
        print_array(input_p, ds.samplesize());

        std::cout << std::endl << "Target: ";
        print_array(target_p, ds.targetsize());
        
        std::cout << std::endl << "Prediction: ";
        print_array(prediction_p, net.outsize());
        
        std::cout << std::endl << "Error: ";
        print_array(error_p, net.outsize());
        
        std::cout << std::endl << std::endl;
    }
}


} } // Namespace