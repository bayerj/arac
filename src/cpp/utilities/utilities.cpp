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


void block_permutation(std::vector<int>& perm, 
                       std::vector<int>& sequence_shape,
                       std::vector<int>& block_shape)
{
    assert(sequence_shape.size() == block_shape.size());
    for (int i = 0; i < sequence_shape.size(); i++)
    {
        assert(sequence_shape[i] % block_shape[i] == 0);
    }

    int dim = sequence_shape.size();
    
    // Make a vector that contains the multiplied sizes of previous dimensions.
    std::vector<int> shape_dims;
    shape_dims.push_back(1);
    for (int i = 0; i < dim; i++)
    {
        shape_dims.push_back(shape_dims[i] * sequence_shape[i]);
    }
    
    int& length = shape_dims.back();
    
    std::vector< std::vector<int> > coords;
    for(int i = dim - 1; i >= 0; i--)
    {
        std::vector<int> this_dims;
        int maxindex = sequence_shape[i] / block_shape[i];
        int chunklength = block_shape[i] * shape_dims[i];
        int n_chunks = length / chunklength;
        for (int j = 0; j < n_chunks; j++)
        {
            int value = j % maxindex;
            for (int k = 0; k < chunklength; k++)
            {
                this_dims.push_back(value);
            }
        }
        coords.push_back(this_dims);
    }
    
    int n_blocks = 1;
    std::vector<int>::iterator intiter, intiter_;
    for (intiter = sequence_shape.begin(), intiter_ = block_shape.begin(); 
         intiter != sequence_shape.end();
         intiter++, intiter_++)
    {
        n_blocks *=  *intiter / *intiter_;
    }
    
    std::vector< std::vector<int> > blocks;
    blocks.reserve(n_blocks);
    for (int i = 0; i < n_blocks; i++)
    {
        std::vector<int> empty;
        blocks.push_back(empty);
    }
    
    std::vector<int> block_dims;
    block_dims.push_back(1);
    for (int i = 0; i < dim; i++)
    {
        block_dims.push_back(block_dims[i] * block_shape[i]);
    }
    
    for(int i = 0; i < length; i++)
    {
        int block_index = 0;
        for (int j = 0; j < dim; j++)
        {
            int block_steps = shape_dims[dim - j - 1] / block_dims[dim - j - 1];
            block_index +=block_steps * coords[j][i];
        }
        blocks[block_index].push_back(i);
    }
    
    std::vector< std::vector<int> >::iterator intveciter;
    for (intveciter = blocks.begin();
         intveciter != blocks.end();
         intveciter++)
    {
        for (intiter = intveciter->begin();
             intiter != intveciter->end();
             intiter++)
        {
            perm.push_back(*intiter);
        }
    }
}                      


} } // Namespace