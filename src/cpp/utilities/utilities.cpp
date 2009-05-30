// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>
#include <cstdlib>

#include "utilities.h"


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
print_parameters(arac::structure::networks::BaseNetwork& net)
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
print_derivatives(arac::structure::networks::BaseNetwork& net)
{
    std::vector<Parametrized*>::const_iterator param_p_iter;
    for(param_p_iter = net.parametrizeds().begin();
        param_p_iter != net.parametrizeds().end();
        param_p_iter++)
    {
        print_array((*param_p_iter)->get_derivatives(), 
                    (*param_p_iter)->size());
    }
}


void
print_activations(BaseNetwork& net, SupervisedDataset<double*, double*>& ds)
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

// TODO: use better rng, like mt from tr1 or sth.
static int count = 0;
void fill_random(double* sink_p, int length, double interval)
{
    int seed = time(NULL) * (count + 1);
    srand(seed);
    count += length;
    for(int i = 0; i < length; i++)
    {
        double value = RAND_MAX - RAND_MAX / 2;
        value /= rand();
        sink_p[i] = value * interval;
    }
}


void
addscale(const double* first_p, const double* second_p, double* sink_p, 
         size_t length, double scale)
{
    for (int i = 0; i < length; i++)
    {
        sink_p[i] = first_p[i] + (scale * second_p[i]);
    }
}


void 
square(double* target_p, size_t length)
{
    for (int i = 0; i < length; i++)
    {
        target_p[i] *= target_p[i];
    }
}


double
sum(const double* target_p, size_t length)
{
    double result = 0;
    for (int i = 0; i < length; i++)
    {
        result += target_p[i];
    }
    return result;
}


void
parametrized_by_network(std::vector<Parametrized*>& params, BaseNetwork& net)
{
    std::vector<BaseNetwork*>::iterator net_iter;
    
    for (net_iter = net.networks().begin();
         net_iter != net.networks().end();
         net_iter++)
    {
        parametrized_by_network(params, **net_iter);
    }

    std::vector<Parametrized*>::iterator param_iter;
    for (param_iter = net.parametrizeds().begin();
         param_iter != net.parametrizeds().end();
         param_iter++)
    {
        params.push_back(*param_iter);
    }
}


double
gradient_check_nonsequential(BaseNetwork& network, bool verbose)
{
    int insize = network.insize();
    int outsize = network.outsize();
    
    // Build up an appropriate input.
    double* input_p = new double[insize];
    double* target_p = new double[outsize];
    const double* result_p;
    double* error_p = new double[outsize];
    fill_random(input_p, insize, 2.5);
    fill_random(target_p, outsize, 2.5);
    
    double epsilon = 1e-5;
    double biggest = 0.0;

    // The derivative as computed by the Parametrized object.
    double param_deriv;
    // The derivative as computed numerically.
    double numeric_deriv;
    
    // Hold pointers to all Parametrized objects here to check them 
    // sequentially.
    std::vector<Parametrized*> params;
    parametrized_by_network(params, network);

    std::vector<arac::structure::Parametrized*>::iterator param_iter;
    // Now iterate over all parametrized objects in order to play with every 
    // parameter to check derivative correctness.
    for (param_iter = params.begin();
         param_iter != params.end();
         param_iter++)
    {
        if (verbose)
        {
          std::cout << "New Parametrized instance." << std::endl;
        }
        Parametrized& parametrized = **param_iter;
        for (int i = 0; i < parametrized.size(); i++)
        {
            // First calculate analytical derivative.
            double& param = parametrized.get_parameters()[i];
            double old_param = param;
            network.clear();
            network.clear_derivatives();
            result_p = network.activate(input_p);
            
            addscale(target_p, result_p, error_p, outsize, -1);
            
            network.back_activate(error_p);
            param_deriv = parametrized.get_derivatives()[i];
            
            network.clear();

            // Calculate point to the right of the target.
            param = old_param + epsilon;
            
            result_p = network.activate(input_p);
            addscale(target_p, result_p, error_p, outsize, -1);
            square(error_p, outsize);
            // Multiply with 0.5 because of the quadratic error funtion.
            double righterror = 0.5 * sum(error_p, outsize);

            network.clear();
            
            // Calculate point to the left of the target.
            param = old_param - epsilon;

            result_p = network.activate(input_p);
            addscale(target_p, result_p, error_p, outsize, -1);
            square(error_p, outsize);
            // Multiply with 0.5 because of the quadratic error funtion.
            double lefterror = 0.5 * sum(error_p, outsize);

            numeric_deriv = (righterror - lefterror) / (epsilon * -2);
            if (verbose)
            {
              std::cout << "Numeric/Analytical:" << numeric_deriv 
                        << " <-> " 
                        << param_deriv
                        << std::endl;
            }

            double diff = numeric_deriv - param_deriv;
            diff *= diff < 0 ? -1 : 1;   // Take absolute value.
            biggest = diff > biggest ? diff : biggest;

            // Correct parameter.
            param = old_param;
        }
    }
    
    return biggest;
}


double
gradient_check(BaseNetwork& network, bool verbose)
{
    // TODO: alternative for sequential/non-sequential
    return gradient_check_nonsequential(network, verbose);
}


} } // Namespace
