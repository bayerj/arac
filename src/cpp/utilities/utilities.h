// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_UTILITIES_UTILITIES_INCLUDED
#define Arac_UTILITIES_UTILITIES_INCLUDED

#include "../structure/networks/networks.h"
#include "../datasets/datasets.h"


namespace arac {
namespace utilities {


void
print_parameters(arac::structure::networks::BaseNetwork& net);


void
print_derivatives(arac::structure::networks::BaseNetwork& net);


void
print_activations(arac::structure::networks::BaseNetwork& net,
                  arac::datasets::SupervisedDataset<double*, double*>& ds);


void block_permutation(std::vector<int>& perm,
                       std::vector<int>& sequence_shape,
                       std::vector<int>& block_shape);


///
/// Fill the sink with random values from the interval [-interval, interval]. 
/// interval defaults to 0.01.
///
void fill_random(double* sink_p, int length, double interval = 0.01);


///
/// Perform a numerical gradient check on a network. The returned value is the
/// biggest difference between an analytical and a numerical gradient.
///
double
gradient_check(arac::structure::networks::BaseNetwork& network,
               bool verbose=false);


///
/// Recursively walk a BaseNetwork topology and push all encountered
/// Parametrized objects to the end of a vector.
///
void
parametrized_by_network(std::vector<arac::structure::Parametrized*>& params, 
                        arac::structure::networks::BaseNetwork& net);



} } // Namespace


#endif
