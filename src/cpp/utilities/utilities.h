// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_UTILITIES_UTILITIES_INCLUDED
#define Arac_UTILITIES_UTILITIES_INCLUDED

#include "../structure/networks/networks.h"
#include "../datasets/datasets.h"


namespace arac {
namespace utilities {


void
print_parameters(arac::structure::networks::Network& net);


void
print_derivatives(arac::structure::networks::Network& net);


void
print_activations(arac::structure::networks::Network& net, 
                  arac::datasets::SupervisedDataset<double*, double*>& ds);


void block_permutation(std::vector<int>& perm, 
                       std::vector<int>& sequence_shape,
                       std::vector<int>& block_shape);


void fill_random(double* sink_p, int length);


double
gradient_check(arac::structure::networks::BaseNetwork& network);




} } // Namespace


#endif
