// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>

#include "blockpermutation.h"
#include "../../utilities/utilities.h"


using arac::structure::connections::BlockPermutationConnection;
using arac::structure::connections::Connection;
using arac::structure::modules::Module;
using arac::utilities::block_permutation;


BlockPermutationConnection::BlockPermutationConnection(
    Module* incoming_p, Module* outgoing_p, 
    std::vector<int> sequence_shape,
    std::vector<int> block_shape) :
    PermutationConnection(incoming_p, outgoing_p)
{
    // TODO: Check that modules are of the same size and that sequence and block
    // shapes divide each other nicely.
    std::vector<int> permutation;
    block_permutation(permutation, sequence_shape, block_shape);
    set_permutation(permutation);
}


BlockPermutationConnection::~BlockPermutationConnection()
{
    
}
