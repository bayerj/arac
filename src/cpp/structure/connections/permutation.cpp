// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>

#include "permutation.h"


using arac::structure::connections::PermutationConnection;
using arac::structure::connections::Connection;
using arac::structure::modules::Module;


PermutationConnection::PermutationConnection(
    Module* incoming_p, Module* outgoing_p) : 
    Connection(incoming_p, outgoing_p)
{
    
}


PermutationConnection::PermutationConnection(Module* incoming_p, Module* outgoing_p, 
                                             std::vector<int> permutation) :
    Connection(incoming_p, outgoing_p),
    _permutation(permutation)
{
    // TODO: Check that modules are of the same size and that permutation is
    // correct.
    assert(incoming_p->outsize() == outgoing_p->insize());
    assert(incoming_p->outsize() == this->permutation().size());
}


PermutationConnection::~PermutationConnection()
{
    
}


void
PermutationConnection::invert()
{
    std::vector<int> old = permutation();
    std::vector<int>::const_iterator permiter;
    int i = 0;
    for (permiter = old.begin(), i = 0; 
         permiter != old.end();
         permiter++, i++)
    {
        _permutation[*permiter] = i;
    }
}


void
PermutationConnection::forward_process(double* sink_p, const double* source_p)
{
    std::vector<int>::const_iterator intiter;
    int i;
    for (intiter = permutation().begin(), i = 0; 
         intiter != permutation().end();
         intiter++, i++)
    {
        sink_p[*intiter] += source_p[i];
    }
}


void
PermutationConnection::backward_process(double* sink_p, const double* source_p)
{
    std::vector<int>::const_iterator intiter;
    int i;
    for (intiter = permutation().begin(), i = 0; 
         intiter != permutation().end();
         intiter++, i++)
    {
        sink_p[i] += source_p[*intiter];
    }
}
