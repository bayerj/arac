// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>

#include "permutation.h"


using arac::structure::connections::PermutationConnection;
using arac::structure::connections::Connection;
using arac::structure::Parametrized;
using arac::structure::modules::Module;


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
PermutationConnection::_forward()
{
    if (timestep() - get_recurrent() < 0)
    {
        return;
    }

    std::vector<int>::const_iterator intiter;
    int i;
    for (intiter = permutation().begin(), i = 0; 
         intiter != permutation().end();
         intiter++, i++)
    {
        outgoing()->input()[timestep()][*intiter] += \
            incoming()->output()[timestep()][i];
    }
}


void
PermutationConnection::_backward()
{
    int this_timestep = timestep() - 1;
    if (this_timestep + get_recurrent() > sequencelength())
    {
        return;
    }

    std::vector<int>::const_iterator intiter;
    int i;
    for (intiter = permutation().begin(), i = 0; 
         intiter != permutation().end();
         intiter++, i++)
    {
        incoming()->outerror()[this_timestep][i] += \
            outgoing()->inerror()[this_timestep][*intiter];
    }
}