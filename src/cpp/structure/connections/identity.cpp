// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <cstring>
#include <iostream>

#include "identity.h"


using arac::structure::connections::Connection;
using arac::structure::connections::IdentityConnection;
using arac::structure::modules::Module;


IdentityConnection::IdentityConnection(Module* incoming_p, Module* outgoing_p) :
    Connection(incoming_p, outgoing_p)
{
    
}


IdentityConnection::IdentityConnection(Module* incoming_p, Module* outgoing_p,
                                       int incomingstart, int incomingstop, 
                                       int outgoingstart, int outgoingstop) :
    Connection(incoming_p, outgoing_p, 
               incomingstart, incomingstop, 
               outgoingstart, outgoingstop)
{
    
}


IdentityConnection::~IdentityConnection()
{
    
}


void
IdentityConnection::forward_process(double* sink_p, const double* source_p)
{
    int indim = _incomingstop - _incomingstart;
    int outdim = _outgoingstop - _outgoingstart;
    
    int size = (_incomingstop - _incomingstart);
    for(int i = 0; i < size; i++)
    {
        sink_p[i] += source_p[i];
    }
}


void
IdentityConnection::backward_process(double* sink_p, const double* source_p)
{
    int size = _incomingstop - _incomingstart;
    for(int i = 0; i < size; i++)
    {
        sink_p[i] += source_p[i];
    }
}
