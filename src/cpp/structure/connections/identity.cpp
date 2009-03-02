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
IdentityConnection::_forward()
{
    if (timestep() - get_recurrent() < 0)
    {
        return;
    }

    int indim = _incomingstop - _incomingstart;
    int outdim = _outgoingstop - _outgoingstart;

    double* sink_p = _outgoing_p->input()[timestep()] + _outgoingstart;
    double* source_p = _incoming_p->output()[timestep() - get_recurrent()];
    source_p += _incomingstart;
    
    int size = (_incomingstop - _incomingstart);
    for(int i = 0; i < size; i++)
    {
        sink_p[i] += source_p[i];
    }
}


void
IdentityConnection::_backward()
{
    int this_timestep = timestep() - 1;
    if (this_timestep + get_recurrent() > sequencelength())
    {
        return;
    }
    
    int size = _incomingstop - _incomingstart;
    double* sink_p = incoming()->outerror()[this_timestep] \
                        + _incomingstart;
    
    double* source_p = outgoing()->inerror()[this_timestep + get_recurrent()] \
                          + _outgoingstart;
                          
    for(int i = 0; i < size; i++)
    {
        sink_p[i] += source_p[i];
    }
}