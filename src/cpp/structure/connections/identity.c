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
    if ((_recurrent) && (_timestep == 0))
    {
        // Don't use recurrent cons in the first timestep.
        return;
    }
    
    double* sourcebuffer_p = _recurrent ? _incoming_p->output()[_timestep - 1] :
                                          _incoming_p->output()[_timestep];
    sourcebuffer_p += _incomingstart;
    
    double* sinkbuffer_p = _outgoing_p->input()[_timestep] + _outgoingstart;
    int size = (_incomingstop - _incomingstart) * sizeof(double);
    memcpy((void*) sinkbuffer_p, (void*) sourcebuffer_p, size);
}


void
IdentityConnection::_backward()
{
    if ((_recurrent) && (_incoming_p->last_timestep()))
    {
        return;
    }
    
    double* sinkbuffer_p = _recurrent ? _incoming_p->outerror()[timestep() - 1] : 
                                        _incoming_p->outerror()[timestep()];                                
    sinkbuffer_p += _incomingstart;

    double* sourcebuffer_p = _outgoing_p->inerror()[timestep()] + _outgoingstart;
    int size = (_incomingstop - _incomingstart) * sizeof(double);
    memcpy((void*) sinkbuffer_p, (void*) sourcebuffer_p, size);
}