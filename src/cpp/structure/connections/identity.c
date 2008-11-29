// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <cstring>
#include "identity.h"


using arac::structure::connections::Connection;
using arac::structure::connections::IdentityConnection;
using arac::structure::modules::Module;


IdentityConnection::IdentityConnection(
    Module* incoming_p, Module* outgoing_p) :
    Connection(incoming_p, outgoing_p)
{
    
}


IdentityConnection::IdentityConnection(
    Module* incoming_p, Module* outgoing_p,
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
IdentityConnection::forward()
{
    void* sourcebuffer_p = \
        (void*) (_incoming_p->output().current() + _incomingstart);
    void* sinkbuffer_p = \
        (void*) (_outgoing_p->input().current() + _outgoingstart);
    int size = (_incomingstop - _incomingstart) * sizeof(double);
    memcpy(sinkbuffer_p, sourcebuffer_p, size);
}


void
IdentityConnection::backward()
{
    void* sinkbuffer_p = \
        (void*) (_incoming_p->outerror().current() + _incomingstart);
    void* sourcebuffer_p = \
        (void*) (_outgoing_p->inerror().current() + _outgoingstart);
    int size = (_incomingstop - _incomingstart) * sizeof(double);
    memcpy(sinkbuffer_p, sourcebuffer_p, size);
}