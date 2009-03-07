// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include "connection.h"

using arac::structure::connections::Connection;

Connection::Connection(Module* incoming_p, Module* outgoing_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop) :
    _incoming_p(incoming_p),
    _outgoing_p(outgoing_p),
    _incomingstart(incomingstart),
    _incomingstop(incomingstop),
    _outgoingstart(outgoingstart),
    _outgoingstop(outgoingstop),
    _recurrent(false)
{
    
}                

Connection::Connection(Module* incoming_p, Module* outgoing_p) :
    _incoming_p(incoming_p),
    _outgoing_p(outgoing_p),
    _incomingstart(0),
    _incomingstop(incoming_p->outsize()),
    _outgoingstart(0),
    _outgoingstop(outgoing_p->insize()),
    _recurrent(false)
{
    
}


Connection::~Connection()
{
    
}


void
Connection::_forward()
{
    if (timestep() - get_recurrent() < 0)
    {
        return;
    }

    // In the case of recurrent connections, all modules have already been 
    // forwarded and thus have equal timesteps. In the case of non-recurrent
    // connections, we have to subtract one from the incoming modules timestep.
    int decr = get_recurrent() ? 0 : 1;

    double* sink_p = _outgoing_p->input()[_outgoing_p->timestep()] + _outgoingstart;
    double* source_p = _incoming_p->output()[_incoming_p->timestep() - decr - get_recurrent()];
    assert(source_p != 0);
    assert(sink_p != 0);

    source_p += _incomingstart;

    forward_process(sink_p, source_p);
}


void
Connection::_backward()
{

    int this_timestep = timestep() - 1;
    if (this_timestep + get_recurrent() > sequencelength())
    {
        return;
    }
    
    double* sink_p = incoming()->outerror()[this_timestep] \
                     + _incomingstart;
    
    double* source_p = outgoing()->inerror()[this_timestep + get_recurrent()] \
                       + _outgoingstart;

    
    backward_process(sink_p, source_p);
}


