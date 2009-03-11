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
    assert(_incomingstart <= incoming()->outsize());
    assert(_incomingstop <= incoming()->outsize());
    assert(_outgoingstart <= outgoing()->insize());
    assert(_outgoingstop <= outgoing()->insize());
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
    // In the case of recurrent connections, all modules have already been
    // forwarded and thus have equal timesteps. In the case of non-recurrent
    // connections, we have to subtract one from the incoming modules timestep.
    int decr = get_recurrent() ? 0 : 1;

    if ((timestep() - get_recurrent() < 0) ||
        (_incoming_p->timestep() - decr - get_recurrent() < 0))
    {
        return;
    }

    double* sink_p = _outgoing_p->input()[_outgoing_p->timestep()];
    assert(sink_p != 0);
    sink_p += _outgoingstart;

    double* source_p = _incoming_p->output()[_incoming_p->timestep() - decr - get_recurrent()];
    assert(source_p != 0);
    source_p += _incomingstart;

    forward_process(sink_p, source_p);
}


void
Connection::_backward()
{
    // Analoguous to decr in the forward pass.
    int incr = get_recurrent() ? 0 : 1;

    if ((timestep() - 1 + get_recurrent() > sequencelength()) ||
        (outgoing()->timestep() - 1 + incr + get_recurrent() > sequencelength()))
    {
        return;
    }

    int sinkidx = incoming()->timestep() - 1;
    double* sink_p = incoming()->outerror()[sinkidx];
    assert(sink_p != 0);
    sink_p += _incomingstart;

    int srcidx = outgoing()->timestep() - 1 + incr + get_recurrent();
    double* source_p = outgoing()->inerror()[srcidx];
    assert(source_p != 0);
    source_p += _outgoingstart;

    backward_process(sink_p, source_p);
}


