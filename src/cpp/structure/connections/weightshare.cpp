// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>

#include "weightshare.h"


using arac::structure::connections::WeightShareConnection;
using arac::structure::connections::Connection;
using arac::structure::Parametrized;
using arac::structure::modules::Module;


WeightShareConnection::WeightShareConnection(Module* incoming_p, Module* outgoing_p, 
                                             int inchunk, int outchunk) :
    Connection(incoming_p, outgoing_p),
    Parametrized(inchunk * outchunk),
    _con(incoming_p, outgoing_p, 0, inchunk, 0, outchunk),
    _inchunk(inchunk),
    _outchunk(outchunk)
{
    assert(incoming()->outsize() / inchunk == outgoing()->insize() / outchunk);
    assert(incoming()->outsize() % inchunk == 0);
    assert(outgoing()->insize() % outchunk == 0);

    _n_chunks = incoming()->outsize() / inchunk;
}


WeightShareConnection::~WeightShareConnection()
{
    
}



void
WeightShareConnection::_forward()
{
    if (timestep() - get_recurrent() < 0)
    {
        return;
    }

    // This has to be done everytime, since it might change in the meantime. An
    // alternative would be to make get_parameters/get_derivatives virtual, but
    // that'd probably hit performance.
    _con.set_parameters(get_parameters());
    _con.set_derivatives(get_derivatives());
    
    for (int i = 0; i < _n_chunks; i++)
    {
        _con.set_incomingstart(i * _inchunk);
        _con.set_incomingstop((i + 1) * _inchunk);
        _con.set_outgoingstart(i * _outchunk);
        _con.set_outgoingstop((i + 1) * _outchunk);
        _con.forward();
    }
}


void
WeightShareConnection::_backward()
{
    int this_timestep = timestep() - 1;
    if (this_timestep + get_recurrent() > sequencelength())
    {
        return;
    }
    
    // See _forward for explanation.
    _con.set_parameters(get_parameters());
    _con.set_derivatives(get_derivatives());
    
    for (int i = 0; i < _n_chunks; i++)
    {
        _con.set_incomingstart(i * _inchunk);
        _con.set_incomingstop((i + 1) * _inchunk);
        _con.set_outgoingstart(i * _outchunk);
        _con.set_outgoingstop((i + 1) * _outchunk);
        _con.backward();
    }
}