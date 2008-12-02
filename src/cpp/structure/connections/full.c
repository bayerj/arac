// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <iostream>
#include "full.h"

extern "C"
{
    #include "cblas.h"
}


using arac::structure::connections::FullConnection;
using arac::structure::connections::Connection;
using arac::structure::Parametrized;
using arac::structure::modules::Module;


FullConnection::FullConnection(Module* incoming_p, Module* outgoing_p) :
    Connection(incoming_p, outgoing_p),
    Parametrized(incoming_p->outsize() * outgoing_p->insize())
{
    
}


FullConnection::FullConnection(Module* incoming_p, Module* outgoing_p,
                               int incomingstart, int incomingstop, 
                               int outgoingstart, int outgoingstop) :
    Connection(incoming_p, outgoing_p, 
               incomingstart, incomingstop, 
               outgoingstart, outgoingstop),
    Parametrized(incoming_p->outsize() * outgoing_p->insize())
{
    
}


FullConnection::FullConnection(Module* incoming_p, Module* outgoing_p,
               double* parameters_p, double* derivatives_p,
               int incomingstart, int incomingstop, 
               int outgoingstart, int outgoingstop) :
    Connection(incoming_p, outgoing_p, 
               incomingstart, incomingstop,
               outgoingstart, outgoingstop),
    Parametrized((incomingstop - incomingstart) * (outgoingstop - outgoingstart))           
{
}   

            
FullConnection::~FullConnection()
{
    if (parameters_owner())
    {
        delete _parameters_p;
        _parameters_p = 0;
    }
    if (derivatives_owner())
    {
        delete _derivatives_p;
        _derivatives_p = 0;
    }
}


void FullConnection::_forward()
{
    if ((_recurrent) && (_timestep == 0))
    {
        return;
    }

    int indim = _incomingstop - _incomingstart;
    int outdim = _outgoingstop - _outgoingstart;
    
    double* sink_p = _outgoing_p->input()[_timestep] + _incomingstart;
    double* source_p = _recurrent ? _incoming_p->output()[_timestep - 1] :
                                    _incoming_p->output()[_timestep];
    source_p += _outgoingstart;

    cblas_dgemv(CblasRowMajor, 
                // Transpose the matrix since we want to multiply from the right
                CblasNoTrans,
                // Dimensions of the matrix
                outdim,        
                indim,
                // Scalar for the matrix
                1.0,                    
                // Pointer to the matrix
                get_parameters(),    
                // Dimension of the vector
                indim,
                // Pointer to the vector
                source_p,
                // Some incrementer.
                1,                      
                // Scalar of the target vector
                1.0,                    
                // Pointer to the target vector
                sink_p,
                // Incrementer.
                1);   
}


void FullConnection::_backward()
{
    if (_outgoing_p->last_timestep())
    {
        return;
    }
    
    int indim = _incomingstop - _incomingstart;
    int outdim = _outgoingstop - _outgoingstart;
    
    double* inerror_p = _recurrent ? _incoming_p->outerror()[_timestep - 1] :
                                     _incoming_p->outerror()[_timestep];
    inerror_p += _incomingstart;
    double* outerror_p = _outgoing_p->inerror()[_timestep] + _outgoingstart;
    double* input_p = _recurrent ? _incoming_p->output()[_timestep - 1] :
                                   _incoming_p->output()[_timestep];

    // TODO: use BLAS for this.
    double* weights_p = get_parameters();
    for (int i = 0; i < outdim; i++)
    {
        for (int j = 0; j < indim; j++)
        {
            inerror_p[j] += *weights_p * outerror_p[i];
            weights_p++;
        }
    }

    double* derivs_p = get_derivatives();
    for (int i = 0; i < outdim; i++)
    {
        for (int j = 0; j < indim; j++)
        {
            *derivs_p += outerror_p[i] * input_p[j];
            derivs_p++;
        }
    }
}
