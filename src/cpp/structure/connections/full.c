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


void FullConnection::forward()
{
    // This will be zero for non-recurrent networks.
    int timestep = _timestep;
    
    // Buffer incrementers with respect to time.
    if (_recurrent) 
    {
        // Move on timestep back if the connection is a recurrent one.
        timestep -= 1;
    }
    
    int indim = _incomingstop - _incomingstart;
    int outdim = _outgoingstop - _outgoingstart;

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
                _incoming_p->output().current() + _incomingstart,
                // ??? some incrementer
                1,                      
                // Scalar of the target vector
                1.0,                    
                // Pointer to the target vector
                _outgoing_p->input().current() + _outgoingstart,
                // ??? some incrementer
                1);   
}


void FullConnection::backward()
{
    int indim = _incomingstop - _incomingstart;
    int outdim = _outgoingstop - _outgoingstart;
    
    double* inerror_p = _incoming_p->outerror().current() + _incomingstart;
    double* outerror_p = _outgoing_p->inerror().current() + _outgoingstart;
    double* input_p = _incoming_p->output().current();

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
