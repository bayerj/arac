// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>

extern "C"
{
    #include "cblas.h"
}

#include "full.h"


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
    Parametrized((incomingstop - incomingstart) * (outgoingstop - outgoingstart))
{
    
}


FullConnection::FullConnection(Module* incoming_p, Module* outgoing_p,
               double* parameters_p, double* derivatives_p,
               int incomingstart, int incomingstop, 
               int outgoingstart, int outgoingstop) :
    Connection(incoming_p, outgoing_p, 
               incomingstart, incomingstop,
               outgoingstart, outgoingstop),
    Parametrized((incomingstop - incomingstart) * (outgoingstop - outgoingstart),
                 parameters_p, derivatives_p)           
{
}   

            
FullConnection::~FullConnection()
{
}


void
FullConnection::forward_process(double* sink_p, const double* source_p)
{
    int indim = _incomingstop - _incomingstart;
    int outdim = _outgoingstop - _outgoingstart;

    cblas_dgemv(CblasRowMajor, 
                // Do not transpose the matrix since we want to multiply from 
                // the right
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


void FullConnection::backward_process(double* sink_p, const double* source_p)
{
    int indim = _incomingstop - _incomingstart;
    int outdim = _outgoingstop - _outgoingstart;
    
    double* input_p = _incoming_p->output()[incoming()->timestep() - 1] \
                      + _incomingstart;

    cblas_dgemv(CblasColMajor, 
                // Do not transpose the matrix since we want to multiply from 
                // the right
                CblasNoTrans,
                // Dimensions of the matrix
                indim,        
                outdim,
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

    cblas_dger(CblasColMajor, 
               indim,
               outdim,
               1.0,
               input_p,
               1, 
               source_p,
               1, 
               get_derivatives(),
               indim);
}
