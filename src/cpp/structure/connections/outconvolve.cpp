// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>

extern "C"
{
    #include "cblas.h"
}


#include "outconvolve.h"


using arac::structure::connections::OutConvolveConnection;
using arac::structure::connections::Connection;
using arac::structure::Parametrized;
using arac::structure::modules::Module;


OutConvolveConnection::OutConvolveConnection(Module* incoming_p, 
                                             Module* outgoing_p, 
                                             int chunk) :
    Connection(incoming_p, outgoing_p),
    Parametrized(chunk * incoming()->outsize()),
    _chunk(chunk)
{
    assert(outgoing()->insize() % chunk == 0);

    _n_chunks = outgoing()->insize() / chunk;
}


OutConvolveConnection::~OutConvolveConnection()
{
    
}


void
OutConvolveConnection::forward_process(double* sink_p, const double* source_p)
{
    for (int i = 0; i < _n_chunks; i++)
    {
        cblas_dgemv(
                CblasRowMajor, 
                // Do not transpose the matrix since we want to multiply from 
                // the right
                CblasNoTrans,
                // Dimensions of the matrix
                _chunk,
                _incoming_p->outsize(),        
                // Scalar for the matrix
                1.0,                    
                // Pointer to the matrix
                get_parameters(),    
                // Dimension of the vector
                _incoming_p->outsize(),
                // Pointer to the vector
                source_p,
                // Some incrementer.
                1,                      
                // Scalar of the target vector
                1.0,                    
                // Pointer to the target vector
                sink_p + (i * _chunk),
                // Incrementer.
                1);   
    }
}


void
OutConvolveConnection::backward_process(double* sink_p, const double* source_p)
{
    int indim = _incomingstop - _incomingstart;
    int outdim = _outgoingstop - _outgoingstart;
    
    double* input_p = _incoming_p->output()[timestep() - 1] \
                      + _incomingstart;
    double* derivs_p = get_derivatives();
    for (int i = 0; i < _n_chunks; i++)
    {
        cblas_dgemv(CblasColMajor, 
                // Do not transpose the matrix since we want to multiply from 
                // the right
                CblasNoTrans,
                // Dimensions of the matrix
                _incoming_p->outsize(),
                _chunk,        
                // Scalar for the matrix
                1.0,                    
                // Pointer to the matrix
                get_parameters(),    
                // Dimension of the vector
                _incoming_p->outsize(),
                // Pointer to the vector
                source_p + (i * _chunk),
                // Some incrementer.
                1,                      
                // Scalar of the target vector
                1.0,                    
                // Pointer to the target vector
                sink_p,
                // Incrementer.
                1);   

        for (int j = 0; j < _chunk; j++)
        {
          for (int k = 0; k < _incoming_p->outsize(); k++)
          {
              derivs_p[j * _incoming_p->outsize() + k] += source_p[i * _chunk + j] * input_p[k];
          }
        }
    }
}
