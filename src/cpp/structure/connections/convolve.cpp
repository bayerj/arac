// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>

extern "C"
{
    #include "cblas.h"
}


#include "convolve.h"


using arac::structure::connections::ConvolveConnection;
using arac::structure::connections::Connection;
using arac::structure::Parametrized;
using arac::structure::modules::Module;


ConvolveConnection::ConvolveConnection(Module* incoming_p, Module* outgoing_p, 
                                             int inchunk, int outchunk) :
    Connection(incoming_p, outgoing_p),
    Parametrized(inchunk * outchunk),
    _inchunk(inchunk),
    _outchunk(outchunk)
{
    assert(incoming()->outsize() / inchunk == outgoing()->insize() / outchunk);
    assert(incoming()->outsize() % inchunk == 0);
    assert(outgoing()->insize() % outchunk == 0);

    _n_chunks = incoming()->outsize() / inchunk;
}


ConvolveConnection::~ConvolveConnection()
{
    
}


void
ConvolveConnection::forward_process(double* sink_p, const double* source_p)
{
    for (int i = 0; i < _n_chunks; i++)
    {
        cblas_dgemv(
                CblasRowMajor, 
                // Do not transpose the matrix since we want to multiply from 
                // the right
                CblasNoTrans,
                // Dimensions of the matrix
                _outchunk,        
                _inchunk,
                // Scalar for the matrix
                1.0,                    
                // Pointer to the matrix
                get_parameters(),    
                // Dimension of the vector
                _inchunk,
                // Pointer to the vector
                source_p + (i * _inchunk),
                // Some incrementer.
                1,                      
                // Scalar of the target vector
                1.0,                    
                // Pointer to the target vector
                sink_p + (i * _outchunk),
                // Incrementer.
                1);   
    }
}


void
ConvolveConnection::backward_process(double* sink_p, const double* source_p)
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
                _inchunk,        
                _outchunk,
                // Scalar for the matrix
                1.0,                    
                // Pointer to the matrix
                get_parameters(),    
                // Dimension of the vector
                _inchunk,
                // Pointer to the vector
                source_p + (i * _outchunk),
                // Some incrementer.
                1,                      
                // Scalar of the target vector
                1.0,                    
                // Pointer to the target vector
                sink_p + (i * _inchunk),
                // Incrementer.
                1);   

        for (int j = 0; j < _outchunk; j++)
        {
          for (int k = 0; k < _inchunk; k++)
          {
              derivs_p[j * _inchunk + k] += source_p[i * _outchunk + j] * input_p[i * _inchunk + k];
          }
        }
    }
}
