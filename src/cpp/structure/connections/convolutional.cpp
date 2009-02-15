// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>

#include "convolutional.h"


using arac::structure::connections::ConvolutionalConnection;
using arac::structure::connections::Connection;
using arac::structure::Parametrized;
using arac::structure::modules::Module;


ConvolutionalConnection::ConvolutionalConnection(Module* incoming_p, Module* outgoing_p, 
                                                 int inchunk, int outchunk) :
    Connection(incoming_p, outgoing_p),
    Parametrized(inchunk * outchunk)
{
    // TODO: Check that insize divides by inchunk and outsize divides by 
    // outchunk.
}


ConvolutionalConnection::~ConvolutionalConnection()
{
    
}



void
ConvolutionalConnection::_forward()
{
    
}


void
ConvolutionalConnection::_backward()
{

}