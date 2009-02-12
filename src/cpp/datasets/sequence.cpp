// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include "sequence.h"


using arac::datasets::Sequence;


Sequence::Sequence(int length, int itemsize, double* data_p) :
    _length(length),
    _itemsize(itemsize),
    _data_p(data_p)
{    
}


Sequence::~Sequence()                  
{
    
}