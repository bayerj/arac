// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include "dataset.h"


using arac::datasets::Dataset;


Dataset::Dataset() :
    _inputsize(0),
    _targetsize(0)
{
    
}


Dataset::Dataset(int inputsize, int targetsize) :
    _inputsize(inputsize),
    _targetsize(targetsize)
{
    
}
