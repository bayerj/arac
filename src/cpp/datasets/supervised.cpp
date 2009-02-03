// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include "supervised.h"


using arac::datasets::SupervisedDataset;


SupervisedDataset::SupervisedDataset(int samplesize, int targetsize) :
    UnsupervisedDataset(samplesize),
    _targetsize(targetsize)
{
    
}


SupervisedDataset::~SupervisedDataset()
{
    
}





