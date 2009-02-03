// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include "dataset_sequence_array.h"


using arac::datasets::Dataset_sequence_array;


Dataset_sequence_array::Dataset_sequence_array(int samplesize, int targetsize) :
    SupervisedDataset(samplesize, targetsize)
{
    
}


Dataset_sequence_array::~Dataset_sequence_array()
{
    
}


void
Dataset_sequence_array::append(Sequence sample, const double* target_p)
{
    std::pair<Sequence, const double*> new_pair(sample, target_p);
    _rows.push_back(new_pair);
}