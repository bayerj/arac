// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include "dataset_array_array.h"


using arac::datasets::Dataset_array_array;


void
Dataset_array_array::append(const double* sample_p, const double* target_p)
{
    std::pair<const double*, const double*> new_pair(sample_p, target_p);
    _rows.push_back(new_pair);
}