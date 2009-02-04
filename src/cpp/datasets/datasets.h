// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_DATASETS_DATASETS_INCLUDED
#define Arac_DATASETS_DATASETS_INCLUDED


#include "sequence.h"
#include "supervised.h"
#include "unsupervised.h"

namespace arac {
namespace datasets {
    
typedef SupervisedDataset<arac::datasets::Sequence, arac::datasets::Sequence> SupervisedSequentialDataset;
typedef SupervisedDataset<arac::datasets::Sequence, double*> SupervisedSemiSequentialDataset;
typedef SupervisedDataset<double*, double*> SupervisedSimpleDataset;
typedef UnsupervisedDataset<double*> UnsupervisedSimpleDataset;
typedef UnsupervisedDataset<arac::datasets::Sequence> UnsupervisedSequenceDataset;
    
} } // Namespace


#endif