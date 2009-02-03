// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_DATASETS_DATASETS_INCLUDED
#define Arac_DATASETS_DATASETS_INCLUDED


#include "sequence.h"
#include "supervised.h"
#include "unsupervised.h"


typedef arac::datasets::SupervisedDataset<arac::datasets::Sequence, arac::datasets::Sequence> SupervisedSequentialDataset;
typedef arac::datasets::SupervisedDataset<arac::datasets::Sequence, double*> SupervisedSemiSequentialDataset;
typedef arac::datasets::SupervisedDataset<double*, double*> SupervisedSimpleDataset;
typedef arac::datasets::UnsupervisedDataset<double*> UnsupervisedSimpleDataset;
typedef arac::datasets::UnsupervisedDataset<arac::datasets::Sequence> UnsupervisedSequenceDataset;


#endif