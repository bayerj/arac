// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include "semisequence.h"


using arac::datasets::SemiSequence;
using arac::datasets::Sequence;


SemiSequence::SemiSequence(int length, int contentsize, int targetsize,
                 const double* contents_p, const double* targets_p) :
        Sequence(length, contentsize, targetsize, contents_p, targets_p)
{
    
}


SemiSequence::~SemiSequence()
{
    
}
