// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include "sequence.h"


using arac::datasets::Sequence;


Sequence::Sequence(int length, int contentsize, int targetsize,
                   const double* contents_p, const double* targets_p) :
    _length(length),
    _contentsize(contentsize),
    _targetsize(targetsize),
    _contents_p(contents_p),
    _targets_p(targets_p)
{    
}


Sequence::Sequence()
{
}


Sequence::~Sequence()                  
{}
