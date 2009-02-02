// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_DATASETS_SEMISEQUENCE_INCLUDED
#define Arac_DATASETS_SEMISEQUENCE_INCLUDED


#include <vector>
#include "sequence.h"


namespace arac {
namespace datasets {


// TODO: document
    

class SemiSequence : public Sequence
{
    public:
        SemiSequence(int length, int contentsize, int targetsize,
                 const double* contents_p, const double* targets_p);
        virtual ~SemiSequence();
        
        virtual const double* targets(int index);
};


inline
const double*
SemiSequence::targets(int index)
{
    
    return Sequence::targets(0);
}

} } // Namespace.

#endif