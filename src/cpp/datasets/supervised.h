// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_DATASETS_SUPERVISED_INCLUDED
#define Arac_DATASETS_SUPERVISED_INCLUDED


#include <vector>

#include "unsupervised.h"


namespace arac {
namespace datasets {
    

// TODO: document.

class SupervisedDataset : public UnsupervisedDataset
{
    public: 
    
        SupervisedDataset(int samplesize, int targetsize);
        virtual ~SupervisedDataset();
        
        // Return the the size of a target.
        int targetsize();

    private:
        
        int _targetsize;
};


inline
int
SupervisedDataset::targetsize()
{
    return _targetsize;
}


}
}

#endif