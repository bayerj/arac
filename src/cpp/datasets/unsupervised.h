// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_DATASETS_UNSUPERVISED_INCLUDED
#define Arac_DATASETS_UNSUPERVISED_INCLUDED


#include <vector>


namespace arac {
namespace datasets {
    

// TODO: document.

class UnsupervisedDataset 
{
    public: 
    
        UnsupervisedDataset(int samplesize);
        virtual ~UnsupervisedDataset();
        
        // Return the the size of a sample.
        int samplesize();

        // Return the number of rows currently in the dataset.
        virtual int size() = 0;
        
    private:
        
        int _samplesize;
        
};


inline
int 
UnsupervisedDataset::samplesize()
{
    return _samplesize;
}


}
}

#endif