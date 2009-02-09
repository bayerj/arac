// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_DATASETS_UNSUPERVISED_INCLUDED
#define Arac_DATASETS_UNSUPERVISED_INCLUDED


#include <vector>


namespace arac {
namespace datasets {
    

///
/// UnsupervisedDataset objects hold single data rows.
///

template<typename SampleType>
class UnsupervisedDataset 
{
    public: 

        ///
        /// Create a new UnsupervisedDataset object where each sample is of
        /// the given samplesize.
        ///
        UnsupervisedDataset(int samplesize);
        
        ///
        /// Destroy the UnsupervisedDataset object.
        ///
        virtual ~UnsupervisedDataset();
        
        ///
        /// Return the the size of a sample.
        ///
        int samplesize();

        ///
        /// Return the number of rows currently in the dataset.
        ///
        virtual int size();
        
        ///
        /// Return a reference to the sample at the given index.
        ///
        const SampleType& operator[](int index);
        
    private:
        
        ///
        /// The size of a sample.
        ///
        int _samplesize;
        
        ///
        /// Vector that holds the samples in the dataset.
        ///
        std::vector<SampleType> _rows;
        
};


template <typename SampleType>
UnsupervisedDataset<SampleType>::UnsupervisedDataset(int samplesize) :
    _samplesize(samplesize)
{
    
}

template <typename SampleType>
UnsupervisedDataset<SampleType>::~UnsupervisedDataset()
{
    
}


template<typename SampleType>
inline
int 
UnsupervisedDataset<SampleType>::samplesize()
{
    return _samplesize;
}


template<typename SampleType>
inline
int 
UnsupervisedDataset<SampleType>::size()
{
    return _rows.size();
}


template<typename SampleType>
inline
const SampleType& 
UnsupervisedDataset<SampleType>::operator[](int index)
{
    return _rows[index];
}


}
}

#endif