// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_DATASETS_SUPERVISED_INCLUDED
#define Arac_DATASETS_SUPERVISED_INCLUDED


#include <vector>

#include "unsupervised.h"


namespace arac {
namespace datasets {
    

///
/// SupervisedDataset objects hold pairs of data, each of possibly different 
/// types.
///

template<typename SampleType, typename TargetType>
class SupervisedDataset : public UnsupervisedDataset<SampleType>
{
    public: 
    
        ///
        /// Create a new SupervisedDataset object with the given samplesize and
        /// the given targetsize.
        ///
        SupervisedDataset(int samplesize, int targetsize);
        
        /// 
        /// Destroy the SupervisedDataset object.
        ///
        virtual ~SupervisedDataset();
        
        ///
        /// Return the amount of rows in the dataset.
        ///
        virtual int size();
        
        ///
        /// Return the the size of a target.
        ///
        int targetsize();

        ///
        /// Add another row to the dataset.
        ///
        void append(SampleType sample, TargetType target);
        
        ///
        /// Return the (sample, target) pair of the dataset at the given index.
        ///
        const std::pair<SampleType, TargetType>& operator[](int index);
        
    private:
 
        ///
        /// Size of a single target.
        ///
        int _targetsize;
        
        /// 
        /// Vector that holds the (sample, target) pairs.
        ///
        std::vector<std::pair<SampleType, TargetType> > _rows;
        
};


template<typename SampleType, typename TargetType>
SupervisedDataset<SampleType, TargetType>::SupervisedDataset(int samplesize, int targetsize) :
    UnsupervisedDataset<SampleType>(samplesize),
    _targetsize(targetsize)
{
    
}


template<typename SampleType, typename TargetType>
SupervisedDataset<SampleType, TargetType>::~SupervisedDataset()
{
    
}


template<typename SampleType, typename TargetType>
inline
int
SupervisedDataset<SampleType, TargetType>::targetsize()
{
    return _targetsize;
}


template<typename SampleType, typename TargetType>
void
SupervisedDataset<SampleType, TargetType>::append(
    SampleType sample, TargetType target)
{
    std::pair<SampleType, TargetType> new_pair(sample, target);
    _rows.push_back(new_pair);
}


template<typename SampleType, typename TargetType>
const std::pair<SampleType, TargetType>& 
SupervisedDataset<SampleType, TargetType>::operator[](int index)
{
    return _rows[index];
}


template<typename SampleType, typename TargetType>
inline
int 
SupervisedDataset<SampleType, TargetType>::size()
{
    return _rows.size();
}


}
}

#endif