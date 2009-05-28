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
        std::pair<SampleType, TargetType>& operator[](int index);

        ///
        /// Tell wether this dataset uses importance.
        ///
        bool has_importance();

        /// 
        /// Return the importance of a sample/target pair.
        ///
        TargetType importance(int index);

        ///
        /// Set the importance of a sample/target pair.
        ///
        void set_importance(int index, TargetType importance);
        
    private:
 
        ///
        /// Size of a single target.
        ///
        int _targetsize;
        
        /// 
        /// Vector that holds the (sample, target) pairs.
        ///
        std::vector<std::pair<SampleType, TargetType> > _rows;

        ///
        /// Vector that holds the importances of (sample, target) pairs.
        ///
        std::vector<TargetType> _importance;

        ///
        /// Flag that tells whether importance is used.
        ///
        bool _has_importance;
};


template<typename SampleType, typename TargetType>
SupervisedDataset<SampleType, TargetType>::SupervisedDataset(int samplesize, 
                                                             int targetsize) :
    UnsupervisedDataset<SampleType>(samplesize),
    _targetsize(targetsize),
    _has_importance(false)
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
std::pair<SampleType, TargetType>& 
SupervisedDataset<SampleType, TargetType>::operator[](int index)
{
    assert(index < size());
    return _rows[index];
}

template<typename SampleType, typename TargetType>
inline
bool
SupervisedDataset<SampleType, TargetType>::has_importance()
{
    return _has_importance;
}


template<typename SampleType, typename TargetType>
inline
TargetType
SupervisedDataset<SampleType, TargetType>::importance(int index)
{
    return _importance[index];
}


template<typename SampleType, typename TargetType>
inline
void
SupervisedDataset<SampleType, TargetType>::set_importance(int index, 
                                                          TargetType importance)
{
    _has_importance = true;
    _importance.reserve(size());
    _importance[index] = importance;
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
