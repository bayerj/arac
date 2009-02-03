// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_DATASETS_DATASET_ARRAY_ARRAY_INCLUDED
#define Arac_DATASETS_DATASET_ARRAY_ARRAY_INCLUDED


#include <vector>
#include "supervised.h"


namespace arac {
namespace datasets {
    

// TODO: document.

class Dataset_array_array : public SupervisedDataset
{
    public: 
        
        Dataset_array_array(int samplesize, int targetsize);
        virtual ~Dataset_array_array();
        

        // Return the number of rows currently in the dataset.
        virtual int size();
        
        // Add another row to the dataset.
        void append(const double* sample_p, const double* target_p);
        
        const std::pair<const double*, const double*>& operator[](int index);
        
    private:
        
        std::vector<std::pair<const double*, const double*> > _rows;
};


inline
int 
Dataset_array_array::size()
{
    return _rows.size();
}


inline
const std::pair<const double*, const double*>&
Dataset_array_array::operator[](int index)
{
    return _rows[index];
}



}
}

#endif