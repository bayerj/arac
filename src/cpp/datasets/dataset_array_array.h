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

        // Return the number of rows currently in the dataset.
        virtual int size();
        
        // Add another row to the dataset.
        void append(const double* sample_p, const double* target_p);
        
    private:
        
        std::vector<std::pair<const double*, const double*> > _rows;
};


inline
int 
Dataset_array_array::size()
{
    return _rows.size();
}

}
}

#endif