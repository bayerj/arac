// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_DATASETS_DATASET_SEQUENCE_ARRAY_INCLUDED
#define Arac_DATASETS_DATASET_SEQUENCE_ARRAY_INCLUDED


#include <vector>
#include "supervised.h"
#include "sequence.h"


namespace arac {
namespace datasets {
    

// TODO: document.

class Dataset_sequence_array : public SupervisedDataset
{
    public: 
        
        Dataset_sequence_array(int samplesize, int targetsize);
        virtual ~Dataset_sequence_array();
        

        // Return the number of rows currently in the dataset.
        virtual int size();
        
        // Add another row to the dataset.
        void append(Sequence sample, const double* target_p);
        
        const std::pair<Sequence, const double*>& operator[](int index);
        
    private:
        
        std::vector<std::pair<Sequence, const double*> > _rows;
};


inline
int 
Dataset_sequence_array::size()
{
    return _rows.size();
}


inline
const std::pair<Sequence, const double*>&
Dataset_sequence_array::operator[](int index)
{
    return _rows[index];
}



}
}

#endif