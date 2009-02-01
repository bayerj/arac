// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_DATASETS_DATASET_INCLUDED
#define Arac_DATASETS_DATASET_INCLUDED


#include <vector>


namespace arac {
namespace datasets {
    

// TODO: document.

class Dataset 
{
    public: 
    
        Dataset();
        Dataset(int inputsize, int targetsize=0);
        virtual ~Dataset();
        
        // Append another row to the dataset.
        void append(double* data_p);
        
        // Return the number of rows currently in the dataset.
        int size();
        
        // Return the inputsize.
        int inputsize();
        
        // Return the targetsize.
        int targetsize();
        
        // Direct access to the pointers.
        double* operator[](int i);
        
    private:
        
        int _inputsize;
        int _targetsize;
        std::vector<double*> _rows;
        
        
};


inline
void
Dataset::append(double* data_p)
{
    _rows.push_back(data_p);
}


inline
int
Dataset::size()
{
    return _rows.size();
}
 
 
inline
int 
Dataset::inputsize()
{
    return _inputsize;
}


inline
int
Dataset::targetsize()
{
    return _targetsize;
}


inline
double*
Dataset::operator[](int i)
{
    return _rows[i];
}


}
}

#endif