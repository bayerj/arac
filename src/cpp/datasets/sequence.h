// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_DATASETS_SEQUENCE_INCLUDED
#define Arac_DATASETS_SEQUENCE_INCLUDED


#include <vector>
#include <cassert>


namespace arac {
namespace datasets {
    

class Sequence 
{
    public:
        
        Sequence(int length, int itemsize, double* data_p);
        virtual ~Sequence();
        
        int length() const;
        int itemsize() const;
        double* operator[](int index);
        
    private:
        int _length;
        int _itemsize;
        double* _data_p;
};


inline 
int
Sequence::length() const
{
    return _length;
}


inline
int
Sequence::itemsize() const
{
    return _itemsize;
}


inline
double*
Sequence::operator[](int index)
{
    assert(index < length());
    return _data_p + _itemsize * index;
}



} } // Namespace.

#endif