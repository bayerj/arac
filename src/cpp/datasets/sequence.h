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
        
        Sequence(int length, int itemsize, const double* data_p);
        virtual ~Sequence();
        
        int length() const;
        int itemsize() const;
        const double* operator[](int index) const;
        
    private:
        int _length;
        const double* _data_p;
        int _itemsize;
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
const double*
Sequence::operator[](int index) const
{
    assert(index < length());
    return _data_p + _itemsize * index;
}


} } // Namespace.

#endif