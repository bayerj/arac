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
        
        Sequence();
        Sequence(int length, int contentsize, int targetsize,
                 const double* contents_p, const double* targets_p);
        virtual ~Sequence();
        
        int length();
        const double* contents(int index);
        virtual const double* targets(int index);
        
    private:
        int _length;
        const double* _contents_p;
        const double* _targets_p;
        int _contentsize;
        int _targetsize;
};


inline 
int
Sequence::length()
{
    return _length;
}


inline
const double*
Sequence::contents(int index)
{
    assert(index < length());
    return _contents_p + _contentsize * index;
}


inline
const double*
Sequence::targets(int index)
{
    assert(index < length());
    return _targets_p + _targetsize * index;
}

} } // Namespace.

#endif