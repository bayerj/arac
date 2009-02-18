// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_NETWORKS_MDRNNS_MDRNN_INCLUDED
#define Arac_STRUCTURE_NETWORKS_MDRNNS_MDRNN_INCLUDED


#include <cassert>

#include "basemdrnn.h"
#include "../../connections/connections.h"
#include "../../modules/modules.h"
#include "../../parametrized.h"


namespace arac {
namespace structure {
namespace networks {
namespace mdrnns {


/// 
/// Template class to implement a grid of modules in order to process 
/// multidimensional sequences.
/// 
/// The networks is presented the whole input at once. It holds an internal
/// structure of modules, which this input will be distributed to on activation.
///

template <class module_type>
class Mdrnn : public BaseMdrnn
{
    public:
        
        /// 
        /// Create a new Mdrnn object for the given timedim with layers of size
        /// hiddensize.
        ///
        Mdrnn(int timedim, int hiddensize);
        virtual ~Mdrnn();

        ///
        /// Set the length of the current sequence in the given dimension.
        ///
        void set_sequence_shape(int dim, int val);
        
        ///
        /// Return the length of the current sequence in the given dimension.
        ///
        int get_sequence_shape(int dim);

        ///
        /// Return the amount of items in the sequence. Each item is a vector
        /// of the size of a block.
        ///
        int sequencelength();

        ///
        /// Set the shape of a block along the given dimension.
        ///
        void set_block_shape(int dim, int val);
        
        ///
        /// Return the shape of a block along the given dimension.
        ///
        int get_block_shape(int dim);

        /// Return the size of single item. This equals the product of the
        /// sidelength's of each block.
        int blocksize();

        void sort();
        
        virtual void _forward();
        
        virtual void _backward();
        
        virtual void clear();
        virtual void clear_derivatives();
        
    protected:
        
        void init_multiplied_sizes();
        
        // FIXME: This should be done with integers!
        void next_coords(double* coords);
        void coords_by_index(double* coords_p, int index);
        void index_by_coords(int& index, double* coords_p);
        void update_sizes();
        
        int _hiddensize;
        
        int _sequencelength;
        int _blocksize;
        
        int* _sequence_shape_p;
        int* _block_shape_p;
        
        /// Size of the previous dimensions in memory; example: if a shape of 
        /// (4, 4, 4) is given, each element holds the product of the previous
        /// dimensions: (1, 4, 16) with the special case of the first element 
        /// being one.
        int* _multiplied_sizes_p;
        
        module_type* _module_p;
        arac::structure::modules::Bias _bias;
        
        std::vector<arac::structure::connections::FullConnection*> _connections;
};


template <class module_type>
inline
void
Mdrnn<module_type>::set_sequence_shape(int dim, int val)
{
    assert(dim < _timedim);
    if (_sequence_shape_p[dim] == val)
    {
        return;
    }
    _sequence_shape_p[dim] = val;
    _dirty = true;
    update_sizes();
}


template <class module_type>
inline
int
Mdrnn<module_type>::get_sequence_shape(int dim)
{
    return _sequence_shape_p[dim];
}


template <class module_type>
inline
int
Mdrnn<module_type>::sequencelength()
{
    return _sequencelength;
}


template <class module_type>
inline
void
Mdrnn<module_type>::set_block_shape(int dim, int val)
{
    assert(dim < _timedim);
    assert(val > 0);
    if (_block_shape_p[dim] == val)
    {
        return;
    }
    
    _dirty = true;
    _block_shape_p[dim] = val;
    
    _blocksize = 1;
    for (int i = 0; i < _timedim; i++)
    {
        _blocksize *= _block_shape_p[i];
    }
    update_sizes();
}


template <class module_type>
inline
int
Mdrnn<module_type>::get_block_shape(int dim)
{
    return _block_shape_p[dim];
}


template <class module_type>
inline
int
Mdrnn<module_type>::blocksize()
{
    return _blocksize;
}


template <class module_type>
inline
void
Mdrnn<module_type>::next_coords(double* coords_p)
{
    int i;
    int carry = 0;
    for(i = 0; i < _timedim; i++)
    {
        if (coords_p[i] < _sequence_shape_p[i] - 1)
        {
            coords_p[i] += 1;
            break;
        }
        else 
        {
            coords_p[i] = 0;
        }
    }
}


template <class module_type>
inline
void
Mdrnn<module_type>::coords_by_index(double* coords_p, int index)
{
    int divisor = sequencelength() / blocksize();
    for(int i = _timedim - 1; i <= 0; i--)
    {
        divisor /= _sequence_shape_p[i] / _block_shape_p[i];
        coords_p[i] = index / divisor;
        index = index % divisor;
    }
}


template <class module_type>
inline
void
Mdrnn<module_type>::index_by_coords(int& index, double* coords_p)
{
    index = 0;
    int smallcubesize = 1;
    for(int i = 0; i < _timedim; i++)
    {
        index += coords_p[i] * _sequence_shape_p[i] / _block_shape_p[i];
        smallcubesize *= _sequence_shape_p[i] / _block_shape_p[i];
    }
}


}
}
}
}

// HACK, because this is a template and we want to make a library. See:
//
//      http://yosefk.com/c++fqa/templates.html#fqa-35.13
//
//  for details.

#include "mdrnn.cpp"


#endif