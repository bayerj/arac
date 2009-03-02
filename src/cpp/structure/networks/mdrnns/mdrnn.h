// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_NETWORKS_MDRNNS_MDRNN_INCLUDED
#define Arac_STRUCTURE_NETWORKS_MDRNNS_MDRNN_INCLUDED


#include <cassert>
#include <iostream>
#include <cstring>

#include "basemdrnn.h"
#include "../../connections/connections.h"
#include "../../modules/modules.h"
#include "../../parametrized.h"
#include "../../component.h"


using arac::structure::Component;
using arac::structure::modules::MdlstmLayer;
using arac::structure::connections::Connection;
using arac::structure::connections::FullConnection;
using arac::structure::connections::IdentityConnection;


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

        virtual void sort();
        
        virtual void _forward();
        
        virtual void _backward();
        
        virtual void clear();
        virtual void clear_derivatives();
        
    protected:
        
        void init_multiplied_sizes();
        
        ///
        /// Create the hidden module.
        ///
        void init_module();
        
        // FIXME: This should be done with integers!
        void next_coords(double* coords);
        void coords_by_index(double* coords_p, int index);
        void index_by_coords(int& index, double* coords_p);
        void update_sizes();
        
        ///
        /// Free the memory held by internal structure.
        ///
        void delete_structure();
        
        int _hiddensize;
        
        int _sequencelength;
        int _blocksize;
        
        int* _sequence_shape_p;
        int* _block_shape_p;
        
        ///
        /// Size of the previous dimensions in memory; example: if a shape of 
        /// (4, 4, 4) is given, each element holds the product of the previous
        /// dimensions: (1, 4, 16) with the special case of the first element 
        /// being one.
        ///
        int* _multiplied_sizes_p;
        
        module_type* _module_p;
        arac::structure::modules::Bias _bias;
        
        ///
        /// Vector of vectors which is used to store the connections along the
        /// differen time dimensions. It has exactly _timedim + 1 items, of 
        /// which the last keeps connections that are invariant to the 
        /// timedimensions.
        ///
        typedef std::vector<arac::structure::connections::Connection*> ConPtrVector;
        typedef std::vector<ConPtrVector> ConPtrVectorVector;
        ConPtrVectorVector _connections;
        
        void init_con_vectors();
        
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


template <class module_type>
Mdrnn<module_type>::Mdrnn(int timedim, int hiddensize) :
    BaseMdrnn(timedim),
    _hiddensize(hiddensize),
    _module_p(0)
{
    _sequence_shape_p = new int[_timedim];
    _block_shape_p = new int[_timedim];
    _multiplied_sizes_p = new int[_timedim];
    for(int i = 0; i < _timedim; i++)
    {
        // We cannot use the setters here, because they invoke the recalculation
        // which results in undefined behaviour.
        _sequence_shape_p[i] = 1;
        _block_shape_p[i] = 1;
    }
    update_sizes();
}


template <class module_type>
Mdrnn<module_type>::~Mdrnn()
{
    delete[] _sequence_shape_p;
    delete[] _block_shape_p;
    delete_structure();
}


template <class module_type>
void
Mdrnn<module_type>::delete_structure()
{
    if (_module_p != 0)
    {
        delete _module_p;
    }
    
    ConPtrVectorVector::iterator con_vec_iter;
    ConPtrVector::iterator con_iter;
    for (con_vec_iter = _connections.begin();
         con_vec_iter != _connections.end();
         con_vec_iter++)
    {
        for (con_iter = con_vec_iter->begin();
             con_iter != con_vec_iter->end();
             con_iter++)
        {
            delete *con_iter;
        }
    }
}


template <class module_type>
void
Mdrnn<module_type>::init_multiplied_sizes()
{
    int size = 1;
    for(int i = 0; i < _timedim; i++)
    {
        _multiplied_sizes_p[i] = size;
        size *= _sequence_shape_p[i];
    }
}


template <class module_type>
void
Mdrnn<module_type>::update_sizes()
{
    _blocksize = 1;
    for (int i = 0; i < _timedim; i++)
    {
        _blocksize *= _block_shape_p[i];
    }
    assert(_blocksize > 0);
    
    _sequencelength = 1;
    for (int i = 0; i < _timedim; i++)
    {
        _sequencelength *= _sequence_shape_p[i];
    }
    
    _sequencelength /= _blocksize;
    _insize = _sequencelength * _blocksize;
    _outsize = _sequencelength * _hiddensize;
}


template <class module_type>
void
Mdrnn<module_type>::init_con_vectors()
{
    _connections.clear();
    ConPtrVector c;
    for (int i = 0; i <= _timedim; i++)
    {
        _connections.push_back(c);
    }
}


template <class module_type>
void
Mdrnn<module_type>::sort()
{
    // Initialize multilied sizes.
    init_multiplied_sizes();
    update_sizes();
    
    delete_structure();
    init_module();
    init_con_vectors();

    // Clear the parametrized vector.
    _parametrizeds.clear();

    // Initialize recurrent self connections.
    int recurrency = 1;
    for(int i = 0; i < _timedim; i++)
    {
        FullConnection* con_p = new FullConnection(_module_p, _module_p);
        con_p->set_mode(Component::Sequential);
        con_p->set_recurrent(recurrency);
        recurrency *= _sequence_shape_p[i] / _block_shape_p[i];
        _connections[i].push_back(con_p);
        _parametrizeds.push_back(con_p);
    }

    // Add a connection from the bias.
    FullConnection* con_p = new FullConnection(&_bias, _module_p);
    _connections[_timedim].push_back(con_p);
    _parametrizeds.push_back(con_p);
    
    // Ininitialize buffers.
    init_buffers();
    
    // Indicate that the net is ready for use.
    _dirty = false;
}

template <class module_type>
void
Mdrnn<module_type>::clear()
{
    
    BaseMdrnn::clear();
    _bias.clear();
    _module_p->clear();
    ConPtrVectorVector::iterator con_vec_iter;
    ConPtrVector::iterator con_iter;
    for (con_vec_iter = _connections.begin();
         con_vec_iter != _connections.end();
         con_vec_iter++)
    {
        for (con_iter = con_vec_iter->begin();
             con_iter != con_vec_iter->end();
             con_iter++)
        {
            (*con_iter)->clear();
        }
    }
}


template <class module_type>
void
Mdrnn<module_type>::clear_derivatives()
{
    std::vector<Parametrized*>::iterator param_iter;
    for (param_iter = _parametrizeds.begin();
         param_iter != _parametrizeds.end();
         param_iter++)
    {
        (*param_iter)->clear_derivatives();
    }
}


template <class module_type>
void
Mdrnn<module_type>::_forward()
{
    // We keep the coordinates of the current block in here.
    double* coords_p = new double[_timedim];
    memset(coords_p, 0, sizeof(double) * _timedim);
    // TODO: save memory by not copying but referencing.
    for(int i = 0; i < sequencelength(); i++)
    {
        ConPtrVectorVector::iterator con_vec_iter;
        ConPtrVector::iterator con_iter;
        int j = 0;
        for (con_vec_iter = _connections.begin();
             con_vec_iter != _connections.end();
             con_vec_iter++, j++)
        {
            for (con_iter = con_vec_iter->begin();
                 con_iter != con_vec_iter->end();
                 con_iter++)
            {
                // If the current coordinate is zero, we are at a border of the 
                // input in that dimension. In that case, the connections may not be
                // forwarded, since we don't want to look around corners.
                if ((j < _timedim) && (coords_p[j] == 0))
                {
                    (*con_iter)->dry_forward();
                }
                else
                {
                    (*con_iter)->forward();
                }
            }
        }
        
        _module_p->add_to_input(input()[timestep()] + blocksize() * i);
        _module_p->forward();
        next_coords(coords_p);
    }
    // Copy the output to the mdrnns outputbuffer.
    // TODO: save memory by not copying but referencing.
    std::vector<double*>::iterator dblp_iter;
    for(int i = 0; i < sequencelength(); i++)
    {
        memcpy(output()[timestep()] + i * _hiddensize, 
               _module_p->output()[i], 
               _hiddensize * sizeof(double));
    }
}


template <class module_type>
void
Mdrnn<module_type>::_backward()
{
    // We keep the coordinates of the current block in here.
    double* coords_p = new double[_timedim];
    memset(coords_p, 0, sizeof(double) * _timedim);
    // TODO: save memory by not copying but referencing.
    for(int i = sequencelength() - 1; i >= 0; i--)
    {
        ConPtrVectorVector::iterator con_vec_iter;
        ConPtrVector::iterator con_iter;
        int j = 0;
        for (con_vec_iter = _connections.begin();
             con_vec_iter != _connections.end();
             con_vec_iter++, j++)
        {
            for (con_iter = con_vec_iter->begin();
                 con_iter != con_vec_iter->end();
                 con_iter++)
            {
                // If the current coordinate is zero, we are at a border of the 
                // input in that dimension. In that case, the connections may not be
                // forwarded, since we don't want to look around corners.
                if ((j < _timedim) && (coords_p[j] == 0))
                {
                    (*con_iter)->dry_backward();
                }
                else
                {
                    (*con_iter)->backward();
                }
            }
        }
        _module_p->add_to_outerror(outerror()[timestep() - 1] + i);
        _module_p->backward();
        next_coords(coords_p);
    }
    
    // Copy the output to the mdrnns outputbuffer.
    // TODO: save memory by not copying but referencing.
    for(int i = 0; i < sequencelength(); i++)
    {
        memcpy(inerror()[timestep() - 1] + i * blocksize(), 
               _module_p->outerror()[i], 
               blocksize() * sizeof(double));
    }
}


}
}
}
}


#endif