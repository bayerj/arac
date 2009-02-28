// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef ARAC_MDRNN_C
#define ARAC_MDRNN_C


#include <iostream>
#include <cstring>

#include "mdrnn.h"


using arac::structure::Component;
using arac::structure::connections::FullConnection;
using arac::structure::networks::mdrnns::Mdrnn;


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
    delete _module_p;
    
    std::vector<FullConnection*>::iterator con_iter;
    for(con_iter = _connections.begin(); 
        con_iter != _connections.end(); 
        con_iter++)
    {
        delete *con_iter;
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
Mdrnn<module_type>::sort()
{
    // Initialize multilied sizes.
    init_multiplied_sizes();
    update_sizes();
    
    // Initialize module.
    if (_module_p != 0)
    {
        delete _module_p;
    }
    _module_p = new module_type(_hiddensize);
    _module_p->set_mode(Component::Sequential);
    
    // Delete connections from previous sortings.
    std::vector<FullConnection*>::iterator con_iter;
    for (con_iter = _connections.begin();
         con_iter != _connections.end();
         con_iter++)
    {
         delete (*con_iter);
    }
    _connections.clear();

    // Also clear the parametrized vector.
    _parametrizeds.clear();
    
    // Initialize recurrent self connections.
    int recurrency = 1;
    for(int i = 0; i < _timedim; i++)
    {
        FullConnection* con_p = new FullConnection(_module_p, _module_p);
        con_p->set_mode(Component::Sequential);
        con_p->set_recurrent(recurrency);
        recurrency *= _sequence_shape_p[i] / _block_shape_p[i];
        _connections.push_back(con_p);
        _parametrizeds.push_back(con_p);
    }

    // Add a connection from the bias.
    FullConnection* con_p = new FullConnection(&_bias, _module_p);
    _parametrizeds.push_back(con_p);
    _connections.push_back(con_p);
    
    // Ininitialize buffers.
    init_buffers();
    
    // Indicate that the net is ready for use.
    _dirty = false;
}


template <class module_type>
void
Mdrnn<module_type>::clear()
{
    _module_p->clear();
}


template <class module_type>
void
Mdrnn<module_type>::clear_derivatives()
{
    std::vector<FullConnection*>::iterator coniter;
    for (coniter = _connections.begin();
         coniter != _connections.end();
         coniter++)
    {
        (*coniter)->clear_derivatives();
    }
}


template <class module_type>
void
Mdrnn<module_type>::_forward()
{
    // We keep the coordinates of the current block in here.
    double* coords_p = new double[_timedim];
    // TODO: save memory by not copying but referencing.
    for(int i = 0; i < sequencelength(); i++)
    {
        std::vector<FullConnection*>::iterator con_iter;
        int j = 0;
        for(con_iter = _connections.begin(); 
            con_iter != _connections.end(); 
            con_iter++)
        {
            // If the current coordinate is zero, we are at a border of the 
            // input in that dimension. In that case, the connections may not be
            // forwarded, since we don't want to look around corners.
            if (coords_p[j] == 0)
            {
                (*con_iter)->dry_forward();
            }
            else
            {
                (*con_iter)->forward();
            }
            j++;
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
        std::vector<FullConnection*>::iterator con_iter;
        int j = 0;
        for(con_iter = _connections.begin(); 
            // All but the bias connection.
            con_iter < _connections.end(); 
            con_iter++, j++)
        {
            // std::cout << coords_p[j] << " ";
            // If the current coordinate is zero, we are at a border of the 
            // input in that dimension. In that case, the connections may not be
            // forwarded, since we don't want to look around corners.
            if (coords_p[j] == 0)
            {
                (*con_iter)->dry_backward();
            }
            else
            {
                (*con_iter)->backward();
            }
        }
        // Bias connection is always backwarded.
        _connections.back()->backward();
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



#endif