// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <iostream>
#include <cstring>

#include "buffer.h"


using arac::common::Buffer;


Buffer::Buffer(size_t rowsize, bool owner) :
    _rowsize(rowsize),
    _owner(owner),
    _contmemory(true)
{
    expand();
}


Buffer::~Buffer()
{
    free_memory();
}


void Buffer::add(double* addend_p, int index)
{
    double* current_p = index == -1 ? _content.back() : _content[index];
    for(size_t i = 0; i < _rowsize; i++)
    {
        current_p[i] += addend_p[i];
    }
}


void
Buffer::append(double* row)
{
    _owner = false;
    if ((_contmemory) && (size() > 0))
    {
        if (row != _content.back() + _rowsize)
        {
            _contmemory = false;
        }
    }
    _content.push_back(row);
}


void Buffer::expand()
{
    if (!owner())
    {
        return;
    }
    double* new_chunk = new double[_rowsize];
    memset((void*) new_chunk, 0, sizeof(double) * _rowsize);
    _content.push_back(new_chunk);
    _contmemory = false;
}


void Buffer::clear()
{
    if (_contmemory)
    {
        memset((void*) _content[0], 0, sizeof(double) * _rowsize * size());
    }
    else
    {
        for(size_t i = 0; i < size(); i++)
        {
            clear_at(i);
        }
    }
}


void
Buffer::clear_at(size_t index)
{
    memset((void*) _content[index], 0, sizeof(double) * _rowsize);
}


void Buffer::free_memory()
{
    if (owner())
    {
        DoublePtrVec::iterator iter;
        for(iter = _content.begin(); iter != _content.end(); iter++)
        {
            delete[] *iter;
        }
    }
    _content.clear();
    _contmemory = true;
}
