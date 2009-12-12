// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <iostream>
#include <cstring>

#include "buffer.h"


using arac::common::Buffer;


Buffer::Buffer(size_t rowsize, bool owner) :
    _rowsize(rowsize),
    _owner(owner),
    _contmemory(true),
    _dirtyindex(0)
{
    expand();
}


Buffer::~Buffer()
{
    free_memory();
}


void Buffer::add(double* addend_p, int index)
{

    if (index == -1)
    {
        // Last element.
        index = _content.size() - 1;
    }
    _dirtyindex = _dirtyindex <= index + 1 ? index + 1 : _dirtyindex;

    double* current_p = _content[index];

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
    _dirtyindex = _content.size();
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
        memset((void*) _content[0], 0, sizeof(double) * _rowsize * _dirtyindex);
    }
    else
    {
        for(size_t i = 0; i < _dirtyindex; i++)
        {
            clear_at(i);
        }
    }
    _dirtyindex = 0;
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
