// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <iostream>

#include "buffer.h"


using arac::common::Buffer;


Buffer::Buffer(int rowsize, bool owner) : 
    _rowsize(rowsize), 
    _owner(owner), 
    _current_index(-1)
{
    expand();
}


Buffer::~Buffer()
{
    if (owner())
        free_memory();
}


void Buffer::add(double* addend_p)
{
    double* current_p = _content[_current_index];
    for(int i = 0; i < _rowsize; i++)
    {
        current_p[i] += addend_p[i];
    }
}


void Buffer::expand()
{
    double* new_chunk = new double[_rowsize];
    memset((void*) new_chunk, 0, sizeof(double) * _rowsize);
    _content.push_back(new_chunk);
    _current_index++;
}


void Buffer::make_zero()
{
    DoublePtrVec::iterator iter;
    for(iter = _content.begin(); iter != _content.end(); iter++)
    {
        memset((void*) *iter, 0, sizeof(double) * _rowsize);
    }
}


void Buffer::free_memory()
{
    DoublePtrVec::iterator iter;
    for(iter = _content.begin(); iter != _content.end(); iter++)
    {
        delete[] *iter;
    }
}


void Buffer::append(double* row)
{
    _content.push_back(row);
}


void Buffer::make_owner()
{
    _owner = true;
}


bool Buffer::owner()
{
    return _owner;
}

double* Buffer::current()
{
    return _content[_current_index];
}
        

        
