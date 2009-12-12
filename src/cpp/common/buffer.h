// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_COMMON_BUFFER_INCLUDED
#define Arac_COMMON_BUFFER_INCLUDED


#include <cassert>
#include <iostream>

#include "typedefs.h"


namespace arac {
namespace common {

///
/// Buffer objects hold pointers to arrays which are serving as 
/// inputs/outputs/errors/etc. to modules. 
///
/// A buffer is organized in different rows which are held as a vector of double
/// pointers. 
/// A buffer may either "own" its arrays or not own them. In the first case, the 
/// buffer frees the memory of the arrays on destruction. 
///

class Buffer
{
    public:
        
        Buffer(size_t rowsize, bool owner=true);
        virtual ~Buffer();
        
        ///
        /// Add the contents at the given pointer to the last buffer row. If an
        /// index is given, the content will be added to that buffer row.
        ///
        void add(double* addend_p, int index=-1);
        
        ///
        /// Append an empty  row to the buffer's contents.
        ///
        void expand();
        
        ///
        /// Set all the contents to zero. 
        ///
        void clear();

        ///
        /// Set all the contents of a specific row to zero.
        ///
        void clear_at(size_t index);

        ///
        /// Free the memory held by the Buffer object. Do nothing if the buffer
        /// does not own the memory.
        ///
        void free_memory();
        
        ///
        /// Add the given pointer as a row. Attention: This removes ownership 
        /// for all buffers.
        ///
        void append(double* row);

        ///
        /// Mark the Buffer as the owner of the memory.
        ///
        void make_owner();

        ///
        /// Tell wether the Buffer is the owner of the memory.
        ///
        bool owner();

        ///
        /// Return the size of a single row in the Buffer.
        ///
        size_t rowsize();

        ///
        /// Set the size of a single row.
        ///
        void set_rowsize(size_t value);

        ///
        /// Return the number of rows in the buffer.
        ///
        size_t size();

        ///
        /// Return a pointer to the row at the given index.
        ///
        double* operator [](size_t index);

        ///
        /// Tell wether the memory held by the buffer is continuous.
        ///
        bool contmemory();


    protected:

        ///
        /// Vector that holds the pointer to the rows.
        ///
        DoublePtrVec _content;
        size_t _rowsize;
        bool _owner;
        bool _contmemory;

        // Index that says up to which row the buffer has been used recently. Is
        // being reset by .clear() and clear_at(). Is updated by append(), 
        // add and operator [].
        // The index is an exclusive upper bound.
        size_t _dirtyindex;

};


inline
size_t
Buffer::rowsize()
{
    return _rowsize;
}


inline 
void
Buffer::set_rowsize(size_t value)
{
    _rowsize = value;
}


inline
size_t 
Buffer::size()
{
    return _content.size();
}

    
inline
double*
Buffer::operator[] (size_t index)
{
    _dirtyindex = _dirtyindex <= index ? index + 1 : _dirtyindex;
    return _content[index];
}


inline
void
Buffer::make_owner()
{
    _owner = true;
}


inline
bool
Buffer::owner()
{
    return _owner;
}


inline
bool
Buffer::contmemory()
{
    return _contmemory;
}



}
}


#endif
