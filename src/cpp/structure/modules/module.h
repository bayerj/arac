// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_MODULE_INCLUDED
#define Arac_STRUCTURE_MODULES_MODULE_INCLUDED


#include <iostream>

#include "../component.h"
#include "../../common/common.h"


using arac::common::Buffer;


namespace arac {
namespace structure {
namespace modules {
    

class Module : public arac::structure::Component
{
    public:

        // Create a new module and allocate the necessary buffers.
        Module(int insize, int outsize);
        
        // Create a new module. 
        //
        // If error_agnostic is true, no buffers will be allocated.
        Module(int insize, int outsize, bool error_agnostic);

        // Destroy the module. Depending on the ownership, the arrays are 
        // deallocated.
        ~Module();
    
        // Add the contents at the given pointer to the input.
        void add_to_input(double* addend_p);
        
        // Add the contents at the given pointer to the outerror.
        void add_to_outerror(double* addend_p);
        
        // Clear input, output, inerror and outerror by setting them to zero.
        void clear();
        
        // Return the input Buffer.
        Buffer& input();
        
        // Return the output Buffer.
        Buffer& output();
        
        // Return the inerror Buffer.
        Buffer& inerror();
        
        // Return the outerror Buffer.
        Buffer& outerror();
        
        // Return the input size of the module.
        int insize();
        
        // Return the output size of the module.
        int outsize();
        
        // Set the sequential flag of the module.
        void set_sequential(bool sequential);
        
        // Tell if the module is sequential.
        bool sequential();
        
    protected:

        bool _sequential;
        int *_timestep_p;
        
        bool _error_agnostic;
        int _insize;
        int _outsize;
        
        Buffer _input;
        Buffer _output;
        Buffer _inerror;
        Buffer _outerror;
};





inline
int
Module::insize()
{
    return _insize;
}


inline
int
Module::outsize()
{
    return _outsize;
}


inline 
void
Module::add_to_input(double* addend_p)
{
    _input.add(addend_p);
}


inline 
void
Module::add_to_outerror(double* addend_p)
{
    _outerror.add(addend_p);
}


inline
Buffer&
Module::input()
{
    return _input; 
}


inline
Buffer&
Module::output()
{
    return _output;
}

inline
Buffer&
Module::inerror()
{
    return _inerror;
}


inline
Buffer&
Module::outerror()
{
    return _outerror;
}


}
}
}


#endif