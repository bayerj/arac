// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_MULTIPLICATION_INCLUDED
#define Arac_STRUCTURE_MODULES_MULTIPLICATION_INCLUDED


#include "module.h"


namespace arac {
namespace structure {
namespace modules {


using arac::structure::modules::Module;


//
// TODO: document
//
class MultiplicationLayer : public Module
{
    public:

        ///
        /// Create a new MultiplicationLayer object of the gíven size.
        ///
        MultiplicationLayer(int size);
        
        /// 
        /// Destroy the MultiplicationLayer object.
        ///
        virtual ~MultiplicationLayer();

    protected:

        virtual void _forward();
        virtual void _backward();
};


inline MultiplicationLayer::~MultiplicationLayer() {}


inline MultiplicationLayer::MultiplicationLayer(int size) :
    Module(size * 2, size)
{
}

    
}
}
}


#endif

