// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_DOUBLEGATE_INCLUDED
#define Arac_STRUCTURE_MODULES_DOUBLEGATE_INCLUDED


#include "module.h"


namespace arac {
namespace structure {
namespace modules {


using arac::structure::modules::Module;


//
// TODO: Documet
//
//
class DoubleGateLayer : public Module
{
    public:

        ///
        /// Create a new DoubleGateLayer object of the gíven size.
        ///
        DoubleGateLayer(int size);
        
        /// 
        /// Destroy the DoubleGateLayer object. 
        ///
        virtual ~DoubleGateLayer();

    protected:

        virtual void _forward();
        virtual void _backward();
};


inline DoubleGateLayer::~DoubleGateLayer() {}


inline DoubleGateLayer::DoubleGateLayer(int size) :
    Module(size * 2, size * 2)
{
}

    
}
}
}


#endif
