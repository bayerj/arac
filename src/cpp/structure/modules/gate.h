// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_GATE_INCLUDED
#define Arac_STRUCTURE_MODULES_GATE_INCLUDED


#include "module.h"


namespace arac {
namespace structure {
namespace modules {


using arac::structure::modules::Module;


///
/// Layer that implements a differntial version of the IF ... construct.
/// The input is double the size of the output. The outout is defined to be
/// sig(x0) * x1.
///
class GateLayer : public Module
{
    public:

        ///
        /// Create a new GateLayer object of the gíven size.
        ///
        GateLayer(int size);
        
        /// 
        /// Destroy the GateLayer object.
        ///
        virtual ~GateLayer();

    protected:

        virtual void _forward();
        virtual void _backward();
};


inline GateLayer::~GateLayer() {}


inline GateLayer::GateLayer(int size) :
    Module(size * 2, size)
{
}

    
}
}
}


#endif