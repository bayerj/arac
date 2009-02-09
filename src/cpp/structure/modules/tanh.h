// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_TANH_INCLUDED
#define Arac_STRUCTURE_MODULES_TANH_INCLUDED


#include "module.h"


namespace arac {
namespace structure {
namespace modules {

using arac::structure::modules::Module;


///
/// A TanhLayer object is of the same in- and outputsize and transforms its
/// input by applying the tan hyperbolicus function to each component.
///
class TanhLayer : public Module
{
    public:

        /// 
        /// Create a new TanhLayer object of the given size.
        ///
        TanhLayer(int size);
        virtual ~TanhLayer();
        
    protected:
        
        virtual void _forward();
        virtual void _backward();
};


inline TanhLayer::~TanhLayer() {}


inline TanhLayer::TanhLayer(int size) :
    Module(size, size)
{
}

    
}
}
}


#endif