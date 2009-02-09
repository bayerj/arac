// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_PARTIALSOFTMAX_INCLUDED
#define Arac_STRUCTURE_MODULES_PARTIALSOFTMAX_INCLUDED


#include "module.h"


namespace arac {
namespace structure {
namespace modules {

using arac::structure::modules::Module;


///
/// A PartialSoftmaxLayer object softmaxes the output over slices. 
/// For example, a layer might have an input size of 4 - if the slicelength is
/// 2, the first two and the second two inputs will be transformed to each add 
/// up to 1.0 in the output.
///

class PartialSoftmaxLayer : public Module
{
    public:

        ///
        /// Create a new PartialSoftmaxLayer object.
        ///
        PartialSoftmaxLayer(int size, int slicelength);
        
        ///
        /// Destroy the PartialSoftmaxLayer object.
        ///
        virtual ~PartialSoftmaxLayer();

    protected:
        
        virtual void _forward();
        virtual void _backward();
        
        int _slicelength;
};


inline PartialSoftmaxLayer::~PartialSoftmaxLayer() {}


inline PartialSoftmaxLayer::PartialSoftmaxLayer(int size, int slicelength) :
    Module(size, size),
    _slicelength(slicelength)
{
}

    
}
}
}


#endif