// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_PARTIALSOFTMAX_INCLUDED
#define Arac_STRUCTURE_MODULES_PARTIALSOFTMAX_INCLUDED


#include "module.h"


namespace arac {
namespace structure {
namespace modules {

using arac::structure::modules::Module;

class PartialSoftmaxLayer : public Module
{
    public:

        PartialSoftmaxLayer(int size, int slicelength);
        ~PartialSoftmaxLayer();

        virtual void forward();
        virtual void backward();
        
    protected:
        
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