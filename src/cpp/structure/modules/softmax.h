// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_SOFTMAX_INCLUDED
#define Arac_STRUCTURE_MODULES_SOFTMAX_INCLUDED


#include "module.h"


namespace arac {
namespace structure {
namespace modules {

using arac::structure::modules::Module;

class SoftmaxLayer : public Module
{
    public:

        SoftmaxLayer(int size);
        ~SoftmaxLayer();

        virtual void forward();
        virtual void backward();
};


inline SoftmaxLayer::~SoftmaxLayer() {}


inline SoftmaxLayer::SoftmaxLayer(int size) :
    Module(size, size)
{
}

    
}
}
}


#endif