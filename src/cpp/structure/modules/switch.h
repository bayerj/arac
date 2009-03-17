// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_SWITCH_INCLUDED
#define Arac_STRUCTURE_MODULES_SWITCH_INCLUDED


#include "module.h"


namespace arac {
namespace structure {
namespace modules {


using arac::structure::modules::Module;


//
// TODO: Document.
//
class SwitchLayer : public Module
{
    public:

        ///
        /// Create a new SwitchLayer object of the gíven size.
        ///
        SwitchLayer(int size);
        
        /// 
        /// Destroy the SwitchLayer object.
        ///
        virtual ~SwitchLayer();

    protected:

        virtual void _forward();
        virtual void _backward();
};


inline SwitchLayer::~SwitchLayer() {}


inline SwitchLayer::SwitchLayer(int size) :
    Module(size, size * 2)
{
}

    
}
}
}


#endif
