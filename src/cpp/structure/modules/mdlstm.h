// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_MDLSTM_INCLUDED
#define Arac_STRUCTURE_MODULES_MDLSTM_INCLUDED


#include "module.h"


namespace arac {
namespace structure {
namespace modules {

using arac::structure::modules::Module;

class MdlstmLayer : public Module
{
    public:

        MdlstmLayer(int timedim, int size);
        ~MdlstmLayer();

        virtual void forward();
        virtual void backward();
        
    protected:
        
        int _timedim;
        
        Buffer _input_squashed;
        Buffer _input_gate_squashed;
        Buffer _input_gate_unsquashed;
        Buffer _output_gate_squashed;
        Buffer _output_gate_unsquashed;
        Buffer _forget_gate_unsquashed;
        Buffer _forget_gate_squashed;
};


}
}
}


#endif