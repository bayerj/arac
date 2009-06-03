// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_LSTM_INCLUDED
#define Arac_STRUCTURE_MODULES_LSTM_INCLUDED


#include "module.h"
#include "mdlstm.h"
#include "../../common/common.h"


namespace arac {
namespace structure {
namespace modules {


///
/// Layer that implements the long short-term memory algorithm. Needs to be
/// sequential.
///
/// LstmLayers are superior to normal layers in terms of remembering inputs
/// longer ago. They keep an additional buffer, the state.
///
/// For more information on the algorithm visit:
///     http://www.idsia.ch/~juergen/lstm/
///

class LstmLayer : public Module
{
    public:

        ///
        /// Create a new LstmLayer with size outputs. The layer will have four
        /// times as many inputs.
        ///
        LstmLayer(int size);
        virtual ~LstmLayer();
        
        virtual void set_mode(arac::structure::Component::Mode mode);
        
        ///
        /// Return a reference to the state buffer.
        ///
        arac::common::Buffer& state();
        arac::common::Buffer& state_error();
        

    protected:
        
        virtual void _forward();
        virtual void _backward();
        virtual void expand();
        
        void fill_internal_state();
        void retrieve_internal_state();
        void fill_internal_input();
        void retrieve_internal_output();
        void fill_internal_outerror();
        void retrieve_internal_inerror();
        void fill_internal_state_error();
        void retrieve_internal_state_error();
        
        ///
        /// Since the lstm cell is a special case of the mdlstm cell, the lstm
        /// layer is implemented by wrapping an MdlstmLayer object.
        ///
        MdlstmLayer _mdlstm;

        arac::common::Buffer* _state_p;
        arac::common::Buffer* _state_error_p;
        
};

inline
arac::common::Buffer& LstmLayer::state()
{
    return *_state_p;
}


inline
arac::common::Buffer& LstmLayer::state_error()
{
    return *_state_error_p;
}




}
}
}


#endif
