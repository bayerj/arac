// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_MDLSTM_INCLUDED
#define Arac_STRUCTURE_MODULES_MDLSTM_INCLUDED


#include "module.h"


namespace arac {
namespace structure {
namespace modules {

///
/// MdlstmLayer objects implement the MDLSTM algorithm as introduced by Alex 
/// Graves. They are a generalization of the LSTM algorithm and suited for
/// multidimensional sequences. 
///
/// MdlstmLayer objects have a special input layout which is outlined below.
///
/// Let s be the size of the layer and d the timedimension.
///
/// Layout of inputs and outputs
/// ----------------------------
/// Depending on how many "real" inputs the layer is given, there will be
///
///     I = (3 + 2 * d) * s
///     
/// inputs over all, where s is the size (= the "true" input) and d is the
/// dimensionality of the MDRNN. The input layout is as follows:
///
///     Name            Size in doubles
///     -------------------------------
///     input gate      s
///     forget gate     s * d
///     input           s
///     output gate     s
///     states          s * d
///
/// The output layout corresponds to
///
///      Name            Size in doubles
///      -------------------------------
///      output          s
///      states          s
///

class MdlstmLayer : public arac::structure::modules::Module
{
    public:

        ///
        /// Create a new MdlstmLayer object of the given timedim and size.
        ///
        MdlstmLayer(int timedim, int size);

        ///
        /// Destroy the MdlstmLayer object.
        ///
        virtual ~MdlstmLayer();
        
        ///
        /// Return a reference to the input_squashed Buffer.
        ///
        arac::common::Buffer& input_squashed();

        ///
        /// Return a reference to the output_gate_squashed Buffer.
        ///
        arac::common::Buffer& output_gate_squashed();

        ///
        /// Return a reference to the output_gate_unsquashed Buffer.
        ///
        arac::common::Buffer& output_gate_unsquashed();

        ///
        /// Return a reference to the input_gate_squashed Buffer.
        ///
        arac::common::Buffer& input_gate_squashed();

        ///
        /// Return a reference to the input_gate_unsquashed Buffer.
        ///
        arac::common::Buffer& input_gate_unsquashed();

        ///
        /// Return a reference to the forget_gate_squashed Buffer.
        ///
        arac::common::Buffer& forget_gate_squashed();

        ///
        /// Return a reference to the forget_gate_unsquashed Buffer.
        ///
        arac::common::Buffer& forget_gate_unsquashed();

    private:
        
        // Set the intermediate buffers to zero.
        // TODO: find better name.
        void clear_intermediates();
        
        virtual void _forward();
        virtual void _backward();
        
        virtual void expand();
        
        int _timedim;
        
        arac::common::Buffer _input_squashed;
        
        arac::common::Buffer _input_gate_squashed;
        arac::common::Buffer _input_gate_unsquashed;
        
        arac::common::Buffer _output_gate_squashed;
        arac::common::Buffer _output_gate_unsquashed;
        
        arac::common::Buffer _forget_gate_squashed;
        arac::common::Buffer _forget_gate_unsquashed;
        
        // Intermediate buffers.
        double* _inter_input_p;
        double* _output_state_p;
        double* _input_state_p;
        double* _output_error_p;
        double* _output_state_error_p;
        double* _output_gate_error_p;
        double* _forget_gate_error_p;
        double* _input_gate_error_p;
        double* _input_error_p;
        double* _input_state_error_p;
        double* _state_error_p;
        
        double* _outputbuffer_p;
};


inline
arac::common::Buffer& 
MdlstmLayer::input_squashed()
{
    return _input_squashed;
}


inline
arac::common::Buffer& 
MdlstmLayer::output_gate_squashed()
{
    return _output_gate_squashed;
    
}


inline
arac::common::Buffer&
MdlstmLayer::output_gate_unsquashed()
{
    return _output_gate_unsquashed;
}


inline
arac::common::Buffer& 
MdlstmLayer::input_gate_squashed()
{
    return _input_gate_squashed;
}


inline
arac::common::Buffer& 
MdlstmLayer::input_gate_unsquashed()
{
    return _input_gate_unsquashed;
}


inline
arac::common::Buffer&
MdlstmLayer::forget_gate_squashed()
{
    return _forget_gate_squashed;
}


inline
arac::common::Buffer& 
MdlstmLayer::forget_gate_unsquashed()
{
    return _forget_gate_unsquashed;
}


} } }  // Namespace.


#endif