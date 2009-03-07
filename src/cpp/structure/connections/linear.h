// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_CONNECTIONS_LINEAR_INCLUDED
#define Arac_STRUCTURE_CONNECTIONS_LINEAR_INCLUDED


#include "../modules/module.h"
#include "../parametrized.h"
#include "connection.h"


namespace arac {
namespace structure {
namespace connections {

    
using namespace arac::structure::modules;
using arac::structure::Parametrized;


/// 
/// A LinearConnection is a parametrized connection which processes the input 
/// from the incoming module to the outgoing module by multiplying it with a
/// specific parameter for each component.  
///

class LinearConnection : public Connection, public Parametrized
{
    public: 
        
        ///
        /// Create a new LinearConnection object. The outsize of the incoming
        /// module and the insize of the outgoing module have to be equal.
        ///
        LinearConnection(Module* incoming_p, Module* outgoing_p);

        ///
        /// Create a new LinearConnection object. The slices have to be of equal
        /// length.
        ///
        LinearConnection(Module* incoming_p, Module* outgoing_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop);

        ///
        /// Create a new LinearConnection object and with the given parameters.
        // The slices have to be of equal length.
        ///
        LinearConnection(Module* incoming_p, Module* outgoing_p,
                       double* parameters_p, double* derivatives_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop);
        virtual ~LinearConnection();
        
    protected:
        
        virtual void forward_process(double* sink_p, const double* source_p);
        virtual void backward_process(double* sink_p, const double* source_p);
};    
    
}
}
}


#endif
