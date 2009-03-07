// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_CONNECTIONS_IDENTITY_INCLUDED
#define Arac_STRUCTURE_CONNECTIONS_IDENTITY_INCLUDED


#include "../modules/module.h"
#include "connection.h"


namespace arac {
namespace structure {
namespace connections {

    
using namespace arac::structure::modules;
using arac::structure::Component;


///
/// An identity connection connects two modules by adding the output of the
/// íncoming module to the input of the outgoing module.
///
class IdentityConnection : public Connection 
{
    public:
        
        ///
        /// Create a new IdentityConnection between two modules, where the
        /// output size of the incoming module equals the input size of the 
        /// outgoing module.
        ///
        IdentityConnection(Module* incoming_p, Module* outgoing_p);

        ///
        /// Create a new IdentityConnection between two modules. Slices have to
        /// be of the same length.
        ///
        IdentityConnection(Module* incoming_p, Module* outgoing_p,
                           int incomingstart, int incomingstop, 
                           int outgoingstart, int outgoingstop);
        virtual ~IdentityConnection();
    
    protected:
        
        virtual void forward_process(double* sink_p, const double* source_p);
        virtual void backward_process(double* sink_p, const double* source_p);
};


}
}
}


#endif
