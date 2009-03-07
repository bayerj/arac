// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_CONNECTIONS_FULL_INCLUDED
#define Arac_STRUCTURE_CONNECTIONS_FULL_INCLUDED


#include "../modules/module.h"
#include "../parametrized.h"
#include "connection.h"


namespace arac {
namespace structure {
namespace connections {

    
using namespace arac::structure::modules;
using arac::structure::Parametrized;


///
/// FullConnection objects connect two layers by connecting each component of 
/// the input to each component of the output, weighted by an individual 
/// parameter.
///

class FullConnection : public Connection, public Parametrized
{
    public: 
   
        ///
        /// Create a new FullConnection object.
        ///
        FullConnection(Module* incoming_p, Module* outgoing_p);

        ///
        /// Create a new FullConnection object of the given slices.
        ///
        FullConnection(Module* incoming_p, Module* outgoing_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop);
 
        ///
        /// Create a new FullConnection object of the given slices and use the
        /// supplied arrays as parameters and derivatives.
        ///
        FullConnection(Module* incoming_p, Module* outgoing_p,
                       double* parameters_p, double* derivatives_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop);
           
        ///            
        /// Destroy the FullConnection object.
        ///
        virtual ~FullConnection();
        
    protected:
        
        virtual void forward_process(double* sink_p, const double* source_p);
        virtual void backward_process(double* sink_p, const double* source_p);
};    
    
}
}
}


#endif
