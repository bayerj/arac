// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_CONNECTIONS_CONVOLUTIONAL_INCLUDED
#define Arac_STRUCTURE_CONNECTIONS_CONVOLUTIONAL_INCLUDED


#include "../modules/module.h"
#include "../parametrized.h"
#include "connection.h"


namespace arac {
namespace structure {
namespace connections {

    
using namespace arac::structure::modules;
using arac::structure::Parametrized;


///
/// TOOD: docu,ent
///

class ConvolutionalConnection : public Connection, public Parametrized
{
    public: 
   
        ///
        /// Create a new ConvolutionalConnection object.
        ///
        ConvolutionalConnection(Module* incoming_p, Module* outgoing_p, 
                                int inchunk, int outchunk);

        ///
        /// Create a new ConvolutionalConnection object and use the
        /// supplied arrays as parameters and derivatives.
        ///
        ConvolutionalConnection(Module* incoming_p, Module* outgoing_p,
                       double* parameters_p, double* derivatives_p);
           
        ///            
        /// Destroy the ConvolutionalConnection object.
        ///
        virtual ~ConvolutionalConnection();
        
    protected:
        
        virtual void _forward();
        virtual void _backward();
};    
    
}
}
}


#endif