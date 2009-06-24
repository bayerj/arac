// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_CONNECTIONS_CONVOLVE_INCLUDED
#define Arac_STRUCTURE_CONNECTIONS_CONVOLVE_INCLUDED


#include "../modules/module.h"
#include "../parametrized.h"
#include "connection.h"
#include "full.h"


namespace arac {
namespace structure {
namespace connections {

    
using namespace arac::structure::modules;
using arac::structure::Parametrized;


///
/// TODO: document
///

class ConvolveConnection : public Connection, public Parametrized
{
    public: 
   
        ///
        /// Create a new ConvolveConnection object.
        ///
        ConvolveConnection(Module* incoming_p, Module* outgoing_p, 
                              int inchunk, int outchunk);

        ///
        /// Create a new ConvolveConnection object and use the
        /// supplied arrays as parameters and derivatives.
        ///
        ConvolveConnection(Module* incoming_p, Module* outgoing_p,
                       double* parameters_p, double* derivatives_p);
           
        ///            
        /// Destroy the ConvolveConnection object.
        ///
        virtual ~ConvolveConnection();
        
    protected:
        virtual void forward_process(double* sink_p, const double* source_p);
        virtual void backward_process(double* sink_p, const double* source_p);

        int _n_chunks;
        int _inchunk;
        int _outchunk;
};    
    
}
}
}


#endif
