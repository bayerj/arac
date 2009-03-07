// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_CONNECTIONS_WEIGHTSHARE_INCLUDED
#define Arac_STRUCTURE_CONNECTIONS_WEIGHTSHARE_INCLUDED


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
/// TOOD: docu,ent
///

class WeightShareConnection : public Connection, public Parametrized
{
    public: 
   
        ///
        /// Create a new ConvolutionalConnection object.
        ///
        WeightShareConnection(Module* incoming_p, Module* outgoing_p, 
                              int inchunk, int outchunk);

        ///
        /// Create a new ConvolutionalConnection object and use the
        /// supplied arrays as parameters and derivatives.
        ///
        WeightShareConnection(Module* incoming_p, Module* outgoing_p,
                       double* parameters_p, double* derivatives_p);
           
        ///            
        /// Destroy the ConvolutionalConnection object.
        ///
        virtual ~WeightShareConnection();
        
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
