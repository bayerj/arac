// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_NETWORKS_BASENETWORK_INCLUDED
#define Arac_STRUCTURE_NETWORKS_BASENETWORK_INCLUDED


#include "../modules/module.h"
#include "../parametrized.h"


namespace arac {
namespace structure {
namespace networks {


///
/// Base class for all kinds of networks. Networks are supposed to organize 
/// components in a certain way to allow them to interact in a way defined by
/// the network.
///

class BaseNetwork : public arac::structure::modules::Module
{

    public:
    
        BaseNetwork();
        virtual ~BaseNetwork();
    
        ///
        /// Copy the given input into the networks input buffer and call
        /// forward, returning a pointer to the result.
        ///
        virtual const double* activate(const double* input_p);
        
        ///
        /// Copy the given error into the networks input buffer and call
        /// backward, returning a pointer to the inerror.
        ///
        virtual const double* back_activate(const double* error_p);

        ///
        /// Copy the given input into the networks input buffer and call
        /// forward. Copy the result into the given array.
        ///
        virtual void activate(const double* input_p, double* output_p);

        ///
        /// Copy the given error into the networks outerror and call
        /// backward. Copy the resulting inerror into the given array.
        ///
        virtual void back_activate(const double* outerror_p, double* inerror_p);
        
        virtual void forward();

        ///
        /// Initialize the network for processing. This method has to be 
        /// overwritten by subclasses.
        ///
        virtual void sort() = 0;
        
        ///
        /// Return a vector to all the Parametrized objects in the Network.
        ///
        std::vector<arac::structure::Parametrized*>& parametrizeds();
        
        ///
        /// Return a vector to all BaseNetwork objects in the Network.
        ///
        std::vector<BaseNetwork*>& networks();
        
        ///
        /// Fill the parametrizers of all Parametrized objects in the network
        /// with random values.
        ///
        // TODO: allow specification of intervals.
        void randomize();
        
        ///
        /// Set the derivatives of all the Parametrized objects in the network
        /// to zero.
        ///
        virtual void clear_derivatives();
        
    protected:
        
        bool _dirty;
        
        std::vector<arac::structure::Parametrized*> _parametrizeds;
        std::vector<BaseNetwork*> _networks;

};


inline
std::vector<Parametrized*>&
BaseNetwork::parametrizeds()
{
    return _parametrizeds;
}


inline
std::vector<BaseNetwork*>&
BaseNetwork::networks()
{
    return _networks;
}


}
}
}


#endif