// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_NETWORKS_MDRNNS_BASEMDRNN_INCLUDED
#define Arac_STRUCTURE_NETWORKS_MDRNNS_BASEMDRNN_INCLUDED


#include "../basenetwork.h"


namespace arac {
namespace structure {
namespace networks {
namespace mdrnns {


///
/// Baseclass for all kinds of multidimensional recurrent neural networks.
/// The common API is a time dimensions, which specifies the dimensionality of
/// the sequence. 
///

class BaseMdrnn : public BaseNetwork
{
    public:
        
        ///
        /// Create a new BaseMdrnn object.
        ///
        BaseMdrnn(int timedim);
        
        ///
        /// Destroy the BaseMdrnn object.
        ///
        virtual ~BaseMdrnn();

    protected:
        
        int _timedim;
};



}
}
}
}


#endif