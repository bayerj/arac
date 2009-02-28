// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_ERRORSCALING_INCLUDED
#define Arac_STRUCTURE_MODULES_ERRORSCALING_INCLUDED


#include "linear.h"


namespace arac {
namespace structure {
namespace modules {


using arac::structure::modules::LinearLayer;


///
/// A ErrorScalingLayer object is of equal in- and outputsize. It does not 
/// transform, the input in any way - but the error is multiplied by constants
/// determined by a vector during the backward pass.
///

class ErrorScalingLayer : public LinearLayer
{
    public:

        /// 
        /// Create a ErrorScalingLayer object of the given size.
        ///
        ErrorScalingLayer(int size, std::vector<double> scale);
        virtual ~ErrorScalingLayer();

    protected:

        virtual void _backward();
        
        std::vector<double> _scale;
};


}
}
}


#endif