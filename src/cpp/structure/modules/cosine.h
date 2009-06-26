// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_COSINE_INCLUDED
#define Arac_STRUCTURE_MODULES_COSINE_INCLUDED


#include "module.h"
#include "../../common/common.h"


namespace arac {
namespace structure {
namespace modules {


using arac::structure::modules::Module;


///
/// A CosineLayer object is of equal in- and outputsize. It does not transform
/// the input in any way.
///

class CosineLayer : public Module
{
    public:

        /// 
        /// Create a CosineLayer object of the given size.
        ///
        CosineLayer(int size);
        virtual ~CosineLayer();

    protected:

        virtual void _forward();
        virtual void _backward();
        virtual void expand();

    private:

        arac::common::Buffer* _interbuffer_p;
        arac::common::Buffer* _intererror_p;
};



inline CosineLayer::CosineLayer(int size) :
    Module(size, size),
    _interbuffer_p(new arac::common::Buffer(size)),
    _intererror_p(new arac::common::Buffer(size))
{
}


}
}
}


#endif
