// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_NETWORKS_BASENETWORK_INCLUDED
#define Arac_STRUCTURE_NETWORKS_BASENETWORK_INCLUDED


#include "../modules/module.h"


namespace arac {
namespace structure {
namespace networks {


// TODO: document.

class BaseNetwork : public arac::structure::modules::Module
{

    public:
    
        BaseNetwork();
        virtual ~BaseNetwork();
    
        virtual const double* activate(const double* input_p);
        virtual const double* back_activate(const double* error_p);

        virtual void activate(const double* input_p, double* output_p);
        virtual void back_activate(const double* outerror_p, double* inerror_p);
        
        virtual void forward();
        
    protected:
        
        bool _dirty;
        
        virtual void sort() = 0;
};


}
}
}


#endif