// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_OPTIMIZATION_DESCENT_STEPDESCENDER_INCLUDED
#define Arac_OPTIMIZATION_DESCENT_STEPDESCENDER_INCLUDED


#include "../../structure/networks/basenetwork.h"
#include "../../structure/parametrized.h"

#include "descender.h"


namespace arac {
namespace optimization {
namespace descent {
    
    
///
/// The StepDescender follows the derivatives with a step that is defined by 
/// the magnitude of the derivative times the stepratio of the StepDescender
/// object.
/// 
    
class StepDescender : public Descender
{
    public:
    
        ///
        /// Create a Descender object that watches a Network object and the 
        /// contained Network and Parametrized objects.
        ///
        StepDescender(arac::structure::networks::BaseNetwork& net, 
                      double stepratio);
        
        ///
        /// Create a Descender object that watches a Parametrized object.
        ///
        StepDescender(arac::structure::Parametrized& parametrized,
                  double stepratio);
        
        ///
        /// Destroy the Descender object.
        ///
        virtual ~StepDescender();
        
        ///
        /// Perform an update on the parameters by moving a step along the 
        /// derivative. The stepsize is given by the magnitude of the derivative
        /// and the stepratio.
        ///
        virtual bool notify();
        
        ///
        /// Return the current stepratio.
        ///
        double stepratio();
        
        ///
        /// Set the current stepratio.
        ///
        void set_stepratio(double stepratio);
        
        
    private:
        
        double _stepratio;
        
};


inline
double
StepDescender::stepratio()
{
    return _stepratio;
}

    
} } } // Namespace.



#endif