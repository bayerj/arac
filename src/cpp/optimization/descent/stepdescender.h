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
                      double stepratio,
                      double momentum=0.0);
        
        ///
        /// Create a Descender object that watches a Parametrized object.
        ///
        StepDescender(arac::structure::Parametrized& parametrized,
                      double stepratio,
                      double momentum=0.0);
        
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
        void set_stepratio(const double stepratio);
        
        
        /// 
        /// Return the momentum.
        ///
        double momentum();

        ///
        /// Set the momentum.
        ///
        void set_momentum(const double value);
        
    private:
        
        void init_updatehistory();
        
        double _stepratio;
        double _momentum;
        double* _updates_p;
        int _n_params;
};


inline
double
StepDescender::stepratio()
{
    return _stepratio;
}


inline
void
StepDescender::set_stepratio(const double stepratio)
{
    _stepratio = stepratio;
}


inline
double
StepDescender::momentum()
{
    return _momentum;
}


inline
void
StepDescender::set_momentum(const double value)
{
    _momentum = value;
}


} } } // Namespace.



#endif
