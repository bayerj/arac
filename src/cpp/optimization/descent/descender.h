// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_OPTIMIZATION_DESCENT_DESCENDER_INCLUDED
#define Arac_OPTIMIZATION_DESCENT_DESCENDER_INCLUDED


#include "../../structure/networks/basenetwork.h"
#include "../../structure/parametrized.h"


namespace arac {
namespace optimization {
namespace descent {
    
    
///
/// Class for performing a gradient descent on Network and Parametrized 
/// objects.
///
/// The descender watches objects and when .update() is called, inspects the 
/// the objects derivatives, possibly performing an update. (A descender might
/// decide not to update always.)
/// 
    
class Descender
{
    public:
    
        ///
        /// Create a Descender object that watches a Network object and the 
        /// contained Network and Parametrized objects.
        ///
        Descender(arac::structure::networks::BaseNetwork& net);
        
        ///
        /// Create a Descender object that watches a Parametrized object.
        ///
        Descender(arac::structure::Parametrized& parametrized);
        
        ///
        /// Destroy the Descender object.
        ///
        virtual ~Descender();
        
        /// 
        /// Return a vector of watched Parametrized objects.
        ///
        std::vector<arac::structure::Parametrized*>& targets();
        
        ///
        /// Notify the Descender object that the derivatives of the watched 
        /// objects have changed. If the Descender object performs an update, 
        /// true is returned, otherwise false.
        ///
        /// Must be implemented by subclass.
        virtual bool notify() = 0;
        
    private:
        
        std::vector<arac::structure::Parametrized*> _targets;
        
};


inline
std::vector<arac::structure::Parametrized*>&
Descender::targets()
{
    return _targets;
}
    
    
} } } // Namespace.



#endif