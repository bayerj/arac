// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_CONNECTIONS_PERMUTATION_INCLUDED
#define Arac_STRUCTURE_CONNECTIONS_PERMUTATION_INCLUDED


#include "../modules/module.h"
#include "connection.h"


namespace arac {
namespace structure {
namespace connections {

    
using namespace arac::structure::modules;


///
/// TOOD: document
///

class PermutationConnection : public Connection
{
    public: 
   
        ///
        /// Create a new PermutationConnection object.
        ///
        PermutationConnection(Module* incoming_p, Module* outgoing_p);
        
        PermutationConnection(Module* incoming_p, Module* outgoing_p, 
                              std::vector<int> permutation);

        ///            
        /// Destroy the PermutationConnection object.
        ///
        virtual ~PermutationConnection();
        
        const std::vector<int>& permutation() const;
        std::vector<int>& permutation();
        
        void set_permutation(std::vector<int> perm);
        
        ///
        /// Invert the current permutation.
        ///
        void invert();
        
    protected:
        std::vector<int> _permutation;
        
        void forward_process(double* sink_p, const double* source_p);
        void backward_process(double* sink_p, const double* source_p);
};    


inline
const std::vector<int>& 
PermutationConnection::permutation() const
{
    return _permutation;
}


inline
std::vector<int>& 
PermutationConnection::permutation()
{
    return _permutation;
}


inline
void
PermutationConnection::set_permutation(std::vector<int> permutation)
{
    assert(permutation.size() == incoming()->outsize());
    _permutation = permutation;
}

    
}
}
}


#endif
