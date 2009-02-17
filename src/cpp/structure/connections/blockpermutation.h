// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_CONNECTIONS_BLOCKPERMUTATION_INCLUDED
#define Arac_STRUCTURE_CONNECTIONS_BLOCKPERMUTATION_INCLUDED


#include "../modules/module.h"
#include "permutation.h"


namespace arac {
namespace structure {
namespace connections {

    
using namespace arac::structure::modules;


///
/// TOOD: document
///

class BlockPermutationConnection : public PermutationConnection
{
    public: 
   
        ///
        /// Create a new PermutationConnection object.
        ///
        BlockPermutationConnection(Module* incoming_p, Module* outgoing_p, 
                                   std::vector<int> sequence_shape,
                                   std::vector<int> block_shape);

        ///            
        /// Destroy the PermutationConnection object.
        ///
        virtual ~BlockPermutationConnection();
};    

    
}
}
}


#endif