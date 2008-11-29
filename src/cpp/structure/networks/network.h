// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_NETWORKS_NETWORK_INCLUDED
#define Arac_STRUCTURE_NETWORKS_NETWORK_INCLUDED


#include <vector>
#include "../component.h"
#include "../connections/connection.h"
#include "../modules/module.h"


namespace arac {
namespace structure {
namespace network {
    
    
using namespace arac::structure::modules;
using namespace arac::structure::connections;


class Network : public arac::structure::Component
{
    public: 
        
        Network();
        ~Network();
        
        void add_module(Module* module_p);
        
        virtual void forward();
        virtual void backward();
        
    private:
        
        bool _dirty;
        std::vector<Module*> _modules;
        std::vector<Connection*> _connections;
    };
 
    
}
}
}


#endif