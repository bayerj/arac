// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_NETWORKS_NETWORK_INCLUDED
#define Arac_STRUCTURE_NETWORKS_NETWORK_INCLUDED


#include <vector>
#include <map>

#include "../component.h"
#include "../parametrized.h"
#include "../connections/connection.h"
#include "../modules/module.h"
#include "basenetwork.h"


namespace arac {
namespace structure {
namespace networks {
    
    
using namespace arac::structure::modules;
using namespace arac::structure::connections;
using arac::structure::Component;
using arac::structure::Parametrized;


///
/// Network objects hold a graph of modules as nodes and connections as edges
/// to process an output.
///
class Network : public BaseNetwork
{
    public: 
        
        ///
        /// Different roles that modules can take in a Network object.
        ///
        enum ModuleType {
            // Modules of this type have no special role.
            Simple = 0,
            // The input of modules of this type is also input of the network.
            InputModule = 1,
            // The output of modules of this type is also output of the network.
            OutputModule = 2,
            // Combination of InputModule and OutputModule.
            InputOutputModule = 3
        };
        
        ///
        /// Create a new Network object.
        ///
        Network();
        
        ///
        /// Destroy the Network object.
        ///
        virtual ~Network();
        
        ///
        /// Set the buffers of all the modules in the network to zero.
        ///
        virtual void clear();

        ///
        /// Add a module of the given type to the Network object.
        ///
        void add_module(Module* module_p, ModuleType type=Simple);
        
        ///
        /// Add a connection to the Network object.
        ///
        void add_connection(Connection* con_p);

        virtual void sort();
        
    protected:
        
        virtual void _forward();
        virtual void _backward();
        
        void add_component(Component* comp_p);

        ///
        /// Fill count with the amount of incoming edges for every module.
        ///
        void incoming_count(std::map<Module*, int>& count);
        
        void init_buffers();

        std::vector<Module*> _inmodules;
        std::vector<Module*> _outmodules;
        std::vector<Component*> _components_sorted;
        std::vector<Component*> _components_rec;
        std::map<Module*, ModuleType> _modules;
        std::vector<Connection*> _connections;
        std::map<Module*, std::vector<Connection*> > _outgoing_connections;
};
 
    
}
}
}


#endif