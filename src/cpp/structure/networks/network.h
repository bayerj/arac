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


// TODO: document.

class Network : public BaseNetwork
{
    public: 
        
        enum ModuleType {
            Simple = 0,
            InputModule = 1,
            OutputModule = 2,
            InputOutputModule = 3
        };
        
        Network();
        virtual ~Network();
        
        virtual void clear();
        
        void add_module(Module* module_p, ModuleType type=Simple);
        
        void add_connection(Connection* con_p);

        virtual void clear_derivatives();

        const std::vector<Parametrized*>& parametrizeds() const;
        
        void randomize();
        
    protected:
        
        virtual void _forward();
        virtual void _backward();
        
        void add_component(Component* comp_p);

        // Fill count with the amount of incoming edges for every module.
        void incoming_count(std::map<Module*, int>& count);
        
        virtual void sort();
        
        void init_buffers();

        std::vector<Module*> _inmodules;
        std::vector<Module*> _outmodules;
        std::vector<Component*> _components_sorted;
        std::vector<Component*> _components_rec;
        std::map<Module*, ModuleType> _modules;
        std::vector<Connection*> _connections;
        std::map<Module*, std::vector<Connection*> > _outgoing_connections;
        std::vector<Parametrized*> _parametrizeds;
};


inline
const std::vector<Parametrized*>&
Network::parametrizeds() const
{
    return _parametrizeds;
}
 
    
}
}
}


#endif