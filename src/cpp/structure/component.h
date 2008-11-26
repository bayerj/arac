// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_COMPONENT_INCLUDED
#define Arac_STRUCTURE_COMPONENT_INCLUDED


namespace arac {
namespace structure {
    

class Component 
{
    public: 
        
        Component();
        ~Component();
        
        virtual void forward() = 0;
        virtual void backward() = 0;

        int _timestep;
};


inline Component::Component() : _timestep(0) {}

inline Component::~Component() {}

    
}
}


#endif