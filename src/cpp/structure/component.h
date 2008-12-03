// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_COMPONENT_INCLUDED
#define Arac_STRUCTURE_COMPONENT_INCLUDED


namespace arac {
namespace structure {
    

class Component 
{
    public: 
        
        enum Mode 
        {
            Simple = 0,
            ErrorAgnostic = 1,
            Sequential = 2,
            
            SequentialErrorAgnostic = 3,
        };
        
        Component();
        
        virtual ~Component();
        
        virtual void forward();
        virtual void backward();

        // Set the mode of the module.
        virtual void set_mode(Mode mode);
        
        // Get the mode of the module.
        Mode get_mode();
        
        // Tell if the module is sequential.
        bool sequential();
        
        // Tell if the module is error agnostic.
        bool error_agnostic();
        
        // Return the current timestep.
        int timestep();
        
    protected:

        virtual void _forward() = 0;
        virtual void _backward() = 0;
        
        int _timestep;
        Mode _mode;
        
};


inline Component::Component() : 
    _timestep(0),
    _mode(Component::Simple)
{
}

inline Component::~Component() {}


inline
void
Component::forward()
{
    _forward();
    if (sequential())
    {
        _timestep += 1;
    }
}


inline
void 
Component::backward()
{
    _backward();
    if (sequential())
    {
        _timestep -= 1;
    }
}



inline 
Component::Mode
Component::get_mode()
{
    return _mode;
}


inline 
void
Component::set_mode(Component::Mode mode)
{
    _mode = mode;
}


inline
bool
Component::sequential()
{
    return _mode & Component::Sequential;
}


inline
bool 
Component::error_agnostic()
{
    return _mode & Component::ErrorAgnostic;
}


inline
int
Component::timestep()
{
    return _timestep;
}
    
}
}


#endif