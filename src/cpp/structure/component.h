// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_COMPONENT_INCLUDED
#define Arac_STRUCTURE_COMPONENT_INCLUDED


namespace arac {
namespace structure {


/// 
/// A Component object is the very basic part of an arac architecture. 
///
/// Each component is characterized by its forward and backward methods, which
/// are processing data in a network of arac components. 
///
/// The process of feeding information forward is done via the forward method, 
/// while propagating errors back is done via the backward method.
///
/// Components also have a mode: a sequential component can keep a history of
/// the processed data and thus incorporate time delay features in contrast to a
/// non sequential component. A component that is error agnostic ignores any 
/// behaviour that has to do with backpropagation. For example, it does not 
/// backwards its own errors and does not calculate derivatives.
///

class Component 
{
    public: 
        
        ///
        /// The modes a component can be in,
        ///
        enum Mode 
        {
            /// The component is not sequential and is not error agnostic.
            Simple = 0,
            /// The component ignores errors.
            ErrorAgnostic = 1,
            /// The component is sequential and does not ignore errors.
            Sequential = 2,

            /// SequentialErrorAgnostic - the component is sequential and 
            SequentialErrorAgnostic = 3,
        };
        
        Component();
        
        virtual ~Component();
        
        ///
        /// Run the forward pass of a component.
        /// 
        virtual void forward();
        
        /// 
        /// Run the forward pass of a component.
        /// 
        virtual void backward();
        
        /// 
        /// Run the side effects of a components forward pass.
        /// 
        virtual void dry_forward();
        
        /// 
        /// Run the side effects of a components backward pass.
        /// 
        virtual void dry_backward();
        
        /// 
        /// Set the mode of the module.
        virtual void set_mode(Mode mode);
        
        /// 
        /// Set the timestep to zero.
        virtual void clear();
        
        /// 
        /// Get the mode of the module.
        /// 
        Mode get_mode();
        
        /// 
        /// Tell if the module is sequential.
        /// 
        bool sequential();
        
        /// 
        /// Return the current timestep.
        /// 
        int timestep();

        /// 
        /// Return the sequence length of the current sequence.
        /// 
        int sequencelength();
        
        /// 
        /// Tell if the module is error agnostic.
        /// 
        bool error_agnostic();
        
    protected:

        /// 
        /// Run the side effects of a component before the actual forward.
        /// 
        virtual void pre_forward();

        /// 
        /// Run the side effects of a component before the actual backward.
        /// 
        virtual void pre_backward();

        /// 
        /// Run the side effects of a component after the actual forward.
        /// 
        virtual void post_forward();

        /// 
        /// Run the side effects of a component after the actual backward.
        /// 
        virtual void post_backward();

        /// 
        /// Do the actual forward. This method should be implemented by 
        /// subclasses.
        /// 
        virtual void _forward() = 0;
        
        /// 
        /// Do the actual Backward. This method should be implemented by 
        /// subclasses.
        /// 
        virtual void _backward() = 0;

    private:
        
        int _timestep;
        int _sequencelength;
        Mode _mode;
        
};


inline
void
Component::forward()
{
    pre_forward();
    _forward();
    post_forward();
}


inline
void 
Component::backward()
{
    pre_backward();
    _backward();
    post_backward();
}


inline
void
Component::dry_forward()
{
    pre_forward();
    post_forward();
}


inline void
Component::dry_backward()
{
    pre_backward();
    post_backward();
}


inline
void
Component::pre_forward()
{
    if (!sequential())
    {
        _timestep = 0;
    }
}


inline
void
Component::pre_backward()
{
    if (!sequential())
    {
        _timestep = 1;
    }
}


inline
void
Component::post_forward()
{
    if (sequential())
    {
        _timestep += 1;
        _sequencelength += 1;
    }
    else
    {
        _timestep = 1;
        _sequencelength = 1;
    }
}


inline
void
Component::post_backward()
{
    if (sequential())
    {
        _timestep -= 1;
    }
    else
    {
        _timestep = 0;
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
void
Component::clear()
{
    _timestep = 0;
}


inline
bool
Component::sequential()
{
    return _mode & Component::Sequential;
}


inline
int
Component::sequencelength()
{
    return _sequencelength;
}


inline
bool 
Component::error_agnostic()
{
    return _mode & Component::ErrorAgnostic;
}


}
}


#endif
