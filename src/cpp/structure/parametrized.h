// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_PARAMETRIZED_INCLUDED
#define Arac_STRUCTURE_PARAMETRIZED_INCLUDED


namespace arac {
namespace structure {
    

///
/// Classes that inherit from parametrized gain the ability to hold learnable
/// parameters.
///

class Parametrized 
{
    public: 

        ///
        /// Create a new Parametrized object.
        ///
        Parametrized();
        
        ///
        /// Create a new Parametrized object of the given size.
        ///
        Parametrized(int size);
        
        ///
        /// Wrap the given array with a new Parametrized object of the given
        /// size.
        Parametrized(int size, double* parameters_p, double* derivatives_p);
        
        ///
        /// Destroy the Parametrized object.
        ///
        virtual ~Parametrized();
        
        /// 
        /// Return a pointer to the parameters of the object.
        ///
        double* get_parameters() const;

        /// 
        /// Set the pointer to the parameters of the object.
        ///
        void set_parameters(double* parameters_p);
        
        /// 
        /// Return a pointer to the derviatives of the object.
        ///
        double* get_derivatives() const;
        
        /// 
        /// Set the pointer to the parameters of the object.
        ///
        void set_derivatives(double* derivatives_p);

        ///
        /// Set all the derivatives to zero.
        ///
        virtual void clear_derivatives();
        
        ///
        /// Tell wether the objects owns the parameters.
        ///
        bool parameters_owner();
        
        ///
        /// Tell wether the object owns the derivatives.
        ///
        bool derivatives_owner();
        
        ///
        /// Return the number of parameters.
        ///
        int size();
        
        /// Set the parameters to random values in (-interval, +interval).
        void randomize(double interval=0.1);
        
    protected:
        
        int _size;
        double* _parameters_p;
        double* _derivatives_p;
        bool _parameters_owner;
        bool _derivatives_owner;
};


inline
int
Parametrized::size()
{
    return _size;
}
 
 
}
}

#endif