// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include "parametrized.h"


using arac::structure::Parametrized;


Parametrized::Parametrized() 
{
    
}


Parametrized::Parametrized(int size) {
    _parameters_p = new double[size];
    _derivatives_p = new double[size];
    _size = size;
}


Parametrized::~Parametrized() {
    if (parameters_owner())
    {
        delete _parameters_p;
    }
    if (derivatives_owner())
    {
        delete _derivatives_p;
    }
}


bool
Parametrized::parameters_owner()
{
    return _parameters_owner;
}


bool
Parametrized::derivatives_owner()
{
    return _derivatives_owner;
}


double*
 Parametrized::get_parameters() const
{
    return _parameters_p;
}


void
Parametrized::set_parameters(double* parameters_p)
{
    if (parameters_owner())
    {
        delete _parameters_p;
    }
    _parameters_p = parameters_p;
    _parameters_owner = false;
}
 
    
double*
Parametrized::get_derivatives() const
{
    return _derivatives_p;
}


void
Parametrized::set_derivatives(double* derivatives_p)
{
    if (derivatives_owner())
    {
        delete _derivatives_p;
    }
    _derivatives_p = derivatives_p;
    _derivatives_owner = false;
}


