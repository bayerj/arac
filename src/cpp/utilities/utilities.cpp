// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include "utilities.h"
#include <iostream>

using namespace arac::structure::networks;
using namespace arac::structure;
using namespace arac::datasets;


namespace arac {
    
namespace utilities {
    
    
void
print_array(const double* array, int length)
{
    for (int i = 0; i < length; i++)
    {
        std::cout << array[i] << " ";
    }
}
    
    
void
print_parameters(arac::structure::networks::Network& net)
{
    std::vector<Parametrized*>::const_iterator param_p_iter;
    for(param_p_iter = net.parametrizeds().begin();
        param_p_iter != net.parametrizeds().end();
        param_p_iter++)
    {
        print_array((*param_p_iter)->get_parameters(), (*param_p_iter)->size());
    }
}


void
print_derivatives(arac::structure::networks::Network& net)
{
    std::vector<Parametrized*>::const_iterator param_p_iter;
    for(param_p_iter = net.parametrizeds().begin();
        param_p_iter != net.parametrizeds().end();
        param_p_iter++)
    {
        print_array((*param_p_iter)->get_derivatives(), (*param_p_iter)->size());
    }
}

} } // Namespace