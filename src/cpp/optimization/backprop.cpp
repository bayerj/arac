// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>
#include <typeinfo>

#include "backprop.h"
#include "../structure/parametrized.h"


using arac::optimization::SimpleBackprop;
using arac::structure::Parametrized;


SimpleBackprop::SimpleBackprop(Network& network, 
               SupervisedDataset<double*, double*>& dataset) :
               Backprop<double*, double*>(network, dataset)
{
    
}   
           
SimpleBackprop::~SimpleBackprop()
{
    
}