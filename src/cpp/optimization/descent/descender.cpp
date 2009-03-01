// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include "descender.h"


using arac::structure::Parametrized;
using arac::structure::networks::BaseNetwork;
using arac::optimization::descent::Descender;


Descender::Descender(BaseNetwork& network)
{
    std::vector<Parametrized*>::iterator param_iter;
    for (param_iter = network.parametrizeds().begin();
         param_iter != network.parametrizeds().end();
         param_iter++)
    {
        _targets.push_back(*param_iter);
    }
    
    
}


Descender::Descender(Parametrized& parametrized)
{
    _targets.push_back(&parametrized);
}


Descender::~Descender()
{
    
}