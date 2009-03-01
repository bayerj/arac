// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include "stepdescender.h"


using arac::structure::Parametrized;
using arac::structure::networks::BaseNetwork;
using arac::optimization::descent::Descender;
using arac::optimization::descent::StepDescender;


StepDescender::StepDescender(BaseNetwork& network, double stepratio) :
    Descender(network),
    _stepratio(stepratio)
    
{
}


StepDescender::StepDescender(Parametrized& parametrized, double stepratio) :
    Descender(parametrized),
    _stepratio(stepratio)

{
}


StepDescender::~StepDescender()
{
}


bool
StepDescender::notify()
{
    std::vector<Parametrized*>::iterator param_iter;
    for (param_iter = targets().begin();
         param_iter != targets().end();
         param_iter++)
    {
        Parametrized& param = **param_iter;
        for (int i = 0; i < param.size(); i++)
        {
            param.get_parameters()[i] += stepratio() * param.get_derivatives()[i];
        }
    }     
    return true;
}