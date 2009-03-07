// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <cstring>

#include "stepdescender.h"


using arac::structure::Parametrized;
using arac::structure::networks::BaseNetwork;
using arac::optimization::descent::Descender;
using arac::optimization::descent::StepDescender;


StepDescender::StepDescender(BaseNetwork& network, 
                             double stepratio, double momentum) :
    Descender(network),
    _stepratio(stepratio),
    _momentum(momentum)
{
    init_updatehistory();
}


StepDescender::StepDescender(Parametrized& parametrized, 
                             double stepratio, double momentum) :
    Descender(parametrized),
    _stepratio(stepratio),
    _momentum(momentum)
{
    init_updatehistory();
}


StepDescender::~StepDescender()
{
}


void
StepDescender::init_updatehistory()
{
    _n_params = 0;
    std::vector<Parametrized*>::iterator param_iter;
    for (param_iter = targets().begin(); 
         param_iter != targets().end();
         param_iter++)
    {
        _n_params += (*param_iter)->size();
    }
    _updates_p = new double[_n_params];
    memset(_updates_p, 0, sizeof(double) * _n_params);
}


bool
StepDescender::notify()
{
    std::vector<Parametrized*>::iterator param_iter;
    int i;
    for (i = 0, param_iter = targets().begin();
         param_iter != targets().end();
         param_iter++)
    {
        Parametrized& param = **param_iter;
        for (int j = 0; j < param.size(); j++, i++)
        {
            // Calculate current update.
            double update = stepratio() * param.get_derivatives()[j];
            // Add momentum term.
            update += momentum() * _updates_p[i];
            // Save current update.
            _updates_p[i] = update;
            // Apply update.
            param.get_parameters()[j] += update;
        }
    }     
    return true;
}
