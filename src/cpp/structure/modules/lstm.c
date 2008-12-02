// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include "lstm.h"


using arac::structure::modules::LstmLayer;
using arac::structure::Component;
using arac::common::Buffer;


LstmLayer::LstmLayer(int size) :
    _mdlstm(size, 1),
    Module(4 * size, size),
    _state_p(new Buffer(size)),
    _state_error_p(new Buffer(size))
{
    set_mode(Component::Sequential);
}


LstmLayer::~LstmLayer()
{
    if (_state_p)
    {
        delete _state_p;
    }
    if (_state_error_p)
    {
        delete _state_error_p;
    }
}


void LstmLayer::set_mode(Mode mode)
{
    if(!(mode & Component::Sequential))
    {
        // FIXME: Error handling
    }
    _mdlstm.set_mode(Component::Sequential);
    Component::set_mode(mode);
}


void LstmLayer::expand()
{
    state().expand();
    state_error().expand();
    Module::expand();
}


void
LstmLayer::fill_internal_input()
{
    // Copy input into internal MdlstmLayer.
    double* in_p = _mdlstm.input()[timestep()];
    memcpy((void*) in_p, input()[timestep()], insize() * sizeof(double));
}


void
LstmLayer::fill_internal_state()
{
    // Copy states into inputbuffer of internal MDLSTM; fill up with zero if
    // we are in the first timestep.
    void* state_p = (void*) (_mdlstm.input()[timestep()] + insize());
    if (timestep() > 0)
    {
        memcpy(state_p, 
               (void*) state()[timestep() - 1], 
               outsize() * sizeof(double));
    }
    else
    {
        memset(state_p, 0, outsize() * sizeof(double));
    }
}


void
LstmLayer::fill_internal_outerror()
{
    memcpy((void*) _mdlstm.outerror()[timestep()], 
           (void*) outerror()[timestep()],
           outsize() * sizeof(double));
}


void
LstmLayer::retrieve_internal_inerror()
{
    memcpy((void*) inerror()[timestep()],
           (void*) _mdlstm.inerror()[timestep()],
           insize() * sizeof(double));
}


void
LstmLayer::retrieve_internal_output()
{
    // Copy information back.
    memcpy((void*) output()[timestep()], 
           (void*) _mdlstm.output()[timestep()], 
           outsize() * sizeof(double));
}


void
LstmLayer::retrieve_internal_state()
{
    memcpy((void*) state()[timestep()], 
           (void*) (_mdlstm.output()[timestep()] + outsize()),
           outsize() * sizeof(double));
}


void 
LstmLayer::retrieve_internal_state_error()
{
    memcpy((void*) state_error()[timestep()],
           (void*) (_mdlstm.inerror()[timestep()] + 4 * outsize()),
           outsize() * sizeof(double));
}


void
LstmLayer::fill_internal_state_error()
{
    if (!last_timestep())
    {
        memcpy((void*) (_mdlstm.outerror()[timestep()] + outsize()),
               (void*) state_error()[timestep() + 1],
               outsize() * sizeof(double));
    }
    else 
    {
        memset((void*) (_mdlstm.outerror()[timestep()] + outsize()),
               0,
               outsize() * sizeof(double));
    }
}


void
LstmLayer::_forward()
{
    int inputmemorysize = sizeof(double) * _insize;
    double* in_p = _mdlstm.input()[timestep()];
    fill_internal_input();
    fill_internal_state();
    _mdlstm.forward();
    retrieve_internal_output();
    retrieve_internal_state();
}


void
LstmLayer::_backward()
{
    fill_internal_outerror();
    fill_internal_state_error();
    _mdlstm.backward();
    retrieve_internal_inerror();
    retrieve_internal_state_error();
}