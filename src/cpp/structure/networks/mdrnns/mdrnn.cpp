// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include "../../modules/modules.h"
#include "mdrnn.h"


namespace arac {
namespace structure {
namespace networks {
namespace mdrnns {
    

using namespace arac::structure::modules;


template <>
void
Mdrnn<LinearLayer>::init_structure()
{
    _inmodule_p = new LinearLayer(blocksize());

    _module_p = new LinearLayer(_hiddensize);
    _module_p->set_mode(Component::Sequential);

    FullConnection* feedcon_p = \
        new FullConnection(_inmodule_p, _module_p, 
                           0, blocksize(),
                           0, _hiddensize);
    feedcon_p->set_mode(Component::Sequential);
    _feedcon_p = feedcon_p;
   
    // Add a connection from the bias.
    _biascon_p = new FullConnection(&_bias, _module_p);
    _biascon_p->set_mode(Component::Sequential);
}


template <>
void
Mdrnn<TanhLayer>::init_structure()
{
    _inmodule_p = new LinearLayer(blocksize());

    _module_p = new TanhLayer(_hiddensize);
    _module_p->set_mode(Component::Sequential);

    FullConnection* feedcon_p = \
        new FullConnection(_inmodule_p, _module_p, 
                           0, blocksize(),
                           0, _hiddensize);
    feedcon_p->set_mode(Component::Sequential);
    _feedcon_p = feedcon_p;
   
    // Add a connection from the bias.
    _biascon_p = new FullConnection(&_bias, _module_p);
    _biascon_p->set_mode(Component::Sequential);
}


template <>
void
Mdrnn<SigmoidLayer>::init_structure()
{
    _inmodule_p = new LinearLayer(blocksize());

    _module_p = new SigmoidLayer(_hiddensize);
    _module_p->set_mode(Component::Sequential);

    FullConnection* feedcon_p = \
        new FullConnection(_inmodule_p, _module_p, 
                           0, blocksize(),
                           0, _hiddensize);
    feedcon_p->set_mode(Component::Sequential);
    _feedcon_p = feedcon_p;
   
    // Add a connection from the bias.
    _biascon_p = new FullConnection(&_bias, _module_p);
    _biascon_p->set_mode(Component::Sequential);
}


template <>
void
Mdrnn<MdlstmLayer>::init_structure()
{
    _inmodule_p = new LinearLayer(blocksize());

    _module_p = new MdlstmLayer(_timedim, _hiddensize);
    _module_p->set_mode(Component::Sequential);
    // Use a FullConnection pointer first so it can be appended to the 
    // parametrizeds vector.
    FullConnection* feedcon_p = \
        new FullConnection(_inmodule_p, _module_p, 
                           0, blocksize(),
                           0, (3 + _timedim) * _hiddensize);
    feedcon_p->set_mode(Component::Sequential);
    _feedcon_p = feedcon_p;
    
    // Add a connection from the bias.
    int full_con_instart = 0;
    int full_con_instop = _hiddensize;
    int full_con_outstart = 0;
    int full_con_outstop = (3 + _timedim) * _hiddensize;
    _biascon_p = new FullConnection(&_bias, _module_p, 
                                    0, 1, 
                                    full_con_outstart, full_con_outstop);
    _biascon_p->set_mode(Component::Sequential);
}


template <>
void
Mdrnn<MdlstmLayer>::sort()
{
    // Initialize multilied sizes.
    init_multiplied_sizes();
    update_sizes();

    delete_structure();
    init_structure();
    init_con_vectors();
    
    // Also clear the parametrized vector.
    _parametrizeds.clear();
    _parametrizeds.push_back(_feedcon_p);
    _parametrizeds.push_back(_biascon_p);
    
    // Initialize recurrent self connections.
    int recurrency = 1;
    
    int full_con_instart = 0;
    int full_con_instop = _hiddensize;
    int full_con_outstart = 0;
    int full_con_outstop = (3 + _timedim) * _hiddensize;
    
    int id_con_instart = full_con_instop;
    int id_con_instop = _module_p->outsize();
    int id_con_outstart = full_con_outstop;
    int id_con_outstop = _module_p->insize();
    
    for(int i = 0; i < _timedim; i++)
    {
        FullConnection* fcon_p = \
            new FullConnection(_module_p, _module_p,
                               full_con_instart, full_con_instop,
                               full_con_outstart, full_con_outstop);
        fcon_p->set_mode(Component::Sequential);
        fcon_p->set_recurrent(recurrency);
        _connections[i].push_back(fcon_p);
        _parametrizeds.push_back(fcon_p);
        
        IdentityConnection* icon_p = \
            new IdentityConnection(_module_p, _module_p,
                                    id_con_instart, id_con_instop,
                                    id_con_outstart, id_con_outstop);
        icon_p->set_mode(Component::Sequential);
        icon_p->set_recurrent(recurrency);
        _connections[i].push_back(icon_p);
        
        // Multiply with the current blocks-per-dimension so that each 
        // connections jumps over one dimension.
        recurrency *= _sequence_shape_p[i] / _block_shape_p[i];
    }

    // Ininitialize buffers.
    init_buffers();
    
    // Indicate that the net is ready for use.
    _dirty = false;
}


} } } } // Namespace.
