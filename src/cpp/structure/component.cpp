// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include "component.h"

namespace arac {
namespace structure {


Component::Component() : 
    _timestep(0),
    _sequencelength(0),
    _mode(Component::Simple)
{
}


Component::~Component() {}


int
Component::timestep()
{
    return _timestep;
}

}
}