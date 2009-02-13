// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_NETWORKS_MDRNNS_INCLUDED
#define Arac_STRUCTURE_NETWORKS_MDRNNS_INCLUDED


#include "mdrnn.h"
#include "basemdrnn.h"
#include "../../modules/modules.h"


typedef Mdrnn<arac::structure::modules::LinearLayer> LinearMdrnn;
typedef Mdrnn<arac::structure::modules::SigmoidLayer> SigmoidMdrnn;
typedef Mdrnn<arac::structure::modules::TanhLayer> TanhMdrnn;


#endif