// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_OPTIMIZER_BACKPROP_INCLUDED
#define Arac_OPTIMIZER_BACKPROP_INCLUDED


#include "../structure/networks/network.h"
#include "../datasets/dataset.h"


using arac::structure::networks::Network;
using arac::datasets::Dataset;


namespace arac {
namespace optimization {
    

// TODO: document.

class Backprop 
{
    public: 
    
        Backprop(Network& network, Dataset& dataset);
        virtual ~Backprop();
        
        void train_stochastic();
        
        Network& network();
        
        Dataset& dataset();
        
        const double& learningrate();
        
        void set_learningrate(const double value);
        
    protected:
        
        void process_sample(const double* input_p, const double* target_p);
        void learn();
        
        Network& _network;
        Dataset& _dataset;
        double _learningrate;
};


inline
Network&
Backprop::network()
{
    return _network;
}


inline
Dataset&
Backprop::dataset()
{
    return _dataset;
}
 
 
inline
const double&
Backprop::learningrate()
{
    return _learningrate;
}
 
 
inline
void
Backprop::set_learningrate(const double value)
{
    _learningrate = value;
}

 
}
}

#endif