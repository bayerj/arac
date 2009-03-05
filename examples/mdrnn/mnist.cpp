#include "arac.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>


using namespace arac::structure::networks;
using namespace arac::structure::networks::mdrnns;
using namespace arac::structure::connections;
using namespace arac::structure::modules;
using namespace arac::optimization;
using namespace arac::datasets;
using namespace arac::utilities;


static int insize = 28 * 28;
static int outsize = 10;


SupervisedDataset<double*, double*> make_dataset()
{
    SupervisedDataset<double*, double*> ds(insize, insize * outsize);
    double* sample_p = new double[insize];
    double* target_p = new double[outsize];
    
    memset(sample_p, 0, sizeof(double) * insize);
    memset(target_p, 0, sizeof(double) * outsize * insize);
    
    ds.append(sample_p, target_p);
    return ds;
}


Network make_network()
{
    int hidden = 5;
    Network net;
    
    Mdrnn<MdlstmLayer>* inlayer_p = new Mdrnn<MdlstmLayer>(2, hidden);
    
    inlayer_p->set_sequence_shape(0, 28);
    inlayer_p->set_sequence_shape(1, 28);
    inlayer_p->set_block_shape(0, 1);
    inlayer_p->set_block_shape(1, 1);
    inlayer_p->sort();
    
    PartialSoftmaxLayer* outlayer_p = \
        new PartialSoftmaxLayer(insize * outsize, outsize);

    std::cout << "Mdrnn-Outsize: " << inlayer_p->outsize() << std::endl
              << "PSM-Insize: " << outlayer_p->insize() << std::endl
              << "Inchunk: " << hidden << std::endl
              << "Outchunk: " << outsize << std::endl
              << "#Parameters: " << hidden * outsize << std::endl
              << std::endl;

    WeightShareConnection* con_p = new WeightShareConnection(
                                        inlayer_p, outlayer_p, 
                                        hidden, outsize);

    net.add_module(inlayer_p, Network::InputModule);
    net.add_module(outlayer_p, Network::OutputModule);

    net.add_connection(con_p);
    
    net.randomize();
    
    return net;
}


int main (int argc, char const *argv[])
{
    srand(0);
    SupervisedDataset<double*, double*> ds = make_dataset();
    Network net = make_network();
    srand(0);
    SimpleBackprop optimizer = SimpleBackprop(net, ds);
    optimizer.set_learningrate(0.001);
    // print_parameters(net);
    std::cout << std::endl;
    for(int i = 0; i < 10000; i++)
    {
        optimizer.train_stochastic();
        // print_derivatives(net);
        std::cout << std::endl;
        std::cout << *optimizer.error() << std::endl;
    }
    std::cout << "Parameter: ";
    print_parameters(net);
    std::cout << std::endl;
    print_activations(net, ds);
}