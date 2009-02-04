#include "arac.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>


using namespace arac::structure::networks;
using namespace arac::structure::connections;
using namespace arac::structure::modules;
using namespace arac::optimization;
using namespace arac::datasets;
using namespace arac::utilities;


SupervisedDataset<double*, double*> make_dataset()
{
    SupervisedDataset<double*, double*> ds(2, 1);
    double* data_p = new double[12];
    data_p[0] = 0;
    data_p[1] = 0;
    data_p[2] = 0;
    
    data_p[3] = 0;
    data_p[4] = 1;
    data_p[5] = 1;

    data_p[6] = 1;
    data_p[7] = 0;
    data_p[8] = 1;

    data_p[9] = 1;
    data_p[10] = 1;
    data_p[11] = 0;
    
    ds.append(data_p, data_p + 2);
    ds.append(data_p + 3, data_p + 5);
    ds.append(data_p + 6, data_p + 8);
    ds.append(data_p + 9, data_p + 11);
    
    return ds;
}


Network make_network()
{
    Network net;
    LinearLayer* inlayer_p = new LinearLayer(2);
    Bias* bias_p = new Bias();
    TanhLayer* hidden_p = new TanhLayer(3);
    LinearLayer* outlayer_p = new LinearLayer(1);
    FullConnection* con1_p = new FullConnection(inlayer_p, hidden_p);
    FullConnection* con2_p = new FullConnection(hidden_p, outlayer_p);
    FullConnection* con3_p = new FullConnection(bias_p, hidden_p);
    
    net.add_module(inlayer_p, Network::InputModule);
    net.add_module(hidden_p);
    net.add_module(bias_p);
    net.add_module(outlayer_p, Network::OutputModule);
    net.add_connection(con1_p);
    net.add_connection(con2_p);
    net.add_connection(con3_p);
    
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
    optimizer.set_learningrate(0.05);
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