#include <gtest/gtest.h>
#include <iostream>
#include "../arac.h"


namespace AracTesting {


using namespace arac::structure::modules;
using namespace arac::structure::connections;
using namespace arac::structure::networks;
using namespace arac::datasets;
using namespace arac::optimization;


TEST(TestBackprop, TestStochasticStep)
{
    LinearLayer* inlayer_p = new LinearLayer(1);
    LinearLayer* outlayer_p = new LinearLayer(2);
    
    FullConnection* con_p = new FullConnection(inlayer_p, outlayer_p);
    con_p->get_parameters()[0] = 1; 
    con_p->get_parameters()[1] = 2.5; 
    
    Network* net_p= new Network();
    net_p->add_module(inlayer_p, Network::InputModule);
    net_p->add_module(outlayer_p, Network::OutputModule);
    net_p->add_connection(con_p);
    
    SupervisedDataset<double*, double*> ds(1, 2);
    double* row_p = new double[3];
    row_p[0] = 1;
    row_p[1] = 2;
    row_p[2] = 3;
    ds.append(row_p, row_p + 1);
    
    SimpleBackprop trainer(*net_p, ds);

    trainer.set_learningrate(1.);
    trainer.train_stochastic();

    EXPECT_DOUBLE_EQ(1, con_p->get_derivatives()[0])
        << "Derivatives not correct.";

    EXPECT_DOUBLE_EQ(0.5, con_p->get_derivatives()[1])
        << "Derivatives not correct.";
    
    EXPECT_DOUBLE_EQ(2, con_p->get_parameters()[0])
        << "Parameters not learned correctly.";

    EXPECT_DOUBLE_EQ(3, con_p->get_parameters()[1])
        << "Parameters not learned correctly.";
}
        
}  // namespace