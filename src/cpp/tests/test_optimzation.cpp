#include <gtest/gtest.h>
#include <iostream>
#include "../arac.h"


namespace AracTesting {


using namespace arac::structure::modules;
using namespace arac::structure::connections;
using namespace arac::structure::networks;
using namespace arac::datasets;
using namespace arac::optimization;
using namespace arac::optimization::descent;


TEST(TestBackprop, TestStochasticStepDoubleDouble)
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


TEST(TestBackprop, TestStochasticStepSequenceDouble)
{
    LinearLayer* inlayer_p = new LinearLayer(1);
    LinearLayer* outlayer_p = new LinearLayer(1);
    inlayer_p->set_mode(Component::Sequential);
    outlayer_p->set_mode(Component::Sequential);
    
    FullConnection* con_p = new FullConnection(inlayer_p, outlayer_p);
    con_p->get_parameters()[0] = 2.0; 
    con_p->set_mode(Component::Sequential);
    
    Network* net_p= new Network();
    net_p->add_module(inlayer_p, Network::InputModule);
    net_p->add_module(outlayer_p, Network::OutputModule);
    net_p->add_connection(con_p);
    net_p->set_mode(Component::Sequential);
    
    double* seq_p = new double[2];
    seq_p[0] = 1;
    seq_p[1] = 2;
    Sequence seq(2, 1, seq_p);
    
    double* target_p = new double[1];
    target_p[0] = 3.0;
    
    SupervisedDataset<Sequence, double*> ds(1, 1);
    ds.append(seq, target_p);
    
    SemiSequentialBackprop trainer(*net_p, ds);
    trainer.set_learningrate(1.);
    trainer.train_stochastic();

    EXPECT_DOUBLE_EQ(-9, con_p->get_derivatives()[0])
        << "Derivatives not correct.";

    EXPECT_DOUBLE_EQ(-7, con_p->get_parameters()[0])
        << "Parameters not learned correctly.";
}


TEST(TestBackprop, TestStochasticStepSequenceSequence)
{
    LinearLayer* inlayer_p = new LinearLayer(1);
    LinearLayer* outlayer_p = new LinearLayer(1);
    inlayer_p->set_mode(Component::Sequential);
    outlayer_p->set_mode(Component::Sequential);

    FullConnection* con_p = new FullConnection(inlayer_p, outlayer_p);
    con_p->get_parameters()[0] = 1.5; 
    con_p->set_mode(Component::Sequential);
    
    Network* net_p= new Network();
    net_p->add_module(inlayer_p, Network::InputModule);
    net_p->add_module(outlayer_p, Network::OutputModule);
    net_p->add_connection(con_p);
    net_p->set_mode(Component::Sequential);
    
    double* inseq_p = new double[2];
    inseq_p[0] = 1;
    inseq_p[1] = 2;
    Sequence inseq(2, 1, inseq_p);
    
    double* outseq_p = new double[2];
    outseq_p[0] = 2;
    outseq_p[1] = 4;
    Sequence outseq(2, 1, outseq_p);
    
    SupervisedDataset<Sequence, Sequence> ds(1, 1);
    ds.append(inseq, outseq);
    
    SequentialBackprop trainer(*net_p, ds);
    trainer.set_learningrate(1.);
    trainer.train_stochastic();

    EXPECT_DOUBLE_EQ(2.5, con_p->get_derivatives()[0])
        << "Derivatives not correct.";

    EXPECT_DOUBLE_EQ(4, con_p->get_parameters()[0])
        << "Parameters not learned correctly.";
}


TEST(TestDescent, TestStepDescent)
{
    double params_p[5] = {1, 2, 3, 4, 5};
    double derivs_p[5] = {-1, -2, -3, -4, -5};
    Parametrized p(5, params_p, derivs_p);
    
    StepDescender d(p, 0.5);
    
    d.notify();
    
    double solution_p[5] = {0.5, 1, 1.5, 2, 2.5};
    for (int i = 0; i < 5; i++)
    {
        EXPECT_EQ(solution_p[i], params_p[i])
            << "Wrong update at " << i;
    }

    d.notify();
    
    double solution2_p[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 5; i++)
    {
        EXPECT_EQ(solution2_p[i], params_p[i])
            << "Wrong update at " << i;
    }
}


TEST(TestDescent, TestStepDescentMomentum)
{
    double params_p[1] = {0};
    double derivs_p[1] = {1};
    Parametrized p(1, params_p, derivs_p);
    
    StepDescender d(p, 0.5, 0.5);
    
    d.notify();
    
    EXPECT_EQ(0.5, params_p[0]) << "Wrong update at first step.";

    derivs_p[0] = 2;
    d.notify();
    
    EXPECT_EQ(1.75, params_p[0]) << "Wrong update at second step.";
}

     
// TODO: write a test for contained networks.
        
}  // namespace
