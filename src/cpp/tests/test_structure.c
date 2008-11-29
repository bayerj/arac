#include <gtest/gtest.h>
#include <iostream>
#include "../arac.h"

namespace AracTesting {


using namespace arac::common;
using namespace arac::structure::modules;
using namespace arac::structure::connections;


TEST(TestCommon, TestBuffer) {
    Buffer buffer = Buffer(2);
    double* addend_p = new double[2];
    addend_p[0] = 1.2;
    addend_p[1] = 2.4;
    
    EXPECT_DOUBLE_EQ(0, buffer.current()[0])
        << "Buffer not correctly initialized";
    EXPECT_DOUBLE_EQ(0, buffer.current()[1])
        << "Buffer not correctly initialized";
    
    buffer.add(addend_p);
    
    EXPECT_DOUBLE_EQ(1.2, buffer.current()[0])
        << "Adding to buffer incorrect.";
    EXPECT_DOUBLE_EQ(2.4, buffer.current()[1])
        << "Adding to buffer incorrect.";
    
    buffer.add(addend_p);
    
    EXPECT_DOUBLE_EQ(2.4, buffer.current()[0])
        << "Adding to buffer incorrect.";
    EXPECT_DOUBLE_EQ(4.8, buffer.current()[1])
        << "Adding to buffer incorrect.";
        
    buffer.expand();
    
    EXPECT_DOUBLE_EQ(0, buffer[1][0])
        << "Buffer not correctly initialized";
    EXPECT_DOUBLE_EQ(0, buffer[1][1])
        << "Buffer not correctly initialized";
        
    EXPECT_EQ(2, buffer.size())
        << "Buffersize incorrect.";
        
    buffer.add(addend_p);
    
    EXPECT_DOUBLE_EQ(1.2, buffer[1][0])
        << "Adding to buffer incorrect.";
    EXPECT_DOUBLE_EQ(2.4, buffer[1][1])
        << "Adding to buffer incorrect.";

    buffer.make_zero();
    
    EXPECT_DOUBLE_EQ(0, buffer[0][0])
        << "Setting buffer to zero incorrect.";
    EXPECT_DOUBLE_EQ(0, buffer[0][1])
        << "Setting buffer to zero incorrect.";
    EXPECT_DOUBLE_EQ(0, buffer[1][0])
        << "Setting buffer to zero incorrect.";
    EXPECT_DOUBLE_EQ(0, buffer[1][1])
        << "Setting buffer to zero incorrect.";
}


TEST(TestModules, LinearLayer) {
    LinearLayer* layer_p = new LinearLayer(2);

    double* input_p = new double[2];
    input_p[0] = 2.;
    input_p[1] = 3.;

    ASSERT_DOUBLE_EQ(0, layer_p->input().current()[0])
        << "LinearLayer::add_to_input not working.";
    ASSERT_DOUBLE_EQ(0, layer_p->input().current()[1])
        << "LinearLayer::add_to_input not working.";

    layer_p->add_to_input(input_p);
    
    ASSERT_DOUBLE_EQ(2, layer_p->input().current()[0])
        << "LinearLayer::add_to_input not working.";
    ASSERT_DOUBLE_EQ(3, layer_p->input().current()[1])
        << "LinearLayer::add_to_input not working.";
    
    layer_p->forward();
    
    ASSERT_DOUBLE_EQ(2, layer_p->output().current()[0])
        << "Forward pass incorrect.";
        
    ASSERT_DOUBLE_EQ(3, layer_p->output().current()[1])
        << "Forward pass incorrect.";

    double* outerror_p = new double[2];
    outerror_p[0] = 1;
    outerror_p[1] = -.3;
    layer_p->add_to_outerror(outerror_p);
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(1, layer_p->inerror().current()[0])
        << "Backward pass incorrect.";
    
    EXPECT_DOUBLE_EQ(-0.3, layer_p->inerror().current()[1])
        << "Backward pass incorrect.";
}


TEST(TestModules, SigmoidLayer) {
    SigmoidLayer* layer_p = new SigmoidLayer(5);
    
    double* input_p = new double[5];
    input_p[0] = -1;
    input_p[1] = -0.5;
    input_p[2] = 0;
    input_p[3] = 0.5;
    input_p[4] = 1;
    
    layer_p->add_to_input(input_p);
    
    ASSERT_DOUBLE_EQ(-1, layer_p->input().current()[0])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(-0.5, layer_p->input().current()[1])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(0, layer_p->input().current()[2])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(0.5, layer_p->input().current()[3])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(1, layer_p->input().current()[4])
        << "add_to_input not working.";
    
    layer_p->forward();
    
    EXPECT_DOUBLE_EQ(0.2689414213699951, layer_p->output().current()[0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.37754066879814541, layer_p->output().current()[1])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.5, layer_p->output().current()[2])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.62245933120185459, layer_p->output().current()[3])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.7310585786300049, layer_p->output().current()[4])
        << "Forward pass incorrect.";
    
    double* outerror_p = new double[5];
    outerror_p[0] = 2;
    outerror_p[1] = 4;
    outerror_p[2] = 6;
    outerror_p[3] = 8;
    outerror_p[4] = 10;
    
    layer_p->add_to_outerror(outerror_p);
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(0.3932238664829637, layer_p->inerror().current()[0])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.94001484880637798, layer_p->inerror().current()[1])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(1.5, layer_p->inerror().current()[2])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(1.880029697612756, layer_p->inerror().current()[3])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(1.9661193324148185, layer_p->inerror().current()[4])
        << "Backward pass incorrect.";
}


TEST(TestModules, TanhLayer) {
    TanhLayer* layer_p = new TanhLayer(5);
    
    double* input_p = new double[5];
    input_p[0] = -1;
    input_p[1] = -0.5;
    input_p[2] = 0;
    input_p[3] = 0.5;
    input_p[4] = 1;
    
    layer_p->add_to_input(input_p);
    
    EXPECT_DOUBLE_EQ(-1, layer_p->input().current()[0])
        << "add_to_input not working.";
    EXPECT_DOUBLE_EQ(-0.5, layer_p->input().current()[1])
        << "add_to_input not working.";
    EXPECT_DOUBLE_EQ(0, layer_p->input().current()[2])
        << "add_to_input not working.";
    EXPECT_DOUBLE_EQ(0.5, layer_p->input().current()[3])
        << "add_to_input not working.";
    EXPECT_DOUBLE_EQ(1, layer_p->input().current()[4])
        << "add_to_input not working.";
    
    layer_p->forward();
    
    EXPECT_DOUBLE_EQ(-0.76159415595576485, layer_p->output().current()[0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(-0.46211715726000974, layer_p->output().current()[1])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0, layer_p->output().current()[2])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.46211715726000974, layer_p->output().current()[3])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.76159415595576485, layer_p->output().current()[4])
        << "Forward pass incorrect.";
    
    double* outerror_p = new double[5];
    outerror_p[0] = 2;
    outerror_p[1] = 4;
    outerror_p[2] = 6;
    outerror_p[3] = 8;
    outerror_p[4] = 10;
    
    layer_p->add_to_outerror(outerror_p);
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(0.83994868322805227, layer_p->inerror().current()[0])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(3.1457909318637096, layer_p->inerror().current()[1])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(6, layer_p->inerror().current()[2])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(6.2915818637274192, layer_p->inerror().current()[3])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(4.1997434161402616, layer_p->inerror().current()[4])
        << "Backward pass incorrect.";
}


TEST(TestModules, SoftmaxLayer) {
    SoftmaxLayer* layer_p = new SoftmaxLayer(2);

    double* input_p = new double[2];
    input_p[0] = 2.;
    input_p[1] = 4.;

    layer_p->add_to_input(input_p);
    
    ASSERT_DOUBLE_EQ(2, layer_p->input().current()[0])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(4, layer_p->input().current()[1])
        << "add_to_input not working.";
    
    layer_p->forward();
    
    ASSERT_DOUBLE_EQ(0.11920292202211756, layer_p->output().current()[0])
        << "Forward pass incorrect.";
        
    ASSERT_DOUBLE_EQ(0.88079707797788243, layer_p->output().current()[1])
        << "Forward pass incorrect.";

    double* outerror_p = new double[2];
    outerror_p[0] = 2;
    outerror_p[1] = 4;
    layer_p->add_to_outerror(outerror_p);
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(2, layer_p->inerror().current()[0])
        << "Backward pass incorrect.";
    
    EXPECT_DOUBLE_EQ(4, layer_p->inerror().current()[1])
        << "Backward pass incorrect.";
}


TEST(TestModules, PartialSoftmaxLayer) {
    PartialSoftmaxLayer* layer_p = new PartialSoftmaxLayer(4, 2);
    
    double* input_p = new double[5];
    input_p[0] = 2;
    input_p[1] = 4;
    input_p[2] = 4;
    input_p[3] = 8;
    
    layer_p->add_to_input(input_p);
    
    ASSERT_DOUBLE_EQ(2, layer_p->input().current()[0])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(4, layer_p->input().current()[1])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(4, layer_p->input().current()[2])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(8, layer_p->input().current()[3])
        << "add_to_input not working.";
    
    layer_p->forward();
    
    EXPECT_DOUBLE_EQ(0.11920292202211756, layer_p->output().current()[0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.88079707797788243, layer_p->output().current()[1])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.098446325560013689, layer_p->output().current()[2])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.90155367443998624, layer_p->output().current()[3])
        << "Forward pass incorrect.";
    
    double* outerror_p = new double[5];
    outerror_p[0] = 2;
    outerror_p[1] = 4;
    outerror_p[2] = 1;
    outerror_p[3] = 3;
    
    layer_p->add_to_outerror(outerror_p);
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(2, layer_p->inerror().current()[0])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(4, layer_p->inerror().current()[1])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(1, layer_p->inerror().current()[2])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(3, layer_p->inerror().current()[3])
        << "Backward pass incorrect.";
}


TEST(TestModules, MdlstmLayer) {
    MdlstmLayer* layer_p = new MdlstmLayer(2, 1);
    
    double* input_p = new double[10];
    input_p[0] = -2;
    input_p[1] = 1;
    input_p[2] = 2;
    input_p[3] = 3;
    input_p[4] = 4;
    input_p[5] = 5;
    input_p[6] = 6;
    input_p[7] = 7;
    input_p[8] = 8;
    input_p[9] = 9;
    
    layer_p->add_to_input(input_p);
    layer_p->forward();
    
    EXPECT_DOUBLE_EQ(0.99752618538000837, layer_p->output().current()[0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.99908893224240791, layer_p->output().current()[1])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(7.1654995964142723, layer_p->output().current()[2])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(9.3041593430291716, layer_p->output().current()[3])
        << "Forward pass incorrect.";
    
    double* outerror_p = new double[5];
    outerror_p[0] = -1;
    outerror_p[1] = 3;
    outerror_p[2] = 1;
    outerror_p[3] = 2;
    
    layer_p->add_to_outerror(outerror_p);
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(0.10492291615431382, layer_p->inerror().current()[0])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.39318818296939362, layer_p->inerror().current()[1])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.83994668169309272, layer_p->inerror().current()[2])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.81317991556297775, layer_p->inerror().current()[3])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.0001598448588049732, layer_p->inerror().current()[4])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.000265495970626106, layer_p->inerror().current()[5])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(-0.0024665063453200445, layer_p->inerror().current()[6])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.0027306634950956059, layer_p->inerror().current()[7])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.8807949791042492, layer_p->inerror().current()[8])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(1.9051483483108722, layer_p->inerror().current()[9])
        << "Backward pass incorrect.";
}


TEST(TestConnections, IdentityConnection) {
    LinearLayer* inlayer_p = new LinearLayer(2);
    LinearLayer* outlayer_p = new LinearLayer(2);
    
    inlayer_p->input().current()[0] = 2.;
    inlayer_p->input().current()[1] = 3.;
    
    IdentityConnection* con_p = new IdentityConnection(inlayer_p, outlayer_p);
    
    inlayer_p->forward();
    con_p->forward();
    outlayer_p->forward();
    
    EXPECT_DOUBLE_EQ(2., outlayer_p->output().current()[0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(3., outlayer_p->output().current()[1])
        << "Forward pass incorrect.";
    
    outlayer_p->outerror().current()[0] = 0.5;
    outlayer_p->outerror().current()[1] = 1.2;
    outlayer_p->backward();
    
    con_p->backward();
    inlayer_p->backward();
    
    EXPECT_DOUBLE_EQ(0.5, inlayer_p->outerror().current()[0])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(1.2, inlayer_p->outerror().current()[1])
            << "Backward pass incorrect.";
}


TEST(TestConnections, IdentityConnectionSliced) {
    LinearLayer* inlayer_p = new LinearLayer(2);
    LinearLayer* outlayer_p = new LinearLayer(2);
    
    inlayer_p->input().current()[0] = 2.;
    inlayer_p->input().current()[1] = 3.;
    
    IdentityConnection* con_p = \
        new IdentityConnection(inlayer_p, outlayer_p, 0, 1, 1, 2);
    
    ASSERT_EQ(0, con_p->get_incomingstart())
        << "_incomingstart not initialized properly.";
    ASSERT_EQ(1, con_p->get_incomingstop())
        << "_incomingstop not initialized properly.";
    ASSERT_EQ(1, con_p->get_outgoingstart())
        << "_outgoingstart not initialized properly.";
    ASSERT_EQ(2, con_p->get_outgoingstop())
        << "_ougoingstop not initialized properly.";
    
    inlayer_p->forward();
    con_p->forward();
    outlayer_p->forward();
    
    ASSERT_DOUBLE_EQ(0., outlayer_p->output().current()[0])
        << "Forward pass incorrect.";
    ASSERT_DOUBLE_EQ(2., outlayer_p->output().current()[1])
            << "Forward pass incorrect.";
    
    outlayer_p->outerror().current()[0] = 0.5;
    outlayer_p->outerror().current()[1] = 1.2;
    outlayer_p->backward();
    
    con_p->backward();
    inlayer_p->backward();
    
    EXPECT_DOUBLE_EQ(1.2, inlayer_p->outerror().current()[0])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0, inlayer_p->outerror().current()[1])
            << "Backward pass incorrect.";
}


TEST(TestConnections, FullConnection) {
    LinearLayer* inlayer_p = new LinearLayer(2);
    LinearLayer* outlayer_p = new LinearLayer(3);
    
    inlayer_p->input().current()[0] = 2.;
    inlayer_p->input().current()[1] = 3.;
    
    FullConnection* con_p = new FullConnection(inlayer_p, outlayer_p);
    con_p->get_parameters()[0] = 0; 
    con_p->get_parameters()[1] = 1; 
    con_p->get_parameters()[2] = 2; 
    con_p->get_parameters()[3] = 3; 
    con_p->get_parameters()[4] = 4; 
    con_p->get_parameters()[5] = 5;
    
    inlayer_p->forward();
    con_p->forward();
    
    EXPECT_DOUBLE_EQ(3, outlayer_p->input().current()[0])
        << "Forward pass not working.";
    EXPECT_DOUBLE_EQ(13, outlayer_p->input().current()[1])
        << "Forward pass not working.";
    EXPECT_DOUBLE_EQ(23, outlayer_p->input().current()[2])
        << "Forward pass not working.";
    
    outlayer_p->forward();
    outlayer_p->outerror().current()[0] = 0.5;
    outlayer_p->outerror().current()[1] = 1.2;
    outlayer_p->outerror().current()[2] = 3.4;
    outlayer_p->backward();
    con_p->backward();
    
    EXPECT_DOUBLE_EQ(1, con_p->get_derivatives()[0])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(1.5, con_p->get_derivatives()[1])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(2.4, con_p->get_derivatives()[2])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(3.6, con_p->get_derivatives()[3])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(6.8, con_p->get_derivatives()[4])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(10.2, con_p->get_derivatives()[5])
        << "Backward pass not working.";
}


TEST(TestConnections, FullConnectionSliced) {
    LinearLayer* inlayer_p = new LinearLayer(2);
    LinearLayer* outlayer_p = new LinearLayer(3);
    
    inlayer_p->input().current()[0] = 2.;
    inlayer_p->input().current()[1] = 3.;
    inlayer_p->output().current()[0] = 0.;
    inlayer_p->output().current()[1] = 0.;
    
    FullConnection* con_p = new FullConnection(inlayer_p, outlayer_p, 0, 1, 0, 4);
    
    ASSERT_EQ(0, con_p->get_incomingstart())
        << "_incomingstart not initialized properly.";
    ASSERT_EQ(1, con_p->get_incomingstop())
        << "_incomingstop not initialized properly.";
    ASSERT_EQ(0, con_p->get_outgoingstart())
        << "_outgoingstart not initialized properly.";
    ASSERT_EQ(4, con_p->get_outgoingstop())
        << "_ougoingstop not initialized properly.";
    
    con_p->get_parameters()[0] = -1;
    con_p->get_parameters()[1] = 1; 
    con_p->get_parameters()[2] = 2; 
    
    inlayer_p->forward();
    con_p->forward();
    
    ASSERT_DOUBLE_EQ(-2, outlayer_p->input().current()[0])
        << "Forward pass not working.";
    ASSERT_DOUBLE_EQ(2, outlayer_p->input().current()[1])
        << "Forward pass not working.";
    ASSERT_DOUBLE_EQ(4, outlayer_p->input().current()[2])
        << "Forward pass not working.";
    
    outlayer_p->forward();
    outlayer_p->outerror().current()[0] = 0.5;
    outlayer_p->outerror().current()[1] = 1.2;
    outlayer_p->outerror().current()[2] = 3.4;
    outlayer_p->backward();
    con_p->backward();
    
    EXPECT_DOUBLE_EQ(1, con_p->get_derivatives()[0])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(2.4, con_p->get_derivatives()[1])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(6.8, con_p->get_derivatives()[2])
        << "Backward pass not working.";
}


}  // namespace


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}