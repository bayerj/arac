#include <gtest/gtest.h>
#include <iostream>
#include "../arac.h"

namespace AracTesting {


using namespace arac::datasets;


TEST(TestDataset, TestConstruction)
{
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
    
    Dataset ds(2, 1);
    ds.append(data_p);
    ds.append(data_p + 3);
    ds.append(data_p + 6);
    ds.append(data_p + 9);
    
    EXPECT_EQ(4, ds.size())
        << "Wrong size of dataset.";
    
    EXPECT_EQ(0, ds[0][0])
        << "Wrong size of dataset.";
    EXPECT_EQ(0, ds[0][1])
        << "Wrong item in dataset.";
    EXPECT_EQ(0, ds[0][2])
        << "Wrong size of dataset.";

    EXPECT_EQ(0, ds[1][0])
        << "Wrong item in dataset.";
    EXPECT_EQ(1, ds[1][1])
        << "Wrong item in dataset.";
    EXPECT_EQ(1, ds[1][2])
        << "Wrong item in dataset.";

    EXPECT_EQ(1, ds[2][0])
        << "Wrong item in dataset.";
    EXPECT_EQ(0, ds[2][1])
        << "Wrong item in dataset.";
    EXPECT_EQ(1, ds[2][2])
        << "Wrong item in dataset.";

    EXPECT_EQ(1, ds[3][0])
        << "Wrong item in dataset.";
    EXPECT_EQ(1, ds[3][1])
        << "Wrong item in dataset.";
    EXPECT_EQ(0, ds[3][2])
        << "Wrong item in dataset.";
}

        
}  // namespace