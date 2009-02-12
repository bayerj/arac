#include <gtest/gtest.h>
#include <iostream>
#include "../arac.h"

namespace AracTesting {


using namespace arac::datasets;


TEST(TestSequence, TestConstruction)
{
    double* data_p = new double[10];
    data_p[0] = 1;
    data_p[1] = 2;
    data_p[2] = 4;
    data_p[3] = 8;
    data_p[4] = 16;
    data_p[5] = 32;
    data_p[6] = 64;
    data_p[7] = 128;
    data_p[8] = 256;
    data_p[9] = 512;
    
    Sequence seq(5, 2, data_p);
    
    EXPECT_EQ(5, seq.length());
    EXPECT_EQ(2, seq.itemsize());
    
    EXPECT_DOUBLE_EQ(1, seq[0][0]) << "Sequence data not correct.";
    EXPECT_DOUBLE_EQ(2, seq[0][1]) << "Sequence data not correct.";
    EXPECT_DOUBLE_EQ(4, seq[1][0]) << "Sequence data not correct.";
    EXPECT_DOUBLE_EQ(8, seq[1][1]) << "Sequence data not correct.";
    EXPECT_DOUBLE_EQ(16, seq[2][0]) << "Sequence data not correct.";
    EXPECT_DOUBLE_EQ(32, seq[2][1]) << "Sequence data not correct.";
    EXPECT_DOUBLE_EQ(64, seq[3][0]) << "Sequence data not correct.";
    EXPECT_DOUBLE_EQ(128, seq[3][1]) << "Sequence data not correct.";
    EXPECT_DOUBLE_EQ(256, seq[4][0]) << "Sequence data not correct.";
    EXPECT_DOUBLE_EQ(512, seq[4][1]) << "Sequence data not correct.";
}


TEST(TestDataset_sequence_array, TestConstruction)
{
    double* first_data_p = new double[10];
    first_data_p[0] = 1;
    first_data_p[1] = 2;
    first_data_p[2] = 4;
    first_data_p[3] = 8;
    first_data_p[4] = 16;
    first_data_p[5] = 32;
    first_data_p[6] = 64;
    first_data_p[7] = 128;
    first_data_p[8] = 256;
    first_data_p[9] = 512;
    
    Sequence first(5, 2, first_data_p);
    
    double* second_data_p = new double[8];
    second_data_p[0] = -1;
    second_data_p[1] = -2;
    second_data_p[2] = -4;
    second_data_p[3] = -8;
    second_data_p[4] = -16;
    second_data_p[5] = -32;
    second_data_p[6] = -64;
    second_data_p[7] = -128;
    
    Sequence second(4, 2, second_data_p);
    
    double* first_target = new double[1];
    first_target[0] = 1;
    double* second_target = new double[1];
    second_target[0] = -1;
    
    SupervisedSemiSequentialDataset ds(2, 1);
    ds.append(first, first_target);
    ds.append(second, second_target);
    
    EXPECT_EQ(2, ds.size())
        << "Wrong size of dataset.";
    
    EXPECT_EQ(1, ds[0].first[0][0]);
    EXPECT_EQ(2, ds[0].first[0][1]);
    EXPECT_EQ(4, ds[0].first[1][0]);
    EXPECT_EQ(8, ds[0].first[1][1]);
    EXPECT_EQ(16, ds[0].first[2][0]);
    EXPECT_EQ(32, ds[0].first[2][1]);
    EXPECT_EQ(64, ds[0].first[3][0]);
    EXPECT_EQ(128, ds[0].first[3][1]);
    EXPECT_EQ(256, ds[0].first[4][0]);
    EXPECT_EQ(512, ds[0].first[4][1]);
    
    EXPECT_EQ(-1, ds[1].first[0][0]);
    EXPECT_EQ(-2, ds[1].first[0][1]);
    EXPECT_EQ(-4, ds[1].first[1][0]);
    EXPECT_EQ(-8, ds[1].first[1][1]);
    EXPECT_EQ(-16, ds[1].first[2][0]);
    EXPECT_EQ(-32, ds[1].first[2][1]);
    EXPECT_EQ(-64, ds[1].first[3][0]);
    EXPECT_EQ(-128, ds[1].first[3][1]);

    EXPECT_EQ(1, ds[0].second[0]);
    EXPECT_EQ(-1, ds[1].second[0]);
}


TEST(TestDataset_array_array, TestConstruction)
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
    
    SupervisedSimpleDataset ds(2, 1);
    ds.append(data_p, data_p + 2);
    ds.append(data_p + 3, data_p + 5);
    ds.append(data_p + 6, data_p + 8);
    ds.append(data_p + 9, data_p + 11);
    
    EXPECT_EQ(4, ds.size())
        << "Wrong size of dataset.";
    
    EXPECT_EQ(0, ds[0].first[0])
        << "Wrong size of dataset.";
    EXPECT_EQ(0, ds[0].first[1])
        << "Wrong item in dataset.";
    EXPECT_EQ(0, ds[0].second[0])
        << "Wrong size of dataset.";

    EXPECT_EQ(0, ds[1].first[0])
        << "Wrong item in dataset.";
    EXPECT_EQ(1, ds[1].first[1])
        << "Wrong item in dataset.";
    EXPECT_EQ(1, ds[1].second[0])
        << "Wrong item in dataset.";

    EXPECT_EQ(1, ds[2].first[0])
        << "Wrong item in dataset.";
    EXPECT_EQ(0, ds[2].first[1])
        << "Wrong item in dataset.";
    EXPECT_EQ(1, ds[2].second[0])
        << "Wrong item in dataset.";

    EXPECT_EQ(1, ds[3].first[0])
        << "Wrong item in dataset.";
    EXPECT_EQ(1, ds[3].first[1])
        << "Wrong item in dataset.";
    EXPECT_EQ(0, ds[3].second[0])
        << "Wrong item in dataset.";
}


TEST(TestDataSet_sequential_sequential, TestConstruction)
{
    
    
}


        
}  // namespace