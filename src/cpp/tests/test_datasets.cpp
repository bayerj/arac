#include <gtest/gtest.h>
#include <iostream>
#include "../arac.h"

namespace AracTesting {


using namespace arac::datasets;


TEST(TestSequence, TestConstruction) {
    double* contents_p = new double[8];
    double* targets_p = new double[4];
    
    contents_p[0] = 1;
    contents_p[1] = 5;
    contents_p[2] = 10;
    contents_p[3] = 14;
    contents_p[4] = 5;
    contents_p[5] = 26;
    contents_p[6] = 7;
    contents_p[7] = 98;

    targets_p[0] = 0;
    targets_p[1] = 1;
    targets_p[2] = 0;
    targets_p[3] = 1;
    
    Sequence seq(4, 2, 1, contents_p, targets_p);

    EXPECT_DOUBLE_EQ(1, seq.contents(0)[0])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(5, seq.contents(0)[1])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(10, seq.contents(1)[0])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(14, seq.contents(1)[1])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(5, seq.contents(2)[0])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(26, seq.contents(2)[1])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(7, seq.contents(3)[0])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(98, seq.contents(3)[1])
        << "Wrong sequence entry.";

    EXPECT_DOUBLE_EQ(0, seq.targets(0)[0])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(1, seq.targets(1)[0])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(0, seq.targets(2)[0])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(1, seq.targets(3)[0])
        << "Wrong sequence entry.";
}


TEST(TestSemiSequence, TestConstruction) {
    double* contents_p = new double[8];
    double* targets_p = new double[2];
    
    contents_p[0] = 1;
    contents_p[1] = 5;
    contents_p[2] = 10;
    contents_p[3] = 14;
    contents_p[4] = 5;
    contents_p[5] = 26;
    contents_p[6] = 7;
    contents_p[7] = 98;

    targets_p[0] = 1.3;
    targets_p[1] = 2.5;
    
    SemiSequence seq(4, 2, 2, contents_p, targets_p);

    EXPECT_DOUBLE_EQ(1, seq.contents(0)[0])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(5, seq.contents(0)[1])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(10, seq.contents(1)[0])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(14, seq.contents(1)[1])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(5, seq.contents(2)[0])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(26, seq.contents(2)[1])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(7, seq.contents(3)[0])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(98, seq.contents(3)[1])
        << "Wrong sequence entry.";

    EXPECT_DOUBLE_EQ(1.3, seq.targets(0)[0])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(2.5, seq.targets(0)[1])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(1.3, seq.targets(1)[0])
        << "Wrong sequence entry.";
    EXPECT_DOUBLE_EQ(2.5, seq.targets(1)[1])
        << "Wrong sequence entry.";
}


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


TEST(TestSemiSequentialDataset, TestConstruction)
{
    
}


TEST(TestSemiSequentialDataset, TestConstruction)
{
    
}

        
}  // namespace