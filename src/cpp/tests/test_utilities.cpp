#include <gtest/gtest.h>
#include <iostream>
#include "../arac.h"

namespace AracTesting {


using namespace arac::utilities;


TEST(TestBlockPermutation, Test44_22) {
    std::vector<int> perm, shape, blockshape;
    
    shape.push_back(4);
    shape.push_back(4);
    blockshape.push_back(2);
    blockshape.push_back(2);

    block_permutation(perm, shape, blockshape);
    ASSERT_EQ(16, perm.size());

    
    int solution_p[16] = {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15};
    
    for (int i = 0; i < 16; i++)
    {
        EXPECT_EQ(solution_p[i], perm[i])
            << "Permutation not correct at " << i << ".";
    }
}


TEST(TestBlockPermutation, Test34_12) {
    std::vector<int> perm, shape, blockshape;
    
    shape.push_back(3);
    shape.push_back(4);
    blockshape.push_back(1);
    blockshape.push_back(2);

    block_permutation(perm, shape, blockshape);
    ASSERT_EQ(12, perm.size());

    int solution_p[12] = {0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11};
    
    for (int i = 0; i < 12; i++)
    {
        EXPECT_EQ(solution_p[i], perm[i])
            << "Permutation not correct at " << i << ".";
    }
}


TEST(TestBlockPermutation, Test442_221) 
{
    std::vector<int> perm, shape, blockshape;
    
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(2);
    blockshape.push_back(2);
    blockshape.push_back(2);
    blockshape.push_back(1);

    block_permutation(perm, shape, blockshape);
    ASSERT_EQ(32, perm.size());

    int solution_p[32] = {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15, 16, 17, 20, 21, 18, 19, 22, 23, 24, 25, 28, 29, 26, 27, 30, 31};
    
    for (int i = 0; i < 32; i++)
    {
        EXPECT_EQ(solution_p[i], perm[i])
            << "Permutation not correct at " << i << ".";
    }
}


TEST(TestBlockPermutation, Test442_222) {
    std::vector<int> perm, shape, blockshape;
    
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(2);
    blockshape.push_back(2);
    blockshape.push_back(2);
    blockshape.push_back(2);

    block_permutation(perm, shape, blockshape);
    ASSERT_EQ(32, perm.size());

    int solution_p[32] = {0, 1, 4, 5, 16, 17, 20, 21, 2, 3, 6, 7, 18, 19, 22, 23, 8, 9, 12, 13, 24, 25, 28, 29, 10, 11, 14, 15, 26, 27, 30, 31};
    
    for (int i = 0; i < 32; i++)
    {
        EXPECT_EQ(solution_p[i], perm[i])
            << "Permutation not correct at " << i << ".";
    }
}



        
}  // namespace