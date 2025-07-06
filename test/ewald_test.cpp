#include <gtest/gtest.h>
#include <hpdmk.h>
#include <ewald.hpp>

#include <cmath>
#include <iostream>

TEST(EwaldTest, BasicAssertions) {
    double q[] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0};

    double r[][3] = {
        {0.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 2.0},
        {6.0, 0.0, 0.0},
        {6.0, 2.0, 0.0},
        {6.0, 0.0, 2.0},
    };

    hpdmk::Ewald ewald(10.0, 4.0, 1.0, 1.0, q, r, 6);
    EXPECT_DOUBLE_EQ(ewald.L, 10.0);
    EXPECT_DOUBLE_EQ(ewald.s, 4.0);
    EXPECT_DOUBLE_EQ(ewald.alpha, 1.0);
    EXPECT_DOUBLE_EQ(ewald.eps, 1.0);
    EXPECT_DOUBLE_EQ(ewald.r_c, 4.0 / 1.0);
    EXPECT_DOUBLE_EQ(ewald.k_c, 8.0);

    std::cout << "dim: " << ewald.k.size() << std::endl;

    std::cout << "neighbors: " << ewald.neighbors.length << std::endl;
    for (int i = 0; i < ewald.neighbors.length; i++) {
        std::cout << "pair: " << ewald.neighbors.pairs[i][0] << " " << ewald.neighbors.pairs[i][1] << " " << ewald.neighbors.distances[i] << std::endl;
        std::cout << "shift: " << ewald.neighbors.shifts[i][0] << " " << ewald.neighbors.shifts[i][1] << " " << ewald.neighbors.shifts[i][2] << std::endl;
    }

    double E = ewald.compute_energy(q, r);
    std::cout << "E: " << E << std::endl;
}