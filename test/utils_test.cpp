#include <gtest/gtest.h>
#include <hpdmk.h>
#include <ewald.hpp>
#include <utils.hpp>

#include <cmath>
#include <iostream>
#include <vector>
#include <omp.h>

TEST(UtilsTest, BasicAssertions) {
    double x = 1.0;
    double y = 2.0;
    double z = 3.0;

    double dist = hpdmk::dist2(x, y, z, 0.0, 0.0, 0.0);
    EXPECT_DOUBLE_EQ(dist, 14.0);

    for (int i = 0; i < 100; i++) {
        double xi = i * 0.01 * 3.0;
        float xif = i * 0.01f * 3.0f;
        double erfc_ref = std::erfc(xi);
        double erfc_test = hpdmk::my_erfc<double>(xi);
        float erfc_test_f = hpdmk::my_erfc<float>(xif);
        EXPECT_NEAR(erfc_ref, erfc_test, 1e-4);
        EXPECT_NEAR(erfc_ref, erfc_test_f, 1e-4);
    }
}