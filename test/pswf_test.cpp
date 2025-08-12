#include <gtest/gtest.h>
#include <hpdmk.h>
#include <pswf.hpp>

#include <cmath>
#include <iostream>
#include <vector>
#include <omp.h>

TEST(PSWFTest, BasicAssertions) {
    double tol = 1e-4;
    int order = 15;

    double c = hpdmk::prolc180(tol);
    double lambda = hpdmk::prolate0_lambda(c);
    EXPECT_NEAR(c, 1.202400000000000e+01, 1e-10);
    EXPECT_NEAR(lambda, 7.228787365921187e-01, 1e-10);

    double C0 = hpdmk::prolate0_int_eval(c, 1.0);

    auto energy_func = [c, C0](double x) {
        return 1 - hpdmk::prolate0_int_eval(c, x) / C0;
    };

    auto fourier_func = [c, lambda, C0](double x) {
        return 2 * M_PI * lambda * hpdmk::prolate0_eval(c, x) / C0;
    };

    auto real_poly = hpdmk::approximate_real_poly<double>(tol, order);
    auto fourier_poly = hpdmk::approximate_fourier_poly<double>(tol, order);

    EXPECT_EQ(real_poly.order, order);
    EXPECT_EQ(fourier_poly.order, order);

    for (int i = 0; i < 100; i++) {
        double x = i * 0.01 * 1.0;
        double energy_ref = energy_func(x);
        double fourier_ref = fourier_func(x);
        double energy_test = real_poly.eval(x);
        double fourier_test = fourier_poly.eval(x);
        EXPECT_NEAR(energy_ref, energy_test, 1e-4);
        EXPECT_NEAR(fourier_ref, fourier_test, 1e-4 * 4 * M_PI);
    }
}