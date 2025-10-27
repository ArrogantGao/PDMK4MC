#include <gtest/gtest.h>
#include <hpdmk.h>
#include <pswf.hpp>
#include <utils.hpp>

#include <cmath>
#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>

using namespace hpdmk;

TEST(PSWFTest, BasicAssertions) {
    for (int n_digits = 3; n_digits <= 12; n_digits += 3) {
        double tol = std::pow(10, -n_digits);
        double c, lambda, C0;
        int n_diff;
        std::vector<double> coefs;
        get_prolate_params(n_digits, c, lambda, C0, n_diff, coefs);

        auto real_poly = PolyFun<double>(coefs);

        auto f = [](double c0, double c, double x) {
            double val = prolate0_int_eval(c, x) / c0;
            val = 1 - val;
            return val;
        };

        for (double x = 0.01; x < 1.0; x += 0.01) {
            double val = f(C0, c, x);
            double val_ref = real_poly.eval(x);
            EXPECT_NEAR(val, val_ref, tol);
        }
    }
}