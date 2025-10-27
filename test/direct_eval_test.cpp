#include <gtest/gtest.h>
#include <hpdmk.h>
#include <direct_eval.hpp>

#include <cmath>
#include <iostream>
#include <vector>
#include <omp.h>
#include <random>
#include <sctl.hpp>
#include <kernels.hpp>
#include <pswf.hpp>

using namespace hpdmk;

template <class Real>
Real residual_direct_sum(const Real* r_src, const Real* q_src, const int n_trg, const Real* r_trg, const Real* q_trg, Real cutoff, int n_digits) {

    double c;
    if (n_digits <= 3) {
        c = 7.2462000846862793;
    } else if (n_digits <= 6) {
        c = 13.739999771118164;
    } else if (n_digits <= 9) {
        c = 20.736000061035156;
    } else if (n_digits <= 12) {
        c = 27.870000839233398;
    } else {
        throw std::runtime_error("n_digits is not supported");
    }

    double lambda = prolate0_lambda(c);
    double C0 = prolate0_int_eval(c, 1.0);

    Real u = 0;
    for (int i = 0; i < n_trg; i++) {
        Real d2 = (r_trg[i * 3] - r_src[0]) * (r_trg[i * 3] - r_src[0]) + (r_trg[i * 3 + 1] - r_src[1]) * (r_trg[i * 3 + 1] - r_src[1]) + (r_trg[i * 3 + 2] - r_src[2]) * (r_trg[i * 3 + 2] - r_src[2]);
        if (d2 < cutoff * cutoff) {
            u += hpdmk::residual_kernel<Real>(std::sqrt(d2), C0, c, cutoff) * q_trg[i] * q_src[0];
        }
    }
    return u;
}

TEST(DirectEvalTest, BasicAssertions) {

    int n_trgs[] = {10, 99, 100};
    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for (int n_digits_ = 3; n_digits_ <= 12; n_digits_ += 3) {
        double r_src[] = {0.1, 0.2, 0.3};
        double q_src = 0.5;

        for (int n_trg : n_trgs) {
            sctl::Vector<double> r_trg(n_trg * 3);
            sctl::Vector<double> q_trg(n_trg);
            for (int i = 0; i < n_trg; i++) {
                r_trg[i * 3] = distribution(generator);
                r_trg[i * 3 + 1] = distribution(generator);
                r_trg[i * 3 + 2] = distribution(generator);
                q_trg[i] = distribution(generator);
            }

            double cutoff = 0.8;
            int n_digits = n_digits_;

            double res_direct = residual_direct_sum<double>(&r_src[0], &q_src, n_trg, &r_trg[0], &q_trg[0], cutoff, n_digits);

            double res = direct_eval<double>(&r_src[0], &q_src, n_trg, &r_trg[0], &q_trg[0], cutoff, n_digits);

            // std::cout << "n_digits: " << n_digits << ", res: " << res << ", res_direct: " << res_direct << ", error: " << std::abs(res - res_direct) << std::endl;
            ASSERT_LT(std::abs(res - res_direct), std::pow(10.0, -n_digits));
        }
    }
}