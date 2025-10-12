#include <gtest/gtest.h>
#include <hpdmk.h>
#include <nudft.hpp>

#include <cmath>
#include <complex>
#include <vector>
#include <random>
#include <omp.h>

using namespace std;

TEST(NudftDoubleTest, BasicAssertions) {
    const int M = 100;

    const int N_half = 10;
    const int N = 21;

    std::vector<double> x(M), y(M), z(M);
    std::vector<complex<double>> c(M);

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, 1.0);

    for (int i = 0; i < M; i++) {
        x[i] = distribution(generator);
        y[i] = distribution(generator);
        z[i] = distribution(generator);
        c[i] = complex<double>(distribution(generator), distribution(generator));
    }

    vector<complex<double>> f(N * N * N), f_half(N * N * N), f_nufft(N * N * N);

    hpdmk::nudft3d1(M, x.data(), y.data(), z.data(), c.data(), 1, N, N, N, f.data());
    hpdmk::nudft3d1_halfplane(M, x.data(), y.data(), z.data(), c.data(), 1, N, N, N, f_half.data());
    hpdmk::nufft3d1(M, x.data(), y.data(), z.data(), c.data(), 1, 1e-6, N, N, N, f_nufft.data());

    for (int i = 0; i < N * N * N; i++) {
        EXPECT_NEAR(real(f[i]), real(f_nufft[i]), 1e-4);
        EXPECT_NEAR(imag(f[i]), imag(f_nufft[i]), 1e-4);
    }

    for (int i = 0; i < N * N * (N_half + 1); i++) {
        EXPECT_NEAR(real(f[i]), real(f_half[i]), 1e-8);
        EXPECT_NEAR(imag(f[i]), imag(f_half[i]), 1e-8);
    }
}

TEST(NudftFloatTest, BasicAssertions) {
    const int M = 100;

    const int N_half = 10;
    const int N = 21;

    const float eps = 1e-6;

    std::vector<float> x(M), y(M), z(M);
    std::vector<complex<float>> c(M);

    std::mt19937 generator;
    std::uniform_real_distribution<float> distribution(0, 1.0);

    for (int i = 0; i < M; i++) {
        x[i] = distribution(generator);
        y[i] = distribution(generator);
        z[i] = distribution(generator);
        c[i] = complex<float>(distribution(generator), distribution(generator));
    }

    vector<complex<float>> f(N * N * N), f_half(N * N * N), f_nufft(N * N * N);

    hpdmk::nudft3d1(M, x.data(), y.data(), z.data(), c.data(), 1, N, N, N, f.data());
    hpdmk::nudft3d1_halfplane(M, x.data(), y.data(), z.data(), c.data(), 1, N, N, N, f_half.data());
    hpdmk::nufft3d1(M, x.data(), y.data(), z.data(), c.data(), 1, eps, N, N, N, f_nufft.data());

    for (int i = 0; i < N * N * N; i++) {
        EXPECT_NEAR(real(f[i]), real(f_nufft[i]), 1e-4);
        EXPECT_NEAR(imag(f[i]), imag(f_nufft[i]), 1e-4);
    }

    for (int i = 0; i < N * N * (N_half + 1); i++) {
        EXPECT_NEAR(real(f[i]), real(f_half[i]), 1e-5);
        EXPECT_NEAR(imag(f[i]), imag(f_half[i]), 1e-5);
    }
}   
