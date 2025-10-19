#include <benchmark/benchmark.h>
#include <hpdmk.h>
#include <nudft.hpp>
#include <complex>
#include <vector>
#include <random>
#include <cmath>
#include <omp.h>

using namespace std;
using namespace hpdmk;

template <typename T>
static void BM_nudft3d1_halfplane(benchmark::State& state) {
    const int N1 = 13, N2 = 13, N3 = 13;
    const int M = 128;
    const int N3_half = (N3 + 1) / 2 + (N3 % 2 == 0);
    const int Fsize = N1 * N2 * N3_half;

    std::vector<T> x(M), y(M), z(M);
    std::vector<std::complex<T>> c(M), f(Fsize, std::complex<T>(0,0));

    std::mt19937 rng(123);
    std::uniform_real_distribution<T> dist(-M_PI, M_PI);

    for (int i = 0; i < M; ++i) {
        x[i] = dist(rng);
        y[i] = dist(rng);
        z[i] = dist(rng);
        c[i] = {dist(rng), dist(rng)};
    }

    for (auto _ : state) {
        std::fill(f.begin(), f.end(), std::complex<T>(0,0));
        benchmark::DoNotOptimize(f);
        nudft3d1_halfplane<T>(M, x.data(), y.data(), z.data(), c.data(),
                              1, N1, N2, N3, f.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(int64_t(state.iterations()) * M);
}

BENCHMARK_TEMPLATE(BM_nudft3d1_halfplane, float)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_nudft3d1_halfplane, double)->Unit(benchmark::kMillisecond);

template <typename T>
static void BM_nudft3d1_single_halfplane(benchmark::State& state) {
    const int N1 = 13, N2 = 13, N3 = 13;
    const int N3_half = (N3 % 2 == 0) ? (N3 / 2 + 1) : ((N3 + 1) / 2);
    const int Fsize = N1 * N2 * N3_half;

    std::vector<std::complex<T>> f(Fsize);
    std::vector<std::complex<T>> x_cache(N1), y_cache(N2), z_cache(N3_half);

    std::mt19937 rng(42);
    std::uniform_real_distribution<T> dist(-M_PI, M_PI);

    const T x = dist(rng);
    const T y = dist(rng);
    const T z = dist(rng);
    const std::complex<T> c{dist(rng), dist(rng)};
    const int iflag = 1;

    for (auto _ : state) {
        std::fill(f.begin(), f.end(), std::complex<T>(0, 0));
        benchmark::DoNotOptimize(f);
        nudft3d1_single_halfplane<T>(x, y, z, c, iflag, N1, N2, N3,
                                     x_cache.data(), y_cache.data(), z_cache.data(), f.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(int64_t(state.iterations()));
}

BENCHMARK_TEMPLATE(BM_nudft3d1_single_halfplane, float)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_nudft3d1_single_halfplane, double)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();