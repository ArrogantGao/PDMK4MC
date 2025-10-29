#include <benchmark/benchmark.h>

#include <random>
#include <cmath>

#include <sctl.hpp>
#include <hpdmk.h>
#include <direct_eval.hpp>

template <typename Real, int digits, int n_trg>
static void BM_direct_eval(benchmark::State& state) {
    std::mt19937 generator;
    std::uniform_real_distribution<Real> distribution(-1.0, 1.0);
    std::uniform_real_distribution<Real> distribution_two(-2.0, 2.0);

    const Real cutoff = 1.0;

    sctl::Vector<Real> r_src(3);
    sctl::Vector<Real> q_src(1);

    for (int i = 0; i < 3; i++) {
        r_src[i] = distribution(generator);
    }
    q_src[0] = distribution(generator);
    
    sctl::Vector<Real> r_trg(n_trg * 3);
    sctl::Vector<Real> q_trg(n_trg);

    for (int i = 0; i < n_trg; i++) {
        r_trg[i * 3] = distribution_two(generator);
        r_trg[i * 3 + 1] = distribution_two(generator);
        r_trg[i * 3 + 2] = distribution_two(generator);
        q_trg[i] = distribution(generator);
    }

    for (auto _ : state) {
        Real u = direct_eval<Real>(&r_src[0], &q_src[0], n_trg, &r_trg[0], &q_trg[0], cutoff, digits);
        benchmark::DoNotOptimize(u);
    }
    state.SetItemsProcessed(state.iterations() * n_trg);
}

BENCHMARK_TEMPLATE(BM_direct_eval, float, 3, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_direct_eval, float, 6, 1024)->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_direct_eval, double, 3, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_direct_eval, double, 6, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_direct_eval, double, 9, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_direct_eval, double, 12, 1024)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();