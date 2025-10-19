#include <benchmark/benchmark.h>
#include <complex>
#include <vector>
#include <random>
#include <hpdmk.h>
#include <vecops.hpp>

using namespace hpdmk;

// fixed array size: 13x13x7, corresponding to 1e-3 accuracy
constexpr int NX = 13;
constexpr int NY = 13;
constexpr int NZ = 7;
constexpr int M  = NX * NY * NZ;  // 1183

template <typename Real>
void init_random(std::vector<std::complex<Real>>& A,
                 std::vector<std::complex<Real>>& B,
                 std::vector<std::complex<Real>>& C,
                 std::vector<Real>& R) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);
    for (int i = 0; i < M; ++i) {
        A[i] = {dist(rng), dist(rng)};
        B[i] = {dist(rng), dist(rng)};
        C[i] = {dist(rng), dist(rng)};
        R[i] = dist(rng);
    }
}

template <typename Real, bool ConjugateA, bool ConjugateB>
static void BM_vec_tridot(benchmark::State& state) {
    std::vector<std::complex<Real>> A(M), B(M);
    std::vector<Real> C(M);
    init_random(A, B, A, C);

    for (auto _ : state) {
        benchmark::DoNotOptimize(
            vec_tridot<Real, ConjugateA, ConjugateB>(M, A.data(), B.data(), C.data())
        );
    }
    state.SetItemsProcessed(state.iterations() * M);
}

BENCHMARK_TEMPLATE(BM_vec_tridot, float, false, false)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_vec_tridot, float, true, true)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_vec_tridot, float, true, false)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_vec_tridot, float, false, true)->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_vec_tridot, double, false, false)->Unit(benchmark::kMicrosecond);

template <typename Real, bool Add, bool ConjugateB>
static void BM_vec_addsub(benchmark::State& state) {
    std::vector<std::complex<Real>> A(M), B(M);
    std::vector<Real> dummy(M);
    init_random(A, B, A, dummy);

    for (auto _ : state) {
        vec_addsub<Real, Add, ConjugateB>(M, A.data(), B.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * M);
}

BENCHMARK_TEMPLATE(BM_vec_addsub, float, true, false)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_vec_addsub, float, true, true)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_vec_addsub, float, false, true)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_vec_addsub, float, false, false)->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_vec_addsub, double, true, true)->Unit(benchmark::kMicrosecond);

template <typename Real, bool Add, bool ConjugateB>
static void BM_vec_muladdsub(benchmark::State& state) {
    std::vector<std::complex<Real>> A(M), B(M), C(M);
    std::vector<Real> dummy(M);
    init_random(A, B, C, dummy);

    for (auto _ : state) {
        vec_muladdsub<Real, Add, ConjugateB>(M, A.data(), B.data(), C.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * M);
}

BENCHMARK_TEMPLATE(BM_vec_muladdsub, float, true, false)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_vec_muladdsub, float, true, true)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_vec_muladdsub, float, false, true)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_vec_muladdsub, float, false, false)->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_vec_muladdsub, double, true, true)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
