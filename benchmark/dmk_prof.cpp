#include <hpdmk.h>
#include <random>
#include <omp.h>
#include <mpi.h>

#include <tree.hpp>
#include <sctl.hpp>
#include <ewald.hpp>
#include <utils.hpp>

#include <gperftools/profiler.h>

void dmk_runtime(int n_src, int n_src_per_leaf, double eps, double L) {
    HPDMKParams params;
    params.n_per_leaf = n_src_per_leaf;
    params.eps = eps;
    params.L = L;

    sctl::Vector<double> r_src(n_src * 3);
    sctl::Vector<double> charge(n_src);

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, params.L);

    for (int i = 0; i < n_src; i++) {
        r_src[i * 3] = distribution(generator);
        r_src[i * 3 + 1] = distribution(generator);
        r_src[i * 3 + 2] = distribution(generator);
        charge[i] = (-1) * (i % 2);
    }

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    ProfilerStart("init.prof");
    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src, charge);
    ProfilerStop();

    ProfilerStart("planewave.prof");
    tree.init_planewave_coeffs();
    ProfilerStop();

    ProfilerStart("energy.prof");
    double E_window = tree.window_energy();
    double E_difference = tree.difference_energy();
    double E_residual = tree.residual_energy();
    ProfilerStop();
}

int main(int argc, char **argv) {
    omp_set_num_threads(1);

    MPI_Init(nullptr, nullptr);

    double rho_0 = 1.0;

    int n_src = 20000;
    int n_src_per_leaf = 200;
    double eps = 1e-4;
    double L = std::pow(n_src / rho_0, 1.0 / 3.0);

    dmk_runtime(n_src, n_src_per_leaf, eps, L);

    return 0;
}