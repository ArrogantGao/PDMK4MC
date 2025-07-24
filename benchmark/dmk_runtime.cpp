#include <hpdmk.h>
#include <random>
#include <omp.h>
#include <mpi.h>

#include <tree.hpp>
#include <sctl.hpp>
#include <ewald.hpp>
#include <utils.hpp>

#include <chrono>
#include <fstream>


void dmk_runtime(int n_src, int n_src_per_leaf, double eps, double L) {
    HPDMKParams params;
    params.n_per_leaf = n_src_per_leaf;
    params.eps = eps;
    params.L = L;

    double r_src[n_src * 3];
    double charge[n_src];

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, params.L);

    for (int i = 0; i < n_src; i++) {
        r_src[i * 3] = distribution(generator);
        r_src[i * 3 + 1] = distribution(generator);
        r_src[i * 3 + 2] = distribution(generator);
        charge[i] = (-1) * (i % 2);
    }

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    sctl::Vector<double> r_src_vec(n_src * 3, const_cast<double *>(r_src), false);
    sctl::Vector<double> charge_vec(n_src, const_cast<double *>(charge), false);

    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src_vec, charge_vec);

    auto start = std::chrono::high_resolution_clock::now();
    tree.init_planewave_coeffs();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_planewave = end - start;

    start = std::chrono::high_resolution_clock::now();
    double E_window = tree.window_energy();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_window = end - start;

    start = std::chrono::high_resolution_clock::now();
    double E_difference = tree.difference_energy();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_difference = end - start;

    start = std::chrono::high_resolution_clock::now();
    double E_residual = tree.residual_energy();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_residual = end - start;

    double time_total = time_planewave.count() + time_window.count() + time_difference.count() + time_residual.count();

    std::ofstream outfile("data/dmk_runtime.csv", std::ios::app);
    outfile << n_src << "," << n_src_per_leaf << "," << eps << "," << L << "," << time_planewave.count() << "," << time_window.count() << "," << time_difference.count() << "," << time_residual.count() << "," << time_total << std::endl;
    outfile.close();
}

int main() {
    omp_set_num_threads(16);

    MPI_Init(nullptr, nullptr);

    double rho_0 = 1.0;

    std::ofstream outfile("data/dmk_runtime.csv");
    outfile << "n_src,n_src_per_leaf,eps,L,time_planewave,time_window,time_difference,time_residual,time_total" << std::endl;
    outfile.close();

    for (int scale = 1; scale <= 6; scale ++) {
        int n_src = 1000 * std::pow(2, scale);
        int n_src_per_leaf = 200;
        double eps = 1e-4;
        double L = std::pow(n_src / rho_0, 1.0 / 3.0);

        std::cout << "n_src: " << n_src << ", n_src_per_leaf: " << n_src_per_leaf << ", eps: " << eps << ", L: " << L << ", density: " << n_src / (L * L * L) << std::endl;

        dmk_runtime(n_src, n_src_per_leaf, eps, L);
    }

    MPI_Finalize();

    return 0;
}