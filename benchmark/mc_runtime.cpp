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


void mc_runtime(int n_src, int n_src_per_leaf, double eps, double L) {
    HPDMKParams params;
    params.n_per_leaf = n_src_per_leaf;
    params.eps = eps;
    params.L = L;

    sctl::Vector<double> r_src_vec(n_src * 3);
    sctl::Vector<double> charge_vec(n_src);

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, params.L);

    std::cout << "init r_src and charge" << std::endl;
    for (int i = 0; i < n_src; i++) {
        r_src_vec[i * 3] = distribution(generator);
        r_src_vec[i * 3 + 1] = distribution(generator);
        r_src_vec[i * 3 + 2] = distribution(generator);
        charge_vec[i] = std::pow(-1, i) * 1.0;
    }
    std::cout << "init r_src and charge done" << std::endl;

    omp_set_num_threads(32);

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    std::cout << "init tree" << std::endl;
    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src_vec, charge_vec);
    std::cout << "init tree done" << std::endl;

    std::cout << "init planewave coeffs" << std::endl;
    tree.init_planewave_coeffs();
    std::cout << "init planewave coeffs done" << std::endl;

    int depth = tree.level_indices.Dim() + 1;

    double tc = 0.0;
    double tw = 0.0;
    double td = 0.0;
    double tr = 0.0;

    // omp_set_num_threads(1);

    int rounds = 100;
    for (int i = 0; i < rounds; i++) {
        double trg_x = distribution(generator);
        double trg_y = distribution(generator);
        double trg_z = distribution(generator);

        auto start = std::chrono::high_resolution_clock::now();
        tree.init_planewave_coeffs_target(trg_x, trg_y, trg_z);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_planewave = end - start;
        tc += time_planewave.count();
        
        start = std::chrono::high_resolution_clock::now();
        double potential_window = tree.potential_target_window(trg_x, trg_y, trg_z);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_window = end - start;
        tw += time_window.count();

        start = std::chrono::high_resolution_clock::now();
        double potential_difference = tree.potential_target_difference(trg_x, trg_y, trg_z);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_difference = end - start;
        td += time_difference.count();

        start = std::chrono::high_resolution_clock::now();
        double potential_residual = tree.potential_target_residual(trg_x, trg_y, trg_z);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_residual = end - start;
        tr += time_residual.count();
    }


    double avg_tw = tw / rounds;
    double avg_td = td / rounds;
    double avg_tr = tr / rounds;
    double avg_tc = tc / rounds;
    double time_total = avg_tw + avg_td + avg_tr + avg_tc;

    std::ofstream outfile("data/mc_runtime.csv", std::ios::app);
    outfile << n_src << "," << n_src_per_leaf << "," << eps << "," << L << "," << depth << "," << avg_tc << "," << avg_tw << "," << avg_td << "," << avg_tr << "," << time_total << std::endl;
    outfile.close();
}

int main() {
    MPI_Init(nullptr, nullptr);

    double rho_0 = 200.0;

    std::ofstream outfile("data/mc_runtime.csv");
    outfile << "n_src,n_src_per_leaf,eps,L,depth,time_planewave,time_window,time_difference,time_residual,time_total" << std::endl;
    outfile.close();

    for (int scale = 0; scale <= 10; scale ++) {
        int n_src = 1000 * std::pow(2, scale);
        int n_src_per_leaf = 100;
        double eps = 1e-3;
        double L = std::pow(n_src / rho_0, 1.0 / 3.0);

        std::cout << "n_src: " << n_src << ", n_src_per_leaf: " << n_src_per_leaf << ", eps: " << eps << ", L: " << L << ", density: " << n_src / (L * L * L) << std::endl;

        mc_runtime(n_src, n_src_per_leaf, eps, L);
    }

    MPI_Finalize();

    return 0;
}