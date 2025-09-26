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


void mc_accuracy(int n_src, int n_src_per_leaf, double eps, double L, int n_samples) {
    HPDMKParams params;
    params.n_per_leaf = n_src_per_leaf;
    params.eps = eps;
    params.L = L;

    HPDMKParams params_ref;
    params_ref.n_per_leaf = n_src_per_leaf;
    params_ref.eps = 1e-6;
    params_ref.L = L;

    std::vector<double> r_src(n_src * 3);
    std::vector<double> charge(n_src);

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, params.L);

    for (int i = 0; i < n_src; i++) {
        r_src[i * 3] = distribution(generator);
        r_src[i * 3 + 1] = distribution(generator);
        r_src[i * 3 + 2] = distribution(generator);
        charge[i] = std::pow(-1, i) * 1.0;
    }
    
    omp_set_num_threads(16);

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    sctl::Vector<double> r_src_vec(n_src * 3, r_src.data(), false);
    sctl::Vector<double> charge_vec(n_src, charge.data(), false);

    std::cout << "tree init" << std::endl;

    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src_vec, charge_vec);
    tree.init_planewave_coeffs();

    std::cout << "tree_ref init" << std::endl;

    hpdmk::HPDMKPtTree<double> tree_ref(sctl_comm, params_ref, r_src_vec, charge_vec);
    tree_ref.init_planewave_coeffs();

    int depth = tree.level_indices.Dim() + 1;

    std::cout << "ewald init" << std::endl;

    double s = 2.5;
    hpdmk::Ewald ewald(L, s, 1.5 * s / std::sqrt(L), 1.0, charge, r_src, n_src);
    ewald.init_planewave_coeffs();

    std::cout << "all init done" << std::endl;

    double absolute_error_dmk = 0.0;
    double absolute_error_ewald = 0.0;
    double relative_error_dmk = 0.0;
    double relative_error_ewald = 0.0;

    for (int i = 0; i < n_samples; i++) {

        double trg_x = distribution(generator);
        double trg_y = distribution(generator);
        double trg_z = distribution(generator);

        tree.init_planewave_coeffs(tree.target_planewave_coeffs, tree.path_to_target, trg_x, trg_y, trg_z, 1.0);
        double potential_dmk = tree.potential_target(trg_x, trg_y, trg_z);

        std::cout << "potential_dmk: " << potential_dmk << std::endl;

        double potential_ewald = ewald.compute_potential(trg_x, trg_y, trg_z);

        std::cout << "potential_ewald: " << potential_ewald << std::endl;

        tree_ref.init_planewave_coeffs(tree_ref.target_planewave_coeffs, tree_ref.path_to_target, trg_x, trg_y, trg_z, 1.0);
        double potential_dmk_ref = tree_ref.potential_target(trg_x, trg_y, trg_z);

        std::cout << "potential_dmk_ref: " << potential_dmk_ref << std::endl;

        absolute_error_dmk += std::abs(potential_dmk - potential_dmk_ref);
        absolute_error_ewald += std::abs(potential_ewald - potential_dmk_ref);
        relative_error_dmk += std::abs(potential_dmk - potential_dmk_ref) / std::abs(potential_dmk_ref);
        relative_error_ewald += std::abs(potential_ewald - potential_dmk_ref) / std::abs(potential_dmk_ref);
    }

    std::ofstream outfile("data/potential_accuracy.csv", std::ios::app);
    outfile << n_src << "," << n_src_per_leaf << "," << eps << "," << L << "," << depth << "," << absolute_error_dmk / n_samples << "," << absolute_error_ewald / n_samples << "," << relative_error_dmk / n_samples << "," << relative_error_ewald / n_samples << std::endl;
    outfile.close();
}

int main() {
    MPI_Init(nullptr, nullptr);

    double rho_0 = 200.0;

    std::ofstream outfile("data/potential_accuracy.csv");
    outfile << "n_src,n_src_per_leaf,eps,L,depth,abserr_dmk,abserr_ewald,relerr_dmk,relerr_ewald" << std::endl;
    outfile.close();

    for (int scale = 0; scale <= 5; scale ++) {
        int n_src = 10000 * std::pow(2, scale);
        int n_src_per_leaf = 500;
        double eps = 1e-3;
        double L = std::pow(n_src / rho_0, 1.0 / 3.0);

        int n_samples = 20;

        std::cout << "n_src: " << n_src << ", n_src_per_leaf: " << n_src_per_leaf << ", eps: " << eps << ", L: " << L << ", density: " << n_src / (L * L * L) << std::endl;

        mc_accuracy(n_src, n_src_per_leaf, eps, L, n_samples);
    }

    MPI_Finalize();

    return 0;
}