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

    double r_src[n_src * 3];
    double charge[n_src];

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, params.L);

    for (int i = 0; i < n_src; i++) {
        r_src[i * 3] = distribution(generator);
        r_src[i * 3 + 1] = distribution(generator);
        r_src[i * 3 + 2] = distribution(generator);
        charge[i] = std::pow(-1, i) * 1.0;
    }

    double q[n_src];
    double r[n_src][3];
    for (int i = 0; i < n_src; i++) {
        q[i] = charge[i];
        r[i][0] = r_src[i * 3 + 0];
        r[i][1] = r_src[i * 3 + 1];
        r[i][2] = r_src[i * 3 + 2];
    }

    omp_set_num_threads(32);

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    sctl::Vector<double> r_src_vec(n_src * 3, const_cast<double *>(r_src), false);
    sctl::Vector<double> charge_vec(n_src, const_cast<double *>(charge), false);

    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src_vec, charge_vec);
    tree.init_planewave_coeffs();

    int depth = tree.level_indices.Dim() + 1;

    hpdmk::Ewald ewald_ref(L, 3.0, 3.0 * 2.5 / 3.68403, 1.0, q, r, n_src);
    hpdmk::Ewald ewald(L, 2.0, 2.0 * 2.5 / 3.68403, 1.0, q, r, n_src);

    double absolute_error_dmk = 0.0;
    double absolute_error_ewald = 0.0;
    double relative_error_dmk = 0.0;
    double relative_error_ewald = 0.0;

    for (int i = 0; i < n_samples; i++) {

        double trg_x = distribution(generator);
        double trg_y = distribution(generator);
        double trg_z = distribution(generator);

        tree.init_planewave_coeffs_target(trg_x, trg_y, trg_z);
        double potential_dmk = tree.potential_target(trg_x, trg_y, trg_z);

        ewald.collect_target_neighbors(trg_x, trg_y, trg_z);
        double potential_ewald = ewald.compute_potential(trg_x, trg_y, trg_z);

        ewald_ref.collect_target_neighbors(trg_x, trg_y, trg_z);
        double potential_ewald_ref = ewald_ref.compute_potential(trg_x, trg_y, trg_z);

        absolute_error_dmk += std::abs(potential_dmk - potential_ewald_ref);
        absolute_error_ewald += std::abs(potential_ewald - potential_ewald_ref);
        relative_error_dmk += std::abs(potential_dmk - potential_ewald_ref) / std::abs(potential_ewald_ref);
        relative_error_ewald += std::abs(potential_ewald - potential_ewald_ref) / std::abs(potential_ewald_ref);
    }

    std::ofstream outfile("data/mc_accuracy.csv", std::ios::app);
    outfile << n_src << "," << n_src_per_leaf << "," << eps << "," << L << "," << depth << "," << absolute_error_dmk / n_samples << "," << absolute_error_ewald / n_samples << "," << relative_error_dmk / n_samples << "," << relative_error_ewald / n_samples << std::endl;
    outfile.close();
}

int main() {
    MPI_Init(nullptr, nullptr);

    double rho_0 = 200.0;

    std::ofstream outfile("data/mc_accuracy.csv");
    outfile << "n_src,n_src_per_leaf,eps,L,depth,abserr_dmk,abserr_ewald,relerr_dmk,relerr_ewald" << std::endl;
    outfile.close();

    for (int scale = 0; scale <= 3; scale ++) {
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