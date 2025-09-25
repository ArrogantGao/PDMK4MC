#include <gtest/gtest.h>
#include <hpdmk.h>
#include <random>
#include <omp.h>
#include <mpi.h>

#include <complex>
#include <cmath>

#include <tree.hpp>
#include <sctl.hpp>
#include <ewald.hpp>
#include <utils.hpp>

using namespace hpdmk;



void compare_direct() {
    HPDMKParams params;
    params.n_per_leaf = 5;
    params.eps = 1e-4;
    params.L = 100.0;

    omp_set_num_threads(16);

    int n_src = 100;
    sctl::Vector<double> r_src_vec(n_src * 3);
    sctl::Vector<double> charge_vec(n_src);

    std::vector<double> r_src(n_src * 3);
    std::vector<double> charge(n_src);

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, params.L);

    for (int i = 0; i < n_src; i++) {
        r_src[i * 3] = distribution(generator);
        r_src_vec[i * 3] = r_src[i * 3];
        r_src[i * 3 + 1] = distribution(generator);
        r_src_vec[i * 3 + 1] = r_src[i * 3 + 1];
        r_src[i * 3 + 2] = distribution(generator);
        r_src_vec[i * 3 + 2] = r_src[i * 3 + 2];
        charge_vec[i] = std::pow(-1, i) * 1.0;
        charge[i] = charge_vec[i];
    }

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src_vec, charge_vec);
    tree.init_planewave_coeffs();

    double E_window = tree.window_energy();
    double E_difference = tree.difference_energy();
    double E_residual = tree.residual_energy();

    double E_window_direct = tree.window_energy_direct();
    double E_difference_direct = tree.difference_energy_direct();
    double E_residual_direct = tree.residual_energy_direct();

    ASSERT_NEAR(E_window, E_window_direct, 1e-3);
    ASSERT_NEAR(E_difference, E_difference_direct, 1e-3);
    ASSERT_NEAR(E_residual, E_residual_direct, 1e-3);

    int nrounds = 10;
    for (int i = 0; i < nrounds; i++) {
        double trg_x = distribution(generator);
        double trg_y = distribution(generator);
        double trg_z = distribution(generator);

        tree.init_planewave_coeffs_target(tree.target_planewave_coeffs, tree.path_to_target, trg_x, trg_y, trg_z);
        double p_window = tree.potential_window(tree.target_planewave_coeffs, trg_x, trg_y, trg_z);
        double p_difference = tree.potential_difference(tree.target_planewave_coeffs, tree.path_to_target, trg_x, trg_y, trg_z);
        double p_residual = tree.potential_residual(tree.path_to_target, trg_x, trg_y, trg_z);

        double p_window_direct = tree.potential_window_direct(trg_x, trg_y, trg_z);
        double p_difference_direct = tree.potential_difference_direct(trg_x, trg_y, trg_z);
        double p_residual_direct = tree.potential_residual_direct(trg_x, trg_y, trg_z);

        ASSERT_NEAR(p_window, p_window_direct, 1e-3);
        ASSERT_NEAR(p_difference, p_difference_direct, 1e-3);
        ASSERT_NEAR(p_residual, p_residual_direct, 1e-3);
    }
}

void compare_ewald() {
    HPDMKParams params;
    params.n_per_leaf = 20;
    params.eps = 1e-4;
    params.L = 100.0;

    omp_set_num_threads(16);

    int n_src = 1000;
    std::vector<double> r_src(n_src * 3);
    std::vector<double> charge(n_src);
    
    sctl::Vector<double> r_src_vec(n_src * 3);
    sctl::Vector<double> charge_vec(n_src);

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, params.L);

    for (int i = 0; i < n_src; i++) {
        r_src[i * 3] = distribution(generator);
        r_src_vec[i * 3] = r_src[i * 3];
        r_src[i * 3 + 1] = distribution(generator);
        r_src_vec[i * 3 + 1] = r_src[i * 3 + 1];
        r_src[i * 3 + 2] = distribution(generator);
        r_src_vec[i * 3 + 2] = r_src[i * 3 + 2];
        charge[i] = std::pow(-1, i) * 1.0;
        charge_vec[i] = charge[i];
    }

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src_vec, charge_vec);
    tree.init_planewave_coeffs();

    double E_hpdmk = tree.energy();

    hpdmk::Ewald ewald(100.0, 3.0, 0.5, 1.0, charge, r_src, n_src);
    double E_ewald = ewald.compute_energy();

    ASSERT_NEAR(E_hpdmk, E_ewald, 1e-3);

    int nrounds = 10;
    for (int i = 0; i < nrounds; i++) {
        double trg_x = distribution(generator);
        double trg_y = distribution(generator);
        double trg_z = distribution(generator);

        tree.init_planewave_coeffs_target(tree.target_planewave_coeffs, tree.path_to_target, trg_x, trg_y, trg_z);
        double p_hpdmk = tree.potential_target(trg_x, trg_y, trg_z);

        double p_ewald = ewald.compute_potential(trg_x, trg_y, trg_z);

        ASSERT_NEAR(p_hpdmk, p_ewald, 1e-3);
    }
}

TEST(HPDMKTest, BasicAssertions) {
    MPI_Init(nullptr, nullptr);
    compare_direct();
    compare_ewald();
    MPI_Finalize();
}