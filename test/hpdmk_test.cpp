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

void compare_energy_and_potential() {
    HPDMKParams params;
    params.n_per_leaf = 20;
    params.eps = 1e-6;
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

    hpdmk::Ewald ewald(100.0, 4.0, 0.5, 1.0, charge, r_src, n_src);
    double E_ewald = ewald.compute_energy();

    ASSERT_NEAR(E_hpdmk, E_ewald, 1e-6);

    int nrounds = 10;
    for (int i = 0; i < nrounds; i++) {
        double trg_x = distribution(generator);
        double trg_y = distribution(generator);
        double trg_z = distribution(generator);

        tree.init_planewave_coeffs_target(trg_x, trg_y, trg_z);
        double p_hpdmk = tree.potential_target(trg_x, trg_y, trg_z);

        double p_ewald = ewald.compute_potential(trg_x, trg_y, trg_z);

        ASSERT_NEAR(p_hpdmk, p_ewald, 1e-6);
    }
}

TEST(HPDMKTest, BasicAssertions) {
    MPI_Init(nullptr, nullptr);
    compare_energy_and_potential();
    MPI_Finalize();
}