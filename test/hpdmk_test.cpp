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
}


void compare_shift() {
    HPDMKParams params;
    params.n_per_leaf = 5;
    params.eps = 1e-6;
    params.L = 10.0;

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
        charge[i] = std::pow(-1, i) * (0.5);
        charge_vec[i] = charge[i];
    }

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src_vec, charge_vec);
    tree.init_planewave_coeffs();

    hpdmk::Ewald ewald(params.L, 3.0, 0.5, 1.0, charge, r_src, n_src);
    double E_ewald_origin = ewald.compute_energy();

    int nrounds = 200;
    int i0 = 1;
    for (int i = i0; i < i0 + nrounds; i++) {
        double dx = distribution(generator);
        double dy = distribution(generator);
        double dz = distribution(generator);

        double E_shift_dmk = tree.energy_shift(tree.indices_invmap[i], dx, dy, dz);

        std::vector<double> r_src_shifted(n_src * 3);
        // copy r_src to r_src_shifted
        std::copy(r_src.begin(), r_src.end(), r_src_shifted.begin());
        r_src_shifted[i * 3] = my_mod<double>(dx + r_src[i * 3], params.L);
        r_src_shifted[i * 3 + 1] = my_mod<double>(dy + r_src[i * 3 + 1], params.L);
        r_src_shifted[i * 3 + 2] = my_mod<double>(dz + r_src[i * 3 + 2], params.L);

        hpdmk::Ewald ewald_shifted(params.L, 3.0, 0.5, 1.0, charge, r_src_shifted, n_src);
        double E_ewald_shifted = ewald_shifted.compute_energy();


        #ifdef DEBUG
        std::cout << "i: " << i << std::endl;
        std::cout << "r_src_shifted: " << r_src_shifted[i * 3] << ", " << r_src_shifted[i * 3 + 1] << ", " << r_src_shifted[i * 3 + 2] << std::endl;
        std::cout << "r_src: " << r_src[i * 3] << ", " << r_src[i * 3 + 1] << ", " << r_src[i * 3 + 2] << std::endl;
        std::cout << "E_shift_dmk: " << E_shift_dmk << ", E_ewald_shift: " << E_ewald_shifted - E_ewald_origin << ", diff: " << E_shift_dmk - (E_ewald_shifted - E_ewald_origin) << std::endl;
        #endif

        ASSERT_NEAR(E_shift_dmk, E_ewald_shifted - E_ewald_origin, 1e-3);
    }
}


TEST(HPDMKTest, BasicAssertions) {
    MPI_Init(nullptr, nullptr);
    // compare_direct();
    // compare_ewald();
    compare_shift();
    MPI_Finalize();
}