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

#include <iomanip>

#include <chrono>
#include <cstdlib> // for std::atoi, std::atof

void test_ewald_potential() {
    int n_src = 1000;
    double L = 10.0;

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, L);

    double r_src[n_src * 3];
    double charge[n_src];

    double trg_x = distribution(generator);
    double trg_y = distribution(generator);
    double trg_z = distribution(generator);

    for (int i = 0; i < n_src; i++) {
        r_src[i * 3] = distribution(generator);
        r_src[i * 3 + 1] = distribution(generator);
        r_src[i * 3 + 2] = distribution(generator);
        charge[i] = std::pow(-1, i) * 1.0;
    }

    double total_charge = 0.0;
    for (int i = 0; i < n_src; i++) {
        total_charge += charge[i];
    }
    std::cout << "total charge: " << total_charge << std::endl;


    double alphas[] = {1.0, 2.0, 3.0, 4.0};

    for (double alpha : alphas) {
        double q[n_src];
        double r[n_src][3];
        for (int i = 0; i < n_src; i++) {
            q[i] = charge[i];
            r[i][0] = r_src[i * 3 + 0];
            r[i][1] = r_src[i * 3 + 1];
            r[i][2] = r_src[i * 3 + 2];
        }

        hpdmk::Ewald ewald(L, 3.0, alpha, 1.0, q, r, n_src);

        ewald.init_planewave_coeffs();
        double energy_ewald = ewald.compute_energy();

        ewald.collect_target_neighbors(trg_x, trg_y, trg_z);
        double potential_ewald = ewald.compute_potential(trg_x, trg_y, trg_z);

        std::cout << "alpha: " << alpha << ", energy ewald: " << energy_ewald << ", potential ewald: " << potential_ewald << std::endl;
    }
}

void test_tree(int n_src_per_leaf, double eps) {
    HPDMKParams params;
    params.n_per_leaf = n_src_per_leaf;
    params.eps = eps;
    params.L = 10.0;

    omp_set_num_threads(16);
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    std::cout << "eps: " << params.eps << ", n_src_per_leaf: " << n_src_per_leaf << std::endl;

    int n_src = 2000;
    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, params.L);
    std::uniform_real_distribution<double> distribution_charge(-1, 1);

    double r_src[n_src * 3];
    double charge[n_src];

    for (int i = 0; i < n_src; i++) {
        r_src[i * 3] = distribution(generator);
        r_src[i * 3 + 1] = distribution(generator);
        r_src[i * 3 + 2] = distribution(generator);
        charge[i] = std::pow(-1, i) * 1.0;
    }

    double total_charge = 0.0;
    for (int i = 0; i < n_src; i++) {
        total_charge += charge[i];
    }
    std::cout << "total charge: " << total_charge << std::endl;

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    sctl::Vector<double> r_src_vec(n_src * 3, const_cast<double *>(r_src), false);
    sctl::Vector<double> charge_vec(n_src, const_cast<double *>(charge), false);

    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src_vec, charge_vec);

    tree.init_planewave_coeffs();

    for (int i = 0; i < 10; ++i) {
        double trg_x = distribution(generator);
        double trg_y = distribution(generator);
        double trg_z = distribution(generator);

        std::cout << "target point is " << trg_x << ", " << trg_y << ", " << trg_z << std::endl;

        // std::cout << "target point is " << trg_x << ", " << trg_y << ", " << trg_z << std::endl;
        // tree.locate_target(trg_x, trg_y, trg_z);
        // std::cout << "path to target: ";
        // for (int j = 0; j < tree.path_to_target.Dim(); ++j) {
        //     std::cout << tree.path_to_target[j] << std::endl;
        //     std::cout << "depth: " << int(tree.GetNodeMID()[tree.path_to_target[j]].Depth()) << std::endl;
        //     std::cout << "center: " << tree.centers[tree.path_to_target[j] * 3] << ", " << tree.centers[tree.path_to_target[j] * 3 + 1] << ", " << tree.centers[tree.path_to_target[j] * 3 + 2] << std::endl;
        //     assert(tree.is_in_node(trg_x, trg_y, trg_z, tree.path_to_target[j]));
        // }

        auto start = std::chrono::high_resolution_clock::now();
        tree.init_planewave_coeffs_target(trg_x, trg_y, trg_z);
        double potential_window = tree.potential_target_window(trg_x, trg_y, trg_z);
        double potential_difference = tree.potential_target_difference(trg_x, trg_y, trg_z);
        double potential_residual = tree.potential_target_residual(trg_x, trg_y, trg_z);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_planewave = end - start;

        std::cout << "time taken to calculate potential: " << time_planewave.count() << " seconds" << std::endl;

        double total_potential = potential_window + potential_difference + potential_residual;

        double potential_residual_direct = tree.potential_target_residual_direct(trg_x, trg_y, trg_z);
        double potential_difference_direct = tree.potential_target_difference_direct(trg_x, trg_y, trg_z);

        double total_potential_direct = potential_window + potential_difference_direct + potential_residual_direct;

        std::cout << std::setprecision(16) << "potential window: " << potential_window << std::endl;
        std::cout << std::setprecision(16) << "potential difference: " << potential_difference << ", direct: " << potential_difference_direct << std::endl;
        std::cout << std::setprecision(16) << "potential residual: " << potential_residual << ", direct: " << potential_residual_direct << std::endl;

        // Ewald potential
        double q[n_src];
        double r[n_src][3];
        for (int i = 0; i < n_src; i++) {
            q[i] = charge[i];
            r[i][0] = r_src[i * 3 + 0];
            r[i][1] = r_src[i * 3 + 1];
            r[i][2] = r_src[i * 3 + 2];
        }

        hpdmk::Ewald ewald(params.L, 3.0, 1.0, 1.0, q, r, n_src);
        ewald.collect_target_neighbors(trg_x, trg_y, trg_z);
        double potential_ewald = ewald.compute_potential(trg_x, trg_y, trg_z);

        std::cout << std::setprecision(16) << "total potential: " << total_potential << ", direct: " << total_potential_direct << ", ewald: " << potential_ewald << std::endl;
        std::cout << std::setprecision(16) << "absolute error: " << std::abs(total_potential - potential_ewald) << std::endl;
        std::cout << std::setprecision(16) << "relative error: " << std::abs(total_potential - potential_ewald) / std::abs(total_potential) << std::endl;
    }

}

int main(int argc, char** argv) {
    int n_src_per_leaf = 200;
    double eps = 1e-4;

    if (argc > 1) n_src_per_leaf = std::atoi(argv[1]);
    if (argc > 2) eps = std::atof(argv[2]);

    MPI_Init(nullptr, nullptr);
    
    test_tree(n_src_per_leaf, eps);
    // test_ewald_potential();

    MPI_Finalize();

    return 0;
}