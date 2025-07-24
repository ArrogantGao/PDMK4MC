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

#include <ctime>

// TEST(HPDMKTest, BasicAssertions) {
//     hpdmk::HPDMKParams<double> params;
//     params.n_per_leaf = 20;
//     params.eps = 1e-4;
//     params.L = 10.0;

//     sctl::Vector<double> r_src(1000);
//     sctl::Vector<double> charge(1000);

//     hpdmk::HPDMKPtTree<double> tree(sctl::Comm::Self(), params, r_src, charge);
// }

void test_tree() {
    HPDMKParams params;
    params.n_per_leaf = 10;
    params.eps = 1e-12;
    params.L = 100.0;

    omp_set_num_threads(16);
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;

    int n_src = 1000;
    double r_src[n_src * 3];
    double charge[n_src];

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, params.L);
    std::uniform_real_distribution<double> distribution_charge(-1, 1);

    for (int i = 0; i < n_src; i++) {
        r_src[i * 3] = distribution(generator);
        r_src[i * 3 + 1] = distribution(generator);
        r_src[i * 3 + 2] = distribution(generator);
        // charge[i] = distribution_charge(generator);
        charge[i] = (-1) * (i % 2);
    }

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    sctl::Vector<double> r_src_vec(n_src * 3, const_cast<double *>(r_src), false);
    sctl::Vector<double> charge_vec(n_src, const_cast<double *>(charge), false);

    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src_vec, charge_vec);

    sctl::Long i_node = tree.level_indices[1][0];
    auto &node_mid = tree.GetNodeMID();
    auto &node_attr = tree.GetNodeAttr();
    auto &node_list = tree.GetNodeLists();

    std::cout << "node " << i_node << " is at depth " << node_mid[i_node].Depth() << std::endl;

    for (int i = 0; i < 27; ++i) {
        if (node_list[i_node].nbr[i] != -1) {
            std::cout << "neighbor " << node_list[i_node].nbr[i] << " is at depth " << int(node_mid[node_list[i_node].nbr[i]].Depth()) << std::endl;
        }
    }

    sctl::Long leaf_node_i = tree.level_indices[tree.max_depth - 2][0];
    std::cout << "leaf node " << leaf_node_i << " is at depth " << int(node_mid[leaf_node_i].Depth()) << std::endl;
    for (int i = 0; i < 27; i++) {
        std::cout << "the " << i + 1 << "th neighbor of leaf node " << leaf_node_i << " is " << node_list[leaf_node_i].nbr[i] << std::endl;
    }

    std::cout << "center of node " << leaf_node_i << " is " << tree.centers[leaf_node_i * 3] << ", " << tree.centers[leaf_node_i * 3 + 1] << ", " << tree.centers[leaf_node_i * 3 + 2] << std::endl;

    for (int i = 0; i < tree.neighbors[leaf_node_i].coarsegrain.Dim(); i++) {
        std::cout << "coarsegrain neighbor " << tree.neighbors[leaf_node_i].coarsegrain[i] << " is at depth " << int(node_mid[tree.neighbors[leaf_node_i].coarsegrain[i]].Depth()) << " and center " << tree.centers[tree.neighbors[leaf_node_i].coarsegrain[i] * 3] << ", " << tree.centers[tree.neighbors[leaf_node_i].coarsegrain[i] * 3 + 1] << ", " << tree.centers[tree.neighbors[leaf_node_i].coarsegrain[i] * 3 + 2] << std::endl;
        auto shift = tree.node_shift(leaf_node_i, tree.neighbors[leaf_node_i].coarsegrain[i]);
        std::cout << "shift from leaf node " << leaf_node_i << " to coarsegrain neighbor " << tree.neighbors[leaf_node_i].coarsegrain[i] << " is " << shift[0] << ", " << shift[1] << ", " << shift[2] << std::endl;
    }
    for (int i = 0; i < tree.neighbors[leaf_node_i].colleague.Dim(); i++) {
        std::cout << "colleague neighbor " << tree.neighbors[leaf_node_i].colleague[i] << " is at depth " << int(node_mid[tree.neighbors[leaf_node_i].colleague[i]].Depth()) << " and center " << tree.centers[tree.neighbors[leaf_node_i].colleague[i] * 3] << ", " << tree.centers[tree.neighbors[leaf_node_i].colleague[i] * 3 + 1] << ", " << tree.centers[tree.neighbors[leaf_node_i].colleague[i] * 3 + 2] << std::endl;
        auto shift = tree.node_shift(leaf_node_i, tree.neighbors[leaf_node_i].colleague[i]);
        std::cout << "shift from leaf node " << leaf_node_i << " to colleague neighbor " << tree.neighbors[leaf_node_i].colleague[i] << " is " << shift[0] << ", " << shift[1] << ", " << shift[2] << std::endl;
    }
}

void compare_ewald() {
    HPDMKParams params;
    params.n_per_leaf = 20;
    params.eps = 1e-4;
    params.L = 10.0;

    omp_set_num_threads(16);
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    std::cout << "eps: " << params.eps << std::endl;

    int n_src = 1000;

    double r_src[n_src * 3];
    double charge[n_src];

    auto particles = hpdmk::read_particle_info("/home/xzgao/code/HybridPeriodicDMK/data/particle_info.txt");

    for (int i = 0; i < n_src; i++) {
        r_src[i * 3] = particles[i][0] * 10;
        r_src[i * 3 + 1] = particles[i][1] * 10;
        r_src[i * 3 + 2] = particles[i][2] * 10;
        charge[i] = particles[i][3];
    }

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    sctl::Vector<double> r_src_vec(n_src * 3, const_cast<double *>(r_src), false);
    sctl::Vector<double> charge_vec(n_src, const_cast<double *>(charge), false);

    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src_vec, charge_vec);

    tree.init_planewave_coeffs();
    
    // double E_total = tree.energy();

    double E_window = tree.window_energy();

    std::cout << "window energy is " << std::setprecision(16) << E_window << std::endl;

    double E_residual = tree.residual_energy();
    double E_residual_direct = tree.residual_energy_direct();
    std::cout << "residual energy is " << std::setprecision(16) << E_residual << " and direct sum is " << std::setprecision(16) << E_residual_direct << std::endl;

    double E_difference = tree.difference_energy();
    double E_difference_direct = tree.difference_energy_direct();
    std::cout << "difference energy is " << std::setprecision(16) << E_difference << " and direct sum is " << std::setprecision(16) << E_difference_direct << std::endl;

    double E_total = E_window + E_difference + E_residual;
    std::cout << "dmk total energy is " << std::setprecision(16) << E_total << std::endl;

    double q[n_src];
    double r[n_src][3];
    for (int i = 0; i < n_src; i++) {
        q[i] = charge[i];
        r[i][0] = r_src[i * 3];
        r[i][1] = r_src[i * 3 + 1];
        r[i][2] = r_src[i * 3 + 2];
    }

    hpdmk::Ewald ewald(tree.L, 5.0, 1.0, 1.0, q, r, n_src);
    double E_ewald = ewald.compute_energy(q, r) * 4 * M_PI;

    std::cout << "ewald energy is " << std::setprecision(16) << E_ewald << std::endl;
    std::cout << "absolute error is " << std::setprecision(16) << std::abs(E_ewald - E_total) << std::endl;
    std::cout << "relative error is " << std::setprecision(16) << std::abs(E_ewald - E_total) / std::abs(E_total) << std::endl;
}


int main() {

    MPI_Init(nullptr, nullptr);
    
    // hpdmk_tree_create(MPI_COMM_WORLD, params, n_src, r_src, charge);
    // test_tree();

    compare_ewald();

    MPI_Finalize();

    return 0;
}