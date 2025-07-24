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

#include <fstream>
#include <sstream>
#include <string>

#include <iomanip>

std::vector<std::vector<double>> read_particle_info(const std::string& filename) {
    std::vector<std::vector<double>> particles;
    std::ifstream infile(filename);
    std::string line;

    // 读取表头
    std::getline(infile, line);

    // 读取每一行数据
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double x, y, z, q;
        if (!(iss >> x >> y >> z >> q)) continue; // 跳过格式不对的行
        particles.push_back({x, y, z, q});
    }
    return particles;
}

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

    tree.init_planewave_coeffs();
    // check the coeffs of the root node
    auto &root_coeffs = tree.plane_wave_coeffs[tree.root()];

    auto center_x = tree.L / 2;
    auto center_y = tree.L / 2;
    auto center_z = tree.L / 2;

    int n_k = tree.n_k[0];
    double delta_k = 2 * M_PI / tree.L;

    // std::cout << "checking the coeffs of the root node" << std::endl;
    // for (int i = 0; i < 2 * n_k + 1; i++) {
    //     for (int j = 0; j < 2 * n_k + 1; j++) {
    //         for (int k = 0; k < 2 * n_k + 1; k++) {
    //             auto t = root_coeffs.value(i, j, k);
    //             std::complex<double> s = 0;
    //             for (int i_particle = 0; i_particle < n_src; i_particle++) {
    //                 double x = r_src[i_particle * 3] - center_x;
    //                 double y = r_src[i_particle * 3 + 1] - center_y;
    //                 double z = r_src[i_particle * 3 + 2] - center_z;
    //                 s += charge[i_particle] * std::exp( - std::complex<double>(0, 1) * ((i - n_k) * delta_k * x + (j - n_k) * delta_k * y + (k - n_k) * delta_k * z));
    //             }
    //             // std::cout << "coeff " << i << ", " << j << ", " << k << " direct sum is " << s << " and tree generated is " << t << " and error is " << std::abs(s - t) << std::endl;
    //             assert(std::abs(s - t) < 1e-10);
    //         }
    //     }
    // }
    // std::cout << "root node coeffs checked" << std::endl;

    double E_window = tree.window_energy();
    double E_window_direct = 0;
    double sigma = tree.sigmas[2];
    #pragma omp parallel for reduction(+:E_window_direct)
    for (int i = 0; i < 2 * n_k + 1; i++) {
        for (int j = 0; j < 2 * n_k + 1; j++) {
            for (int k = 0; k < 2 * n_k + 1; k++) {
                double k_x = (i - n_k) * delta_k;
                double k_y = (j - n_k) * delta_k;
                double k_z = (k - n_k) * delta_k;
                double k2 = k_x * k_x + k_y * k_y + k_z * k_z;

                if (k2 > 0) {
                    std::complex<double> s = 0;
                    for (int i_particle = 0; i_particle < n_src; i_particle++) {
                        double x = r_src[i_particle * 3];
                        double y = r_src[i_particle * 3 + 1];
                        double z = r_src[i_particle * 3 + 2];
                        s += charge[i_particle] * std::exp( - std::complex<double>(0, 1) * (k_x * x + k_y * y + k_z * z));
                    }
                    
                    double d = 4 * M_PI * std::exp(- k2 * sigma * sigma / 4) / k2;
                    E_window_direct += std::real(s * std::conj(s)) * d;
                }
            }
        }
    }

    E_window_direct /= (2 * std::pow(tree.L, 3));
    E_window_direct -= tree.Q / (std::sqrt(M_PI) * sigma);
    
    std::cout << "window energy direct sum is " << E_window_direct << " and tree generated is " << E_window << " and error is " << std::abs(E_window_direct - E_window) << std::endl;

    // std::cout << "interaction matrix for window is " << tree.interaction_matrices[2].tensor << std::endl;

    // auto &node_depth3 = tree.level_indices[3];
    // for (int i = 0; i < node_depth3.Dim(); i++) {
    //     auto &nbrs = node_list[node_depth3[i]].nbr;
    //     for (int j = 0; j < 27; j++) {
    //         if (nbrs[j] != -1) {
    //             auto center_xi = tree.centers[node_depth3[i] * 3];
    //             auto center_xj = tree.centers[nbrs[j] * 3];
    //             auto center_yi = tree.centers[node_depth3[i] * 3 + 1];
    //             auto center_yj = tree.centers[nbrs[j] * 3 + 1];
    //             auto center_zi = tree.centers[node_depth3[i] * 3 + 2];
    //             auto center_zj = tree.centers[nbrs[j] * 3 + 2];

    //             auto shift = tree.node_shift(node_depth3[i], nbrs[j]);
    //             std::cout << "shift from node " << node_depth3[i] << " to node " << nbrs[j] << " is " << shift[0] << ", " << shift[1] << ", " << shift[2] << std::endl;
    //             std::cout << "center of node " << node_depth3[i] << " is " << center_xi << ", " << center_yi << ", " << center_zi << std::endl;
    //             std::cout << "center of node " << nbrs[j] << " is " << center_xj << ", " << center_yj << ", " << center_zj << std::endl << std::endl;
    //         }
    //     }
    // }
    
}

void compare_ewald() {
    HPDMKParams params;
    params.n_per_leaf = 20;
    params.eps = 1e-8;
    params.L = 10.0;

    omp_set_num_threads(16);
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;

    int n_src = 1000;

    double r_src[n_src * 3];
    double charge[n_src];

    auto particles = read_particle_info("/home/xzgao/code/HybridPeriodicDMK/data/particle_info.txt");

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
    std::cout << "total energy is " << std::setprecision(16) << E_total << std::endl;

    // double E_window_residual = 0;
    // double sigma = tree.sigmas[2];
    // #pragma omp parallel for reduction(+:E_window_residual)
    // for (int i = 0; i < n_src - 1; i++) {
    //     for (int j = i + 1; j < n_src; j++) {
    //         for (int mx = -1; mx <= 1; mx++) {
    //             for (int my = -1; my <= 1; my++) {
    //                 for (int mz = -1; mz <= 1; mz++) {
    //                     double r_ij = std::sqrt(std::pow(r_src[i * 3] - r_src[j * 3] + mx * tree.L, 2) + std::pow(r_src[i * 3 + 1] - r_src[j * 3 + 1] + my * tree.L, 2) + std::pow(r_src[i * 3 + 2] - r_src[j * 3 + 2] + mz * tree.L, 2));
    //                     double d = std::erfc(r_ij / (sigma)) / r_ij;
    //                     E_window_residual += charge[i] * charge[j] * d;
    //                 }
    //             }
    //         }
    //     }
    // }

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
    std::cout << "error is " << std::setprecision(16) << std::abs(E_ewald - E_total) / std::abs(E_total) << std::endl;
}


int main() {

    MPI_Init(nullptr, nullptr);
    
    // hpdmk_tree_create(MPI_COMM_WORLD, params, n_src, r_src, charge);
    // test_tree();
    compare_ewald();

    MPI_Finalize();

    return 0;
}