#include <gtest/gtest.h>
#include <hpdmk.h>
#include <tree.hpp>

#include <cmath>
#include <complex>
#include <vector>
#include <random>
#include <omp.h>
#include <mpi.h>

using namespace hpdmk;

void compare_planewave(int threshold) {
    HPDMKParams params;
    params.n_per_leaf = 5;
    params.eps = 1e-3;
    params.L = 20.0;
    params.nufft_eps = 1e-8;
    params.nufft_threshold = threshold;

    omp_set_num_threads(16);

    int n_src = 1000;
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
    tree.form_outgoing_pw();

    int n_trials = 10;

    // verify root
    auto &outgoing_pw_root = tree.outgoing_pw[tree.root()];
    auto n_root = tree.n_window;
    auto d_root = 2 * n_root + 1;
    auto dk_root = tree.delta_k[0];
    std::uniform_real_distribution<double> distribution_xy(0, 2 * n_root);
    std::uniform_real_distribution<double> distribution_z(0, n_root);

    for (int i = 0; i < n_trials; i++) {
        int id_x = int(ceil(distribution_xy(generator)));
        int id_y = int(ceil(distribution_xy(generator)));
        int id_z = int(ceil(distribution_z(generator)));

        double kx = (id_x - n_root) * dk_root;
        double ky = (id_y - n_root) * dk_root;
        double kz = (id_z - n_root) * dk_root;

        // std::cout << "kx: " << kx << ", ky: " << ky << ", kz: " << kz << std::endl;

        auto rho = outgoing_pw_root[id_x + id_y * d_root + id_z * d_root * d_root];
        std::complex<double> rho_direct = 0;
        for (int i = 0; i < n_src; i++) {
            double x = r_src[i * 3];
            double y = r_src[i * 3 + 1];
            double z = r_src[i * 3 + 2];
            rho_direct += std::exp( - std::complex<double>(0, 1) * (kx * x + ky * y + kz * z)) * std::complex<double>(charge[i], 0);
        }
        ASSERT_NEAR(std::real(rho) / std::real(rho_direct), 1, 1e-3);
        ASSERT_NEAR(std::imag(rho) / std::imag(rho_direct), 1, 1e-3);
    }

    // verify non-root
    for (int l = 2; l < tree.max_depth - 1; l++) {
        for (auto i_node : tree.level_indices[l]) {
            if (tree.node_particles[i_node].Dim() == 0) {
                continue;
            }

            // std::cout << "l: " << l << ", i_node: " << i_node << ", num_particles: " << tree.node_particles[i_node].Dim() << std::endl;

            auto &outgoing_pw_l = tree.outgoing_pw[i_node];
            auto n_l = tree.n_diff;
            auto d_l = 2 * n_l + 1;
            auto dk_l = tree.delta_k[l];
            std::uniform_real_distribution<double> distribution_xy(0, 2 * n_l);
            std::uniform_real_distribution<double> distribution_z(0, n_l);

            // std::cout << "outgoing_pw_l dim: " << outgoing_pw_l.Dim() << std::endl;

            for (int i = 0; i < n_trials; i++) {
                int id_x = int(ceil(distribution_xy(generator)));
                int id_y = int(ceil(distribution_xy(generator)));
                int id_z = int(ceil(distribution_z(generator)));

                double kx = (id_x - n_l) * dk_l;
                double ky = (id_y - n_l) * dk_l;
                double kz = (id_z - n_l) * dk_l;

                // std::cout << "l: " << l << ", kx: " << kx << ", ky: " << ky << ", kz: " << kz << ", inode: " << i_node << std::endl;
                
                auto rho = outgoing_pw_l[id_x + id_y * d_l + id_z * d_l * d_l];

                // std::cout << "rho: " << std::real(rho) << ", " << std::imag(rho) << std::endl;
                std::complex<double> rho_direct = 0;

                auto& r_src_node = tree.r_src_sorted;
                auto& charge_node = tree.charge_sorted;

                // std::cout << "particles in node: " << tree.node_particles[i_node] << std::endl;

                for (auto j_particle : tree.node_particles[i_node]) {
                    double x = r_src_node[j_particle * 3];
                    double y = r_src_node[j_particle * 3 + 1];
                    double z = r_src_node[j_particle * 3 + 2];
                    double charge = charge_node[j_particle];
                    // std::cout << "x: " << x << ", y: " << y << ", z: " << z << ", charge: " << charge << std::endl;
                    rho_direct += std::exp( - std::complex<double>(0, 1) * (kx * x + ky * y + kz * z)) * std::complex<double>(charge, 0);
                }

                // std::cout << "rho_direct: " << std::real(rho_direct) << ", " << std::imag(rho_direct) << std::endl;
                
                ASSERT_NEAR(std::real(rho) / std::real(rho_direct), 1, 1e-3);
                ASSERT_NEAR(std::imag(rho) / std::imag(rho_direct), 1, 1e-3);

                // std::cout << "outgoing_pw_l, " << "l: " << l << ", " << "rho: " << std::real(rho) << ", " << std::imag(rho) << ", rho_direct: " << std::real(rho_direct) << ", " << std::imag(rho_direct) << std::endl;
            }
            break;
        }
    }


    // check the incoming planewave
    tree.form_incoming_pw();
    for (int l = 2; l < tree.max_depth - 1; l++) {
        for (auto i_node : tree.level_indices[l]) {
            if (isleaf(tree.GetNodeAttr()[i_node])) {
                continue;
            }

            // std::cout << "l: " << l << ", i_node: " << i_node << ", num_particles: " << tree.node_particles[i_node].Dim() << std::endl;

            auto &incoming_pw_l = tree.incoming_pw[i_node];
            auto n_l = tree.n_diff;
            auto d_l = 2 * n_l + 1;
            auto dk_l = tree.delta_k[l];
            std::uniform_real_distribution<double> distribution_xy(0, 2 * n_l);
            std::uniform_real_distribution<double> distribution_z(0, n_l);

            // std::cout << "outgoing_pw_l dim: " << outgoing_pw_l.Dim() << std::endl;

            for (int i = 0; i < n_trials; i++) {
                int id_x = int(ceil(distribution_xy(generator)));
                int id_y = int(ceil(distribution_xy(generator)));
                int id_z = int(ceil(distribution_z(generator)));

                double kx = (id_x - n_l) * dk_l;
                double ky = (id_y - n_l) * dk_l;
                double kz = (id_z - n_l) * dk_l;

                // std::cout << "l: " << l << ", kx: " << kx << ", ky: " << ky << ", kz: " << kz << ", inode: " << i_node << std::endl;
                
                auto rho = incoming_pw_l[id_x + id_y * d_l + id_z * d_l * d_l];

                // std::cout << "rho: " << std::real(rho) << ", " << std::imag(rho) << std::endl;
                std::complex<double> rho_direct = 0;

                auto& r_src_node = tree.r_src_sorted;
                auto& charge_node = tree.charge_sorted;

                auto& neighbors = tree.neighbors[i_node].colleague;
                assert(neighbors.Dim() == 26);

                auto center_xi = tree.centers[i_node * 3];
                auto center_yi = tree.centers[i_node * 3 + 1];
                auto center_zi = tree.centers[i_node * 3 + 2];

                for (auto j_node : neighbors) {
                    for (auto j_particle : tree.node_particles[j_node]) {
                        double x = r_src_node[j_particle * 3];
                        double y = r_src_node[j_particle * 3 + 1];
                        double z = r_src_node[j_particle * 3 + 2];
                        double charge = charge_node[j_particle];

                        // std::cout << "center i: "<< center_xi << ", " << center_yi << ", " << center_zi << std::endl;
                        // std::cout << "x: " << x << ", y: " << y << ", z: " << z << std::endl;

                        // shift the particle j is needed
                        if (x - center_xi > tree.L / 2) {
                            x -= tree.L;
                        } else if (x - center_xi < -tree.L / 2) {
                            x += tree.L;
                        }
                        if (y - center_yi > tree.L / 2) {
                            y -= tree.L;
                        } else if (y - center_yi < -tree.L / 2) {
                            y += tree.L;
                        }
                        if (z - center_zi > tree.L / 2) {
                            z -= tree.L;
                        } else if (z - center_zi < -tree.L / 2) {
                            z += tree.L;
                        }

                        // std::cout << "shifted x: " << x << ", y: " << y << ", z: " << z << std::endl;
                        // std::cout << "x: " << x << ", y: " << y << ", z: " << z << ", charge: " << charge << std::endl;
                        
                        rho_direct += std::exp(std::complex<double>(0, 1) * (kx * x + ky * y + kz * z)) * std::complex<double>(charge, 0);
                    }
                }

                // std::cout << "rho_direct: " << std::real(rho_direct) << ", " << std::imag(rho_direct) << std::endl;
                
                ASSERT_NEAR(std::real(rho) / std::real(rho_direct), 1, 1e-3);
                ASSERT_NEAR(std::imag(rho) / std::imag(rho_direct), 1, 1e-3);
                // std::cout << "incoming_pw_l, " << "l: " << l << ", " << "rho: " << std::real(rho) << ", " << std::imag(rho) << ", rho_direct: " << std::real(rho_direct) << ", " << std::imag(rho_direct) << std::endl;
            }
            break;
        }
    }
}


TEST(PlanewaveTest, BasicAssertions) {
    MPI_Init(nullptr, nullptr);

    // compare_planewave(10);
    compare_planewave(1000);
    
    MPI_Finalize();
}