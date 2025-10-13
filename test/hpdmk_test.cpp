#include <gtest/gtest.h>
#include <hpdmk.h>
#include <tree.hpp>
#include <ewald.hpp>

#include <cmath>
#include <complex>
#include <vector>
#include <utils.hpp>
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
    sctl::Vector<double> r_src(n_src * 3);
    sctl::Vector<double> charge(n_src);

    random_init(r_src, 0.0, params.L);
    random_init(charge, -1.0, 1.0);
    double total_charge = std::accumulate(charge.begin(), charge.end(), 0.0);
    charge -= total_charge / n_src;
    ASSERT_NEAR(std::accumulate(charge.begin(), charge.end(), 0.0), 0.0, 1e-12);

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);
    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src, charge);
    tree.form_outgoing_pw();

    int n_trials = 10;

    // verify root
    auto &outgoing_pw_root = tree.outgoing_pw[tree.root()];
    auto n_root = tree.n_window;
    auto d_root = 2 * n_root + 1;
    auto dk_root = tree.delta_k[0];

    std::mt19937 generator;
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


void compare_energy() {
    HPDMKParams params;
    params.n_per_leaf = 10;
    params.eps = 1e-4;
    params.L = 20.0;

    omp_set_num_threads(1);

    int n_src;
    int ntrials = 10;
    for (int i = 0; i < ntrials; i++) {
        n_src = 1000 + i * 100;
        sctl::Vector<double> r_src(n_src * 3);
        sctl::Vector<double> charge(n_src);

        random_init(r_src, 0.0, params.L);
        random_init(charge, -1.0, 1.0);
        double total_charge = std::accumulate(charge.begin(), charge.end(), 0.0);
        charge -= total_charge / n_src;

        ASSERT_NEAR(std::accumulate(charge.begin(), charge.end(), 0.0), 0.0, 1e-12);

        double s = 4.0;
        double alpha = 1.0;
        Ewald ewald(params.L, s, alpha, 1.0, &charge[0], &r_src[0], n_src);
        double E_ewald = ewald.compute_energy();

        // std::cout << "E_ewald: " << E_ewald << std::endl;

        const sctl::Comm sctl_comm(MPI_COMM_WORLD);
        HPDMKPtTree<double> tree(sctl_comm, params, r_src, charge);

        tree.form_outgoing_pw();
        tree.form_incoming_pw();

        double E_hpdmk_window = tree.eval_energy_window_direct();
        double E_hpdmk_diff = tree.eval_energy_diff_direct();
        double E_hpdmk_res = tree.eval_energy_res_direct();
        // std::cout << "E_hpdmk_window: " << E_hpdmk_window << ", E_hpdmk_diff: " << E_hpdmk_diff << ", E_hpdmk_res: " << E_hpdmk_res << std::endl;
        double E_hpdmk = E_hpdmk_window + E_hpdmk_diff + E_hpdmk_res;

        double E_direct_window = tree.eval_energy_window_direct();
        double E_direct_diff = tree.eval_energy_diff_direct();
        double E_direct_res = tree.eval_energy_res_direct();
        double E_direct = E_direct_window + E_direct_diff + E_direct_res;
        // std::cout << "E_direct_window: " << E_direct_window << ", E_direct_diff: " << E_direct_diff << ", E_direct_res: " << E_direct_res << std::endl;

        // std::cout << "eval_energy done, E_hpdmk: " << E_hpdmk << ", E_direct: " << E_direct << std::endl;

        ASSERT_NEAR(E_hpdmk_window, E_direct_window, 1e-3);
        ASSERT_NEAR(E_hpdmk_diff, E_direct_diff, 1e-3);
        ASSERT_NEAR(E_hpdmk_res, E_direct_res, 1e-3);
        ASSERT_NEAR(E_hpdmk, E_direct, 1e-3);
        ASSERT_NEAR(E_hpdmk, E_ewald, 1e-3);
    }
}



TEST(HPDMKTest, BasicAssertions) {
    MPI_Init(nullptr, nullptr);

    // compare_planewave(10);
    compare_planewave(1000);

    compare_energy();
    
    MPI_Finalize();
}