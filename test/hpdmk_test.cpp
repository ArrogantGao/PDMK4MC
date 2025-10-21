#include <gtest/gtest.h>
#include <hpdmk.h>
#include <tree.hpp>
#include <ewald.hpp>
#include <upward_pass.hpp>

#include <cmath>
#include <complex>
#include <algorithm>
#include <utils.hpp>
#include <random>
#include <omp.h>
#include <mpi.h>

using namespace hpdmk;

void compare_nufft_proxycharge(int threshold) {
    HPDMKParams params;
    params.n_per_leaf = 5;
    params.eps = 1e-3;
    params.L = 20.0;
    params.nufft_eps = 1e-8;
    params.nufft_threshold = threshold;

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
    decltype(tree.outgoing_pw) outgoing_pw;
    hpdmk::upward_pass(tree, outgoing_pw);

    const auto n_pw = tree.n_diff;
    const int last_pw_i = n_pw * n_pw * (n_pw / 2 + 1);
    for (int i_level = 2; i_level < tree.n_levels(); ++i_level) {
        for (auto &i_box : tree.level_indices[i_level]) {
            if (!tree.r_src_cnt_all[i_box])
                continue;

            ASSERT_EQ(outgoing_pw[i_box].Dim(), tree.outgoing_pw[i_box].Dim());
            if (!outgoing_pw[i_box].Dim())
                continue;

            double l2_norm_real{0}, l2_norm_imag{0};
            double sum_real{0}, sum_imag{0};
            auto real_diff_max_index = 0;
            auto real_diff_max = 0.0;
            for (int i = 0; i < last_pw_i; i++) {
                auto real_diff = std::real(outgoing_pw[i_box][i]) - std::real(tree.outgoing_pw[i_box][i]);
                if (std::abs(real_diff) > real_diff_max) {
                    real_diff_max = std::abs(real_diff);
                    real_diff_max_index = i;
                }
                l2_norm_real += std::pow(std::real(outgoing_pw[i_box][i]) - std::real(tree.outgoing_pw[i_box][i]), 2);
                sum_real += std::real(tree.outgoing_pw[i_box][i]) * std::real(tree.outgoing_pw[i_box][i]);
                l2_norm_imag += std::pow(std::imag(outgoing_pw[i_box][i]) - std::imag(tree.outgoing_pw[i_box][i]), 2);
                sum_imag += std::imag(tree.outgoing_pw[i_box][i]) * std::imag(tree.outgoing_pw[i_box][i]);
            }
            if (sum_real)
                ASSERT_NEAR(std::sqrt(l2_norm_real / sum_real), 0, params.eps);
            if (sum_imag)
                ASSERT_NEAR(std::sqrt(l2_norm_imag / sum_imag), 0, params.eps);
        }
    }
}

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
                
                ASSERT_NEAR(std::real(rho), std::real(rho_direct), 1e-3);
                ASSERT_NEAR(std::imag(rho), std::imag(rho_direct), 1e-3);

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
                
                auto rho = incoming_pw_l[id_x + id_y * d_l + id_z * d_l * d_l] - std::conj(outgoing_pw_l[id_x + id_y * d_l + id_z * d_l * d_l]);

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
                
                ASSERT_NEAR(std::real(rho), std::real(rho_direct), 1e-3);
                ASSERT_NEAR(std::imag(rho), std::imag(rho_direct), 1e-3);
                // std::cout << "incoming_pw_l, " << "l: " << l << ", " << "rho: " << std::real(rho) << ", " << std::imag(rho) << ", rho_direct: " << std::real(rho_direct) << ", " << std::imag(rho_direct) << std::endl;
            }
            break;
        }
    }
}

void compare_planewave_single(){

    HPDMKParams params;
    params.n_per_leaf = 5;
    params.eps = 1e-3;
    params.L = 20.0;
    params.nufft_eps = 1e-8;

    int n_src = 1000;
    sctl::Vector<double> r_src(n_src * 3);
    sctl::Vector<double> charge(n_src);

    omp_set_num_threads(1);

    random_init(r_src, 0.0, params.L);
    random_init(charge, -1.0, 1.0);
    unify_charge(charge);
    ASSERT_NEAR(std::accumulate(charge.begin(), charge.end(), 0.0), 0.0, 1e-12);

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);
    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src, charge);
    tree.form_outgoing_pw();
    tree.form_incoming_pw();

    int n_trials = 10;
    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, tree.L);

    std::complex<double> rho, rho_direct;
    double kx, ky, kz;

    for (int i_trial = 0; i_trial < n_trials; i_trial++) {

        double x_0 = distribution(generator);
        double y_0 = distribution(generator);
        double z_0 = distribution(generator);

        tree.locate_particle(tree.path_to_origin, x_0, y_0, z_0);
        for (auto i_node : tree.path_to_origin) {
            ASSERT_TRUE(tree.is_in_node(x_0, y_0, z_0, i_node));
        }

        tree.form_outgoing_pw_single(tree.outgoing_pw_origin, tree.path_to_origin, x_0, y_0, z_0, 1.0);

        // compare the root node outgoing
        auto &pw_root = tree.outgoing_pw_origin[0];
        auto n_window = tree.n_window;
        auto d_window = 2 * n_window + 1;
        auto delta_k_window = tree.delta_k[0];
        for (int id_x = 0; id_x < d_window; id_x++) {
            for (int id_y = 0; id_y < d_window; id_y++) {
                for (int id_z = 0; id_z < n_window + 1; id_z++) {
                    kx = (id_x - n_window) * delta_k_window;
                    ky = (id_y - n_window) * delta_k_window;
                    kz = (id_z - n_window) * delta_k_window;
                    rho_direct = std::exp( - std::complex<double>(0, kx * x_0 + ky * y_0 + kz * z_0)) * std::complex<double>(1.0, 0);
                    rho = pw_root[id_x + id_y * d_window + id_z * d_window * d_window];

                    // std::cout << "xyz: " << x_0 << ", " << y_0 << ", " << z_0 << ", kx: " << kx << ", ky: " << ky << ", kz: " << kz << ", rho: " << std::real(rho) << ", " << std::imag(rho) << ", rho_direct: " << std::real(rho_direct) << ", " << std::imag(rho_direct) << std::endl;

                    ASSERT_NEAR(std::real(rho), std::real(rho_direct), 1e-3);
                    ASSERT_NEAR(std::imag(rho), std::imag(rho_direct), 1e-3);
                }
            }
        }

        // compare the non-root nodes outgoing
        int end_depth = std::min(int(tree.max_depth - 1), int(tree.path_to_origin.Dim()));
        for (int l = 2; l < end_depth; l++) {
            auto i_node = tree.path_to_origin[l];
            auto &pw_l = tree.outgoing_pw_origin[l];
            auto n_l = tree.n_diff;
            auto d_l = 2 * n_l + 1;
            auto delta_k_l = tree.delta_k[l];
            for (int id_x = 0; id_x < d_l; id_x++) {
                for (int id_y = 0; id_y < d_l; id_y++) {
                    for (int id_z = 0; id_z < n_l + 1; id_z++) {
                        kx = (id_x - n_l) * delta_k_l;
                        ky = (id_y - n_l) * delta_k_l;
                        kz = (id_z - n_l) * delta_k_l;
                        rho_direct = std::exp( - std::complex<double>(0, kx * x_0 + ky * y_0 + kz * z_0)) * std::complex<double>(1.0, 0);
                        rho = pw_l[id_x + id_y * d_l + id_z * d_l * d_l];

                        ASSERT_NEAR(std::real(rho), std::real(rho_direct), 1e-3);
                        ASSERT_NEAR(std::imag(rho), std::imag(rho_direct), 1e-3);
                    }
                }
            }
        }
    }
}

void compare_energy(double eps, int prolate_order) {
    HPDMKParams params;
    params.n_per_leaf = 10;
    params.eps = eps;
    params.prolate_order = prolate_order;
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

        double E_hpdmk_window = tree.eval_energy_window();
        double E_hpdmk_diff = tree.eval_energy_diff();
        double E_hpdmk_res = tree.eval_energy_res();
        // std::cout << "E_hpdmk_window: " << E_hpdmk_window << ", E_hpdmk_diff: " << E_hpdmk_diff << ", E_hpdmk_res: " << E_hpdmk_res << std::endl;
        double E_hpdmk = E_hpdmk_window + E_hpdmk_diff + E_hpdmk_res;

        double E_direct_window = tree.eval_energy_window_direct();
        double E_direct_diff = tree.eval_energy_diff_direct();
        double E_direct_res = tree.eval_energy_res_direct();
        double E_direct = E_direct_window + E_direct_diff + E_direct_res;
        // std::cout << "E_direct_window: " << E_direct_window << ", E_direct_diff: " << E_direct_diff << ", E_direct_res: " << E_direct_res << std::endl;

        // std::cout << "eval_energy done, E_hpdmk: " << E_hpdmk << ", E_direct: " << E_direct << std::endl;

        ASSERT_NEAR((E_hpdmk_window - E_direct_window) / E_ewald, 0, 10 * eps);
        ASSERT_NEAR((E_hpdmk_diff - E_direct_diff) / E_ewald, 0, 10 * eps);
        ASSERT_NEAR((E_hpdmk_res - E_direct_res) / E_ewald, 0, 10 * eps);
        ASSERT_NEAR((E_hpdmk - E_direct) / E_ewald, 0, 10 * eps);
    }
}

void compare_shift_energy(double eps, int prolate_order){

    HPDMKParams params;
    params.n_per_leaf = 10;
    params.eps = eps;
    params.prolate_order = prolate_order;
    params.L = 20.0;

    int n_src = 1000;
    sctl::Vector<double> r_src(n_src * 3);
    sctl::Vector<double> charge(n_src);

    sctl::Vector<double> r_src_new(n_src * 3);

    omp_set_num_threads(1);

    random_init(r_src, 0.0, params.L);
    random_init(charge, -1.0, 1.0);
    unify_charge(charge);
    ASSERT_NEAR(std::accumulate(charge.begin(), charge.end(), 0.0), 0.0, 1e-12);

    r_src_new.SetZero();
    r_src_new += r_src;

    double s = 4.0;
    double alpha = 1.0;
    Ewald ewald(params.L, s, alpha, 1.0, &charge[0], &r_src[0], n_src);
    double E_ewald_old = ewald.compute_energy();
    // std::cout << "E_ewald_old: " << E_ewald_old << std::endl;

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);
    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src, charge);
    tree.form_outgoing_pw();
    tree.form_incoming_pw();

    double E_hpdmk_window_old = tree.eval_energy_window();
    double E_hpdmk_diff_old = tree.eval_energy_diff();
    double E_hpdmk_res_old = tree.eval_energy_res();

    double E_hpdmk_old = tree.eval_energy();

    // std::cout << "E_hpdmk_old: " << E_hpdmk_old << std::endl;

    int n_trials = 10;
    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, tree.L);
    std::uniform_int_distribution<int> distribution_int(0, n_src - 1);

    for (int i_trial = 0; i_trial < n_trials; i_trial++) {

        double dx = distribution(generator);
        double dy = distribution(generator);
        double dz = distribution(generator);
        int i_particle = distribution_int(generator);

        auto E_shift = tree.eval_shift_energy(i_particle, dx, dy, dz);

        // std::cout << "original xyz out: " << r_src[i_particle * 3] << ", " << r_src[i_particle * 3 + 1] << ", " << r_src[i_particle * 3 + 2] << std::endl;

        r_src_new[i_particle * 3] = my_mod(dx + r_src[i_particle * 3], params.L);
        r_src_new[i_particle * 3 + 1] = my_mod(dy + r_src[i_particle * 3 + 1], params.L);
        r_src_new[i_particle * 3 + 2] = my_mod(dz + r_src[i_particle * 3 + 2], params.L);

        // std::cout << "target xyz out: " << r_src_new[i_particle * 3] << ", " << r_src_new[i_particle * 3 + 1] << ", " << r_src_new[i_particle * 3 + 2] << std::endl;

        hpdmk::Ewald ewald_new(params.L, s, alpha, 1.0, &charge[0], &r_src_new[0], n_src);
        double E_ewald_new = ewald_new.compute_energy();

        double E_shift_ewald = E_ewald_new - E_ewald_old;

        hpdmk::HPDMKPtTree<double> tree_new(sctl_comm, params, r_src_new, charge);
        tree_new.form_outgoing_pw();
        tree_new.form_incoming_pw();
        double E_hpdmk_window_new = tree_new.eval_energy_window();
        double E_hpdmk_diff_new = tree_new.eval_energy_diff();
        double E_hpdmk_res_new = tree_new.eval_energy_res();

        // std::cout << "E_ewald_new: " << E_ewald_new << std::endl;
        // std::cout << "E_hpdmk_new: " << E_hpdmk_window_new + E_hpdmk_diff_new + E_hpdmk_res_new << std::endl;

        // std::cout << "dE_window_direct: " << E_hpdmk_window_new - E_hpdmk_window_old << ", dE_diff_direct: " << E_hpdmk_diff_new -   E_hpdmk_diff_old << ", dE_res_direct: " << E_hpdmk_res_new - E_hpdmk_res_old << std::endl;

        // recover the r_src and charge
        r_src_new[i_particle * 3] = r_src[i_particle * 3];
        r_src_new[i_particle * 3 + 1] = r_src[i_particle * 3 + 1];
        r_src_new[i_particle * 3 + 2] = r_src[i_particle * 3 + 2];

        // std::cout << "dx: " << dx << ", dy: " << dy << ", dz: " << dz << ", i_particle: " << i_particle << std::endl; 
        //  std::cout << "E_shift: " << E_shift << ", E_shift_direct: " << E_hpdmk_window_new - E_hpdmk_window_old + E_hpdmk_diff_new - E_hpdmk_diff_old + E_hpdmk_res_new - E_hpdmk_res_old << ", E_shift_ewald: " << E_shift_ewald << std::endl;

        ASSERT_NEAR((E_shift_ewald - E_shift) / E_ewald_old, 0, 10 * eps);
    }
}


void compare_update(){

    HPDMKParams params;
    params.n_per_leaf = 10;
    params.eps = 1e-3;
    params.L = 20.0;
    params.prolate_order = 16;

    int n_src = 1000;
    sctl::Vector<double> r_src(n_src * 3);
    sctl::Vector<double> charge(n_src);

    omp_set_num_threads(1);

    random_init(r_src, 0.0, params.L);
    random_init(charge, -1.0, 1.0);
    unify_charge(charge);
    ASSERT_NEAR(std::accumulate(charge.begin(), charge.end(), 0.0), 0.0, 1e-12);

    double Ewald_old, Ewald_new, Ewald_shift;
    double E_hpdmk, E_hpdmk_shift;

    double s = 4.0;
    double alpha = 1.0;
    Ewald ewald(params.L, s, alpha, 1.0, &charge[0], &r_src[0], n_src);
    Ewald_old = ewald.compute_energy();
    // std::cout << "E_ewald_old: " << E_ewald_old << std::endl;

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);
    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src, charge);
    tree.form_outgoing_pw();
    tree.form_incoming_pw();

    int n_trials = 100;
    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, tree.L);
    std::uniform_int_distribution<int> distribution_int(0, n_src - 1);

    for (int i_trial = 0; i_trial < n_trials; i_trial++) {

        double dx = distribution(generator);
        double dy = distribution(generator);
        double dz = distribution(generator);
        int i_particle = distribution_int(generator);

        E_hpdmk_shift = tree.eval_shift_energy(i_particle, dx, dy, dz);

        r_src[i_particle * 3] = my_mod(dx + r_src[i_particle * 3], params.L);
        r_src[i_particle * 3 + 1] = my_mod(dy + r_src[i_particle * 3 + 1], params.L);
        r_src[i_particle * 3 + 2] = my_mod(dz + r_src[i_particle * 3 + 2], params.L);

        hpdmk::Ewald ewald_new(params.L, s, alpha, 1.0, &charge[0], &r_src[0], n_src);
        Ewald_new = ewald_new.compute_energy();

        Ewald_shift = Ewald_new - Ewald_old;
        Ewald_old = Ewald_new;

        // ASSERT_NEAR((Ewald_shift - E_shift) / Ewald_old, 0, 1e-3);
        tree.update_shift(i_particle, dx, dy, dz);
        // E_hpdmk = tree.eval_energy();
        auto E_hpdmk_window = tree.eval_energy_window();
        auto E_hpdmk_diff = tree.eval_energy_diff();
        auto E_hpdmk_res = tree.eval_energy_res();

        auto E_hpdmk_window_direct = tree.eval_energy_window_direct();
        auto E_hpdmk_diff_direct = tree.eval_energy_diff_direct();
        auto E_hpdmk_res_direct = tree.eval_energy_res_direct();
        
        // std::cout << "window, diff, res: " << E_hpdmk_window << ", " << E_hpdmk_diff << ", " << E_hpdmk_res << std::endl;
        // std::cout << "window_direct, diff_direct, res_direct: " << E_hpdmk_window_direct << ", " << E_hpdmk_diff_direct << ", " << E_hpdmk_res_direct << std::endl;

        E_hpdmk = E_hpdmk_window + E_hpdmk_diff + E_hpdmk_res;

        std::cout << "Ewald_shift: " << Ewald_shift << ", E_hpdmk_shift: " << E_hpdmk_shift << ", E_hpdmk: " << E_hpdmk << ", Ewald_old: " << Ewald_old << std::endl;

        // ASSERT_NEAR((E_hpdmk_window_direct - E_hpdmk_window) / Ewald_old, 0, 1e-2);
        // ASSERT_NEAR((E_hpdmk_diff_direct - E_hpdmk_diff) / Ewald_old, 0, 1e-2);
        // ASSERT_NEAR((E_hpdmk_res_direct - E_hpdmk_res) / Ewald_old, 0, 1e-2);
        ASSERT_NEAR((Ewald_shift - E_hpdmk_shift) / Ewald_old, 0, 1e-2);
    }
}

int main(int argc, char** argv) {
    MPI_Init(nullptr, nullptr);
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

TEST(HPDMKTest, Planewave) {
    compare_planewave(10);
    compare_planewave(1000);
}

TEST(HPDMKTest, PlanewaveProxyNufft) {
    compare_nufft_proxycharge(10);
    compare_nufft_proxycharge(1000);
}

TEST(HPDMKTest, PlanewaveSingle) {
    compare_planewave_single();
}

TEST(HPDMKTest, Energy) {
    compare_energy(1e-3, 16);
    // compare_energy(1e-6, 30);
}

TEST(HPDMKTest, ShiftEnergy) {
    compare_shift_energy(1e-3, 16);
    // compare_shift_energy(1e-6, 30);
}

TEST(HPDMKTest, Update) {
    compare_update();
}
