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


void mc_runtime(int n_src, int n_src_per_leaf, double eps, double L) {
    HPDMKParams params;
    params.n_per_leaf = n_src_per_leaf;
    params.eps = eps;
    params.L = L;

    sctl::Vector<float> r_src(n_src * 3);
    sctl::Vector<float> charge(n_src);

    hpdmk::random_init(r_src, 0.0f, float(params.L));
    hpdmk::random_init(charge, -1.0f, 1.0f);
    hpdmk::unify_charge(charge);

    std::mt19937 generator;
    std::uniform_real_distribution<float> distribution(0.0f, float(params.L));
    std::uniform_int_distribution<int> distribution_int(0, n_src - 1);

    omp_set_num_threads(64);
    const sctl::Comm sctl_comm(MPI_COMM_WORLD);
    hpdmk::HPDMKPtTree<float> tree(sctl_comm, params, r_src, charge);

    std::cout << "init tree done" << std::endl;

    int depth = tree.level_indices.Dim() + 1;


    double t_shift = 0.0;
    double t_update = 0.0;

    double tw = 0.0;
    double td = 0.0;
    double tr = 0.0;
    double tpw = 0.0;
    double tl = 0.0;

    int rounds;
    if (depth <= 8) {
        rounds = 10000;
    } else {
        rounds = 1000;
    }

    for (int n_threads = 1; n_threads <= 1; n_threads *= 2) {
        omp_set_num_threads(n_threads);
        for (int i = 0; i < rounds; i++) {
            int idx = distribution_int(generator);
            int mapped_idx = tree.indices_invmap[idx];

            double dx = distribution(generator);
            double dy = distribution(generator);
            double dz = distribution(generator);

            double q = 1.0;
            double x_o = r_src[idx * 3];
            double y_o = r_src[idx * 3 + 1];
            double z_o = r_src[idx * 3 + 2];

            double x_t = hpdmk::my_mod(x_o + dx, params.L);
            double y_t = hpdmk::my_mod(y_o + dy, params.L);
            double z_t = hpdmk::my_mod(z_o + dz, params.L);

            auto start = std::chrono::high_resolution_clock::now();
            tree.eval_shift_energy(idx, dx, dy, dz);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time = end - start;
            t_shift += time.count();

            // auto start_locate = std::chrono::high_resolution_clock::now();
            // tree.locate_particle(tree.path_to_target, x_t, y_t, z_t);
            // tree.locate_particle(tree.path_to_origin, x_o, y_o, z_o);
            // auto end_locate = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> time_locate = end_locate - start_locate;
            // tl += time_locate.count();

            // auto start_pw = std::chrono::high_resolution_clock::now();
            // tree.form_outgoing_pw_single(tree.outgoing_pw_target, tree.path_to_target, x_t, y_t, z_t, q);
            // tree.form_outgoing_pw_single(tree.outgoing_pw_origin, tree.path_to_origin, x_o, y_o, z_o, q);
            // auto end_pw = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> time_pw = end_pw - start_pw;
            // tpw += time_pw.count();

            // auto start_window = std::chrono::high_resolution_clock::now();
            // tree.eval_shift_energy_window();
            // auto end_window = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> time_window = end_window - start_window;
            // tw += time_window.count();

            // auto start_diff = std::chrono::high_resolution_clock::now();
            // tree.eval_shift_energy_diff(mapped_idx);
            // auto end_diff = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> time_diff = end_diff - start_diff;
            // td += time_diff.count();

            // auto start_res = std::chrono::high_resolution_clock::now();
            // tree.eval_shift_energy_res(mapped_idx, tree.path_to_target, x_t, y_t, z_t, q);
            // tree.eval_shift_energy_res(mapped_idx, tree.path_to_origin, x_o, y_o, z_o, q);
            // auto end_res = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> time_res = end_res - start_res;
            // tr += time_res.count();

            auto start_u = std::chrono::high_resolution_clock::now();
            tree.update_shift(idx, dx, dy, dz);
            auto end_u = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_u = end_u - start_u;
            t_update += time_u.count();
        }

        double avg_t_shift = t_shift / rounds;
        double avg_t_update = t_update / rounds;

        // double avg_tl = tl / rounds;
        // double avg_tw = tw / rounds;
        // double avg_td = td / rounds;
        // double avg_tr = tr / rounds;
        // double avg_tpw = tpw / rounds;
        
        std::cout << "n_threads: " << n_threads << ", depth: " << depth << std::endl;
        std::cout << "avg time update: " << avg_t_update << ", avg time shift: " << avg_t_shift << std::endl;
        // std::cout << "avg time locate: " << avg_tl << ", avg time pw: " << avg_tpw << ", avg time window: " << avg_tw << ", avg time diff: " << avg_td << ", avg time res: " << avg_tr << std::endl;

        std::ofstream outfile("data/dmk_energyshift_runtime.csv", std::ios::app);
        outfile << n_src << "," << n_src_per_leaf << "," << eps << "," << L << "," << depth << "," << n_threads << "," << avg_t_update << "," << avg_t_shift << std::endl;
        outfile.close();
    }
}

int main() {
    MPI_Init(nullptr, nullptr);

    double rho_0 = 200.0;

    std::ofstream outfile("data/dmk_energyshift_runtime.csv");
    outfile << "n_src,n_src_per_leaf,eps,L,depth,n_threads,time_update,time_shift" << std::endl;
    outfile.close();

    for (int scale = 0; scale < 15; scale ++) {
        int n_src = int(std::ceil(10000 * std::pow(2.0, scale)) / 2) * 2;
        int n_src_per_leaf = 50;
        double eps = 1e-3;
        double L = std::pow(n_src / rho_0, 1.0 / 3.0);

        std::cout << "n_src: " << n_src << ", n_src_per_leaf: " << n_src_per_leaf << ", eps: " << eps << ", L: " << L << ", density: " << n_src / (L * L * L) << std::endl;

        mc_runtime(n_src, n_src_per_leaf, eps, L);
    }

    MPI_Finalize();

    return 0;
}