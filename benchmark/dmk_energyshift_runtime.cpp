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


void mc_runtime(int n_src, int n_src_per_leaf, int digits, double L, int rounds) {
    HPDMKParams params;
    params.n_per_leaf = n_src_per_leaf;
    params.digits = digits;
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

    int depth = tree.level_indices.Dim() + 1;
    std::cout << "depth: " << depth << std::endl;

    double t_eval = 0.0;
    double t_update = 0.0;

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
            t_eval += time.count();

            start = std::chrono::high_resolution_clock::now();
            tree.update_shift(idx, dx, dy, dz);
            end = std::chrono::high_resolution_clock::now();
            time = end - start;
            t_update += time.count();
        }
        
        t_eval /= rounds;
        t_update /= rounds;

        std::cout << "time update: " << t_update << ", time eval: " << t_eval << std::endl;

        std::ofstream outfile("data/dmk_energyshift_runtime.csv", std::ios::app);
        outfile << n_src << "," << n_src_per_leaf << "," << digits << "," << L << "," << depth << "," << n_threads << "," << t_update << "," << t_eval << std::endl;
        outfile.close();
    }
}

int main(int argc, char** argv) {
    MPI_Init(nullptr, nullptr);

    double rho_0 = 200.0;

    std::ofstream outfile("data/dmk_energyshift_runtime.csv");
    outfile << "n_src,n_src_per_leaf,digits,L,depth,n_threads,time_update,time_shift" << std::endl;
    outfile.close();

    for (int scale = 0; scale < 15; scale+=1) {
        int n_src = int(std::ceil(10000 * std::pow(2.0, scale)) / 2) * 2;
        int n_src_per_leaf = 200;
        int digits = 3;
        double L = std::pow(n_src / rho_0, 1.0 / 3.0);
        int rounds = 10000;

        std::cout << "n_src: " << n_src << ", n_src_per_leaf: " << n_src_per_leaf << ", digits: " << digits << ", L: " << L << ", density: " << n_src / (L * L * L) << std::endl;

        mc_runtime(n_src, n_src_per_leaf, digits, L, rounds);
    }

    MPI_Finalize();

    return 0;
}