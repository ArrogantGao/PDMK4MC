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


void ewald_runtime(int n_src, double s, double L, double alpha) {
    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, L);

    omp_set_num_threads(32);

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    std::cout << "init r_src and charge done" << std::endl;

    std::vector<double> q(n_src);
    std::vector<double> r(n_src * 3);
    for (int i = 0; i < n_src; i++) {
        q[i] = std::pow(-1, i) * 1.0;
        r[i * 3 + 0] = distribution(generator);
        r[i * 3 + 1] = distribution(generator);
        r[i * 3 + 2] = distribution(generator);
    }

    std::cout << "init ewald" << std::endl;
    hpdmk::Ewald ewald(L, s, alpha, 1.0, q, r, n_src);
    std::cout << "init ewald done" << std::endl;

    double time_total = 0.0;

    int n_samples = 10;

    for (int i = 0; i < n_samples; i++) {
        double trg_x = distribution(generator);
        double trg_y = distribution(generator);
        double trg_z = distribution(generator);

        ewald.collect_target_neighbors(trg_x, trg_y, trg_z);

        auto start = std::chrono::high_resolution_clock::now();
        double potential_ewald = ewald.compute_potential(trg_x, trg_y, trg_z);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time = end - start;
        time_total += time.count();
    }

    double avg_time = time_total / n_samples;

    std::ofstream outfile("data/ewald_runtime.csv", std::ios::app);
    outfile << n_src << "," << L << "," << s << "," << alpha << "," << avg_time << std::endl;
    outfile.close();
}

int main() {
    MPI_Init(nullptr, nullptr);

    double rho_0 = 200.0;

    // std::ofstream outfile("data/ewald_runtime.csv");
    // outfile << "n_src,L,s,alpha,time_total" << std::endl;
    // outfile.close();

    for (int scale = 5; scale <= 6; scale ++) {
        int n_src = 10000 * std::pow(4, scale);
        double L = std::pow(n_src / rho_0, 1.0 / 3.0);
        double s = 3;
        double alpha = s / std::sqrt(L);

        std::cout << "n_src: " << n_src << ", s: " << s << ", L: " << L << ", alpha: " << alpha << ", density: " << n_src / (L * L * L) << std::endl;

        ewald_runtime(n_src, s, L, alpha);
    }

    MPI_Finalize();

    return 0;
}