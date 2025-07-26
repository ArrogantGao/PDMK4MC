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


void ewald_runtime(int n_src, double s, double L) {
    sctl::Vector<double> r_src_vec(n_src * 3);
    sctl::Vector<double> charge_vec(n_src);

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, params.L);

    std::cout << "init r_src and charge" << std::endl;
    for (int i = 0; i < n_src; i++) {
        r_src_vec[i * 3] = distribution(generator);
        r_src_vec[i * 3 + 1] = distribution(generator);
        r_src_vec[i * 3 + 2] = distribution(generator);
        charge_vec[i] = std::pow(-1, i) * 1.0;
    }
    std::cout << "init r_src and charge done" << std::endl;

    omp_set_num_threads(32);

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    double q[n_src];
    double r[n_src][3];
    for (int i = 0; i < n_src; i++) {
        q[i] = charge_vec[i];
        r[i][0] = r_src_vec[i * 3 + 0];
        r[i][1] = r_src_vec[i * 3 + 1];
        r[i][2] = r_src_vec[i * 3 + 2];
    }

    std::cout << "init ewald" << std::endl;
    hpdmk::Ewald ewald(L, s, 2.5 * s / L, 1.0, q, r, n_src);
    std::cout << "init ewald done" << std::endl;

    double time_total = 0.0;

    int n_samples = 100;

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
    outfile << n_src << "," << L << "," << s << "," << 2.5 * s / L << "," << avg_time << std::endl;
    outfile.close();
}

int main() {
    MPI_Init(nullptr, nullptr);

    double rho_0 = 200.0;

    std::ofstream outfile("data/ewald_runtime.csv");
    outfile << "n_src,L,s,alpha,time_total" << std::endl;
    outfile.close();

    for (int scale = 0; scale <= 4; scale ++) {
        int n_src = 10000 * std::pow(8, scale);
        double L = std::pow(n_src / rho_0, 1.0 / 3.0);
        double s = 2.5;

        std::cout << "n_src: " << n_src << ", s: " << s << ", L: " << L << ", density: " << n_src / (L * L * L) << std::endl;

        mc_runtime(n_src, s, L);
    }

    MPI_Finalize();

    return 0;
}