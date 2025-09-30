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

void dmk_shift_runtime(int n_src, int n_src_per_leaf, double eps, double L) {
    HPDMKParams params;
    params.n_per_leaf = n_src_per_leaf;
    params.eps = eps;
    params.L = L;

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

    std::cout << "init tree" << std::endl;
    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src_vec, charge_vec);
    std::cout << "init tree done" << std::endl;

    omp_set_num_threads(1);

    std::cout << "init planewave coeffs" << std::endl;
    tree.init_planewave_coeffs();
    std::cout << "init planewave coeffs done" << std::endl;

    int depth = tree.level_indices.Dim() + 1;

    double ts = 0.0;
    double tu = 0.0;

    int rounds = 1000;
    for (int i = 0; i < rounds; i++) {
        double dx = distribution(generator);
        double dy = distribution(generator);
        double dz = distribution(generator);

        auto start = std::chrono::high_resolution_clock::now();
        tree.energy_shift(i, dx, dy, dz);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_shift = end - start;
        ts += time_shift.count();
        
        start = std::chrono::high_resolution_clock::now();
        tree.update_shift(i, dx, dy, dz);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_update = end - start;
        tu += time_update.count();
    }


    double avg_ts = ts / rounds;
    double avg_tu = tu / rounds;
    double time_total = avg_ts + avg_tu;

    std::ofstream outfile("data/dmk_shift_runtime.csv", std::ios::app);
    outfile << n_src << "," << n_src_per_leaf << "," << eps << "," << L << "," << depth << "," << avg_ts << "," << avg_tu << "," << time_total << std::endl;
    outfile.close();
}

int main() {
    MPI_Init(nullptr, nullptr);

    double rho_0 = 200.0;

    std::ofstream outfile("data/dmk_shift_runtime.csv");
    outfile << "n_src,n_src_per_leaf,eps,L,depth,ts,tu,tt" << std::endl;
    outfile.close();

    for (int scale = 0; scale <= 4; scale ++) {
        int n_src = 10000 * std::pow(8, scale);
        int n_src_per_leaf = 500;
        double eps = 1e-3;
        double L = std::pow(n_src / rho_0, 1.0 / 3.0);

        std::cout << "n_src: " << n_src << ", n_src_per_leaf: " << n_src_per_leaf << ", eps: " << eps << ", L: " << L << ", density: " << n_src / (L * L * L) << std::endl;

        dmk_shift_runtime(n_src, n_src_per_leaf, eps, L);
    }

    MPI_Finalize();

    return 0;
}