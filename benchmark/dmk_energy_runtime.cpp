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


void dmk_runtime(int n_src, int n_src_per_leaf, float eps, float L) {
    HPDMKParams params;
    params.n_per_leaf = n_src_per_leaf;
    params.eps = eps;
    params.L = float(L);
    params.nufft_eps = float(1e-4);
    params.nufft_threshold = int(4000);

    sctl::Vector<float> r_src(n_src * 3);
    sctl::Vector<float> charge(n_src);

    hpdmk::random_init(r_src, 0.0f, float(params.L));
    hpdmk::random_init(charge, -1.0f, 1.0f);
    hpdmk::unify_charge(charge);

    std::cout << "init r_src and charge done" << std::endl;

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    hpdmk::HPDMKPtTree<float> tree(sctl_comm, params, r_src, charge);

    auto start = std::chrono::high_resolution_clock::now();
    tree.form_outgoing_pw();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_outgoing_pw = end - start;

    start = std::chrono::high_resolution_clock::now();
    tree.form_incoming_pw();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_incoming_pw = end - start;

    start = std::chrono::high_resolution_clock::now();
    double E_window = tree.eval_energy_window();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_window = end - start;

    start = std::chrono::high_resolution_clock::now();
    double E_difference = tree.eval_energy_diff();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_difference = end - start;

    start = std::chrono::high_resolution_clock::now();
    double E_residual = tree.eval_energy_res();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_residual = end - start;

    double time_total = time_outgoing_pw.count() + time_incoming_pw.count() + time_window.count() + time_difference.count() + time_residual.count();

    std::ofstream outfile("data/dmk_energy_runtime.csv", std::ios::app);
    outfile << n_src << "," << n_src_per_leaf << "," << eps << "," << L << "," << time_outgoing_pw.count() << "," << time_incoming_pw.count() << "," << time_window.count() << "," << time_difference.count() << "," << time_residual.count() << "," << time_total << std::endl;
    outfile.close();
}

int main() {
    omp_set_num_threads(1);

    MPI_Init(nullptr, nullptr);

    double rho_0 = 1.0;

    std::ofstream outfile("data/dmk_energy_runtime.csv");
    outfile << "n_src,n_src_per_leaf,eps,L,time_outgoing_pw,time_incoming_pw,time_window,time_difference,time_residual,time_total" << std::endl;
    outfile.close();

    for (int scale = 2; scale <= 10; scale ++) {
        int n_src = 1000 * std::pow(2, scale);
        int n_src_per_leaf = 100;
        float eps = 1e-3;
        float L = std::pow(n_src / rho_0, 1.0 / 3.0);

        std::cout << "n_src: " << n_src << ", n_src_per_leaf: " << n_src_per_leaf << ", eps: " << eps << ", L: " << L << ", density: " << n_src / (L * L * L) << std::endl;

        dmk_runtime(n_src, n_src_per_leaf, eps, L);
    }

    MPI_Finalize();

    return 0;
}