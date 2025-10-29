#include <hpdmk.h>
#include <random>
#include <omp.h>
#include <mpi.h>

#include <tree.hpp>
#include <sctl.hpp>

#include <chrono>
#include <fstream>
#include <cstdlib> // for std::atoi, std::atof··

double dmk_energy(sctl::Vector<double> &r_src, sctl::Vector<double> &charge, int n_src, int n_src_per_leaf, int digits, double L) {
    HPDMKParams params;
    params.n_per_leaf = n_src_per_leaf;
    params.digits = digits;
    params.L = L;

    omp_set_num_threads(16);

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);

    hpdmk::HPDMKPtTree<double> tree(sctl_comm, params, r_src, charge);

    tree.form_outgoing_pw();

    // tree.init_planewave_coeffs();
    // double E = tree.energy();

    return 0;
}

int main(int argc, char **argv) {
    int n_src = 100000;
    int n_src_per_leaf = 200;
    int digits = 3;
    double rho = 200.0;

    if (argc > 1) n_src = std::atoi(argv[1]);
    if (argc > 2) n_src_per_leaf = std::atoi(argv[2]);
    if (argc > 3) digits = std::atoi(argv[3]);
    if (argc > 4) rho = std::atof(argv[4]);

    double L = std::pow(n_src / rho, 1.0 / 3.0);

    sctl::Vector<double> r_src(n_src * 3);
    sctl::Vector<double> charge(n_src);
    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, L);
    for (int i = 0; i < n_src; i++) {
        r_src[i * 3] = distribution(generator);
        r_src[i * 3 + 1] = distribution(generator);
        r_src[i * 3 + 2] = distribution(generator);
        charge[i] = std::pow(-1, i) * 1.0;
    }

    MPI_Init(nullptr, nullptr);
    
    double E = dmk_energy(r_src, charge, n_src, n_src_per_leaf, digits, L);
    std::cout << "E: " << E << std::endl;

    MPI_Finalize();

    return 0;
}