#include <gtest/gtest.h>
#include <hpdmk.h>
#include <random>
#include <omp.h>
#include <mpi.h>

// TEST(HPDMKTest, BasicAssertions) {
//     hpdmk::HPDMKParams<double> params;
//     params.n_per_leaf = 20;
//     params.eps = 1e-4;
//     params.L = 10.0;

//     sctl::Vector<double> r_src(1000);
//     sctl::Vector<double> charge(1000);

//     hpdmk::HPDMKPtTree<double> tree(sctl::Comm::Self(), params, r_src, charge);
// }

int main() {
    std::cout << "Hello, World!" << std::endl;

    HPDMKParams params;
    params.n_per_leaf = 100;
    params.eps = 1e-4;
    params.L = 10.0;

    omp_set_num_threads(16);

    int n_src = 10000;
    double r_src[n_src * 3];
    double charge[n_src];

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, 10);
    std::uniform_real_distribution<double> distribution_charge(-1, 1);

    for (int i = 0; i < n_src; i++) {
        r_src[i * 3] = distribution(generator);
        r_src[i * 3 + 1] = distribution(generator);
        r_src[i * 3 + 2] = distribution(generator);
        charge[i] = distribution_charge(generator);
    }

    MPI_Init(nullptr, nullptr);
    hpdmk_tree_create(MPI_COMM_WORLD, params, n_src, r_src, charge);
    MPI_Finalize();
    return 0;
}