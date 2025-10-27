#include <hpdmk.h>
#include <random>
#include <omp.h>
#include <mpi.h>
#include <cmath>
#include <numeric>

#include <tree.hpp>
#include <sctl.hpp>
#include <ewald.hpp>
#include <utils.hpp>

using namespace hpdmk;
using namespace std;

void mc_accuracy(int n_src, int n_src_per_leaf, int digits, double L, int rounds) {
    HPDMKParams params;
    params.n_per_leaf = n_src_per_leaf;
    params.digits = digits;
    params.L = L;
    params.init = DIRECT;

    sctl::Vector<double> r_src(n_src * 3);
    sctl::Vector<double> r_src_new(n_src * 3);
    sctl::Vector<double> charge(n_src);

    random_init(r_src, 0.0, double(params.L));
    random_init(charge, -1.0, 1.0);
    unify_charge(charge);

    r_src_new.SetZero();
    r_src_new += r_src;

    double s = 5.0;
    double alpha = 1.0;
    Ewald ewald(params.L, s, alpha, 1.0, &charge[0], &r_src[0], n_src);
    double E_ewald_old = ewald.compute_energy();
    // std::cout << "E_ewald_old: " << E_ewald_old << std::endl;

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);
    HPDMKPtTree<double> tree(sctl_comm, params, r_src, charge);
    tree.form_outgoing_pw();
    tree.form_incoming_pw();

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, tree.L);
    std::uniform_int_distribution<int> distribution_int(0, n_src - 1);

    sctl::Vector<double> abs_err(rounds), rel_err(rounds);

    for (int i_trial = 0; i_trial < rounds; i_trial++) {

        double dx = distribution(generator);
        double dy = distribution(generator);
        double dz = distribution(generator);
        int i_particle = distribution_int(generator);

        auto E_shift = tree.eval_shift_energy(i_particle, dx, dy, dz);

        r_src_new[i_particle * 3] = my_mod(dx + r_src[i_particle * 3], params.L);
        r_src_new[i_particle * 3 + 1] = my_mod(dy + r_src[i_particle * 3 + 1], params.L);
        r_src_new[i_particle * 3 + 2] = my_mod(dz + r_src[i_particle * 3 + 2], params.L);

        Ewald ewald_new(params.L, s, alpha, 1.0, &charge[0], &r_src_new[0], n_src);
        double E_ewald_new = ewald_new.compute_energy();

        double E_shift_ewald = E_ewald_new - E_ewald_old;

        r_src_new[i_particle * 3] = r_src[i_particle * 3];
        r_src_new[i_particle * 3 + 1] = r_src[i_particle * 3 + 1];
        r_src_new[i_particle * 3 + 2] = r_src[i_particle * 3 + 2];

        // cout << "E_shift_ewald: " << E_shift_ewald << ", E_shift: " << E_shift << ", absolute error: " << abs(E_shift_ewald - E_shift) << endl;
        abs_err[i_trial] = abs(E_shift_ewald - E_shift);
        rel_err[i_trial] = abs(E_shift_ewald - E_shift) / E_ewald_old;
    }

    std::cout << "abs_err_mean, rel_err_mean: " << std::accumulate(abs_err.begin(), abs_err.end(), 0.0) / rounds << ", " << std::accumulate(rel_err.begin(), rel_err.end(), 0.0) / rounds << std::endl;
    std::cout << "abs_err_max, rel_err_max: " << *std::max_element(abs_err.begin(), abs_err.end()) << ", " << *std::max_element(rel_err.begin(), rel_err.end()) << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(nullptr, nullptr);
    
    int n_src = 1000;
    int n_src_per_leaf = 50;
    double L = 20.0;
    int rounds = 100;

    std::cout << "Testing 3 digits accuracy" << std::endl;
    mc_accuracy(n_src, n_src_per_leaf, 3, L, rounds);
    std::cout << "Testing 6 digits accuracy" << std::endl;
    mc_accuracy(n_src, n_src_per_leaf, 6, L, rounds);
    std::cout << "Testing 9 digits accuracy" << std::endl;
    mc_accuracy(n_src, n_src_per_leaf, 9, L, rounds);
    std::cout << "Testing 12 digits accuracy" << std::endl;
    mc_accuracy(n_src, n_src_per_leaf, 12, L, rounds);


    MPI_Finalize();

    return 0;
}