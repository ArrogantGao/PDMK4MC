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

template <typename Real>
Real my_max(const sctl::Vector<Real>& vec) {
    Real max_val = vec[0];
    for (int i = 1; i < vec.Dim(); i++) {
        if (vec[i] > max_val) {
            max_val = vec[i];
        }
    }
    return max_val;
}

template <typename Real>
Real my_mean(const sctl::Vector<Real>& vec) {
    Real sum = 0.0;
    for (int i = 0; i < vec.Dim(); i++) {
        sum += abs(vec[i]);
    }
    return sum / Real(vec.Dim());
}

void mc_accuracy_float(int n_src, int n_src_per_leaf, int digits, double L, int rounds) {
    HPDMKParams params;
    params.n_per_leaf = n_src_per_leaf;
    params.digits = digits;
    params.L = float(L);
    params.init = DIRECT;

    sctl::Vector<float> r_src(n_src * 3);
    sctl::Vector<float> charge(n_src);

    sctl::Vector<double> r_src_ref(n_src * 3);
    sctl::Vector<double> charge_ref(n_src);

    random_init(r_src, 0.0f, float(params.L));
    random_init(charge, -1.0f, 1.0f);
    unify_charge(charge);

    for (int i = 0; i < n_src; i++) {
        r_src_ref[i * 3] = double(r_src[i * 3]);
        r_src_ref[i * 3 + 1] = double(r_src[i * 3 + 1]);
        r_src_ref[i * 3 + 2] = double(r_src[i * 3 + 2]);
        charge_ref[i] = double(charge[i]);
    }

    double s = 4.0;
    double alpha = 1.0;
    Ewald ewald(params.L, s, alpha, 1.0, &charge_ref[0], &r_src_ref[0], n_src);
    double E_ewald_old = ewald.compute_energy();

    const sctl::Comm sctl_comm(MPI_COMM_WORLD);
    HPDMKPtTree<float> tree(sctl_comm, params, r_src, charge);
    tree.form_outgoing_pw();
    tree.form_incoming_pw();

    std::mt19937 generator;
    std::uniform_real_distribution<float> distribution(0.0f, float(params.L));
    std::uniform_int_distribution<int> distribution_int(0, n_src - 1);

    sctl::Vector<float> abs_err(rounds), rel_err(rounds);

    for (int i_trial = 0; i_trial < rounds; i_trial++) {
        float dx = distribution(generator);
        float dy = distribution(generator);
        float dz = distribution(generator);
        int i_particle = distribution_int(generator);

        auto E_shift = tree.eval_shift_energy(i_particle, dx, dy, dz);

        r_src_ref[i_particle * 3] = my_mod(dx + r_src[i_particle * 3], float(params.L));
        r_src_ref[i_particle * 3 + 1] = my_mod(dy + r_src[i_particle * 3 + 1], float(params.L));
        r_src_ref[i_particle * 3 + 2] = my_mod(dz + r_src[i_particle * 3 + 2], float(params.L));
        
        Ewald ewald_new(params.L, s, alpha, 1.0, &charge_ref[0], &r_src_ref[0], n_src);
        double E_ewald_new = ewald_new.compute_energy();

        double E_shift_ewald = E_ewald_new - E_ewald_old;

        abs_err[i_trial] = abs(E_shift_ewald - E_shift);
        rel_err[i_trial] = abs_err[i_trial] / abs(E_ewald_old);

        r_src_ref[i_particle * 3] = double(r_src[i_particle * 3]);
        r_src_ref[i_particle * 3 + 1] = double(r_src[i_particle * 3 + 1]);
        r_src_ref[i_particle * 3 + 2] = double(r_src[i_particle * 3 + 2]);
    }

    std::cout << "abs_err_mean, rel_err_mean: " << my_mean(abs_err) << ", " << my_mean(rel_err) << std::endl;
    std::cout << "abs_err_max, rel_err_max: " << my_max(abs_err) << ", " << my_max(rel_err) << std::endl;
}

void mc_accuracy_double(int n_src, int n_src_per_leaf, int digits, double L, int rounds) {
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

    std::cout << "tree.n_levels: " << tree.n_levels() << std::endl;

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
        rel_err[i_trial] = abs(E_shift_ewald - E_shift) / abs(E_ewald_old);
    }

    std::cout << "abs_err_mean, rel_err_mean: " << my_mean(abs_err) << ", " << my_mean(rel_err) << std::endl;
    std::cout << "abs_err_max, rel_err_max: " << my_max(abs_err) << ", " << my_max(rel_err) << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(nullptr, nullptr);
    
    int n_src = 1000;
    int n_src_per_leaf = 3;
    double L = 20.0;
    int rounds = 10;

    std::cout << "Testing 3 digits accuracy for float precision" << std::endl;
    mc_accuracy_float(n_src, n_src_per_leaf, 3, L, rounds);
    std::cout << "Testing 6 digits accuracy for float precision" << std::endl;
    mc_accuracy_float(n_src, n_src_per_leaf, 6, L, rounds);

    std::cout << "Testing 3 digits accuracy for double precision" << std::endl;
    mc_accuracy_double(n_src, n_src_per_leaf, 3, L, rounds);
    std::cout << "Testing 6 digits accuracy for double precision" << std::endl;
    mc_accuracy_double(n_src, n_src_per_leaf, 6, L, rounds);
    std::cout << "Testing 9 digits accuracy for double precision" << std::endl;
    mc_accuracy_double(n_src, n_src_per_leaf, 9, L, rounds);
    std::cout << "Testing 12 digits accuracy for double precision" << std::endl;
    mc_accuracy_double(n_src, n_src_per_leaf, 12, L, rounds);


    MPI_Finalize();

    return 0;
}