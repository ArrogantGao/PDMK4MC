#ifndef EWALD_HPP
#define EWALD_HPP

#include <hpdmk.h>
#include <vesin.h>
#include <vector>
#include <array>
#include <complex>

#include <sctl.hpp>
#include <utils.hpp>

namespace hpdmk {
    struct Ewald {
        Ewald(const double L, const double s, const double alpha, const double eps, const double *q, const double (*r)[3], const int n_particles);

        int n_particles;

        double L; // the system is assumed to be a cube with side length L
        double s;
        double alpha;
        double eps;

        double V;
        double r_c;
        double k_c;

        double *q;
        double (*r)[3];

        std::vector<double> k; // kx, ky, kz are the same since Lx = Ly = Lz

        VesinNeighborList neighbors;

        CubicTensor<std::complex<double>> planewave_coeffs;
        CubicTensor<double> interaction_matrix;

        void init_interaction_matrix();
        void init_planewave_coeffs();

        double compute_energy();

        std::vector<int> target_neighbors;
        std::vector<double> target_distances;
        CubicTensor<std::complex<double>> target_planewave_coeffs;

        void collect_target_neighbors(const double trg_x, const double trg_y, const double trg_z);
        double compute_potential(const double trg_x, const double trg_y, const double trg_z);
    };
} // namespace hpdmk

#endif