#ifndef EWALD_HPP
#define EWALD_HPP

#include <hpdmk.h>
#include <vesin.h>
#include <vector>
#include <array>
#include <complex>

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

        std::vector<double> k; // kx, ky, kz are the same since Lx = Ly = Lz

        VesinNeighborList neighbors;

        // std::vector<std::complex<double>> planewave_coeffs;
        // std::vector<std::complex<double>> interaction_matrix;

        double compute_energy(const double *q, const double (*r)[3]);
        double compute_potential(const double *q, const double (*r)[3], const double trg_x, const double trg_y, const double trg_z);
    };
} // namespace hpdmk

#endif