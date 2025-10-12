#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include <complex>

#include <hpdmk.h>
#include <sctl.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace hpdmk {
    
    inline bool isleaf(sctl::Tree<3>::NodeAttr node_attr) {
        return bool(node_attr.Leaf);
    }

    template <typename Real>
    struct ShiftMatrix {
        sctl::Vector<std::complex<Real>> sx;
        sctl::Vector<std::complex<Real>> sy;
        sctl::Vector<std::complex<Real>> sz;
        sctl::Vector<std::complex<Real>> conj_sx;
        sctl::Vector<std::complex<Real>> conj_sy;
        sctl::Vector<std::complex<Real>> conj_sz;
        sctl::Vector<std::complex<Real>> ones;

        ShiftMatrix() {}
        ShiftMatrix(sctl::Long n_k, Real delta_k, Real delta_x){
            sx.ReInit(2 * n_k + 1);
            sy.ReInit(2 * n_k + 1);
            sz.ReInit(2 * n_k + 1);
            conj_sx.ReInit(2 * n_k + 1);
            conj_sy.ReInit(2 * n_k + 1);
            conj_sz.ReInit(2 * n_k + 1);
            ones.ReInit(2 * n_k + 1);

            // since this is part of the precomputation, use exp directly
            for (int i = -n_k; i <= n_k; ++i) {
                sx[i + n_k] = std::exp(std::complex<Real>(0, i * delta_k * delta_x));
                sy[i + n_k] = std::exp(std::complex<Real>(0, i * delta_k * delta_x));
                sz[i + n_k] = std::exp(std::complex<Real>(0, i * delta_k * delta_x));
                conj_sx[i + n_k] = std::conj(sx[i + n_k]);
                conj_sy[i + n_k] = std::conj(sy[i + n_k]);
                conj_sz[i + n_k] = std::conj(sz[i + n_k]);
                ones[i + n_k] = std::complex<Real>(1, 0);
            }
        }

        sctl::Vector<std::complex<Real>>& select_sx(Real dx){
            const Real eps = 0.00001;
            if (dx > eps) {
                return sx;
            } else if (dx < -eps) {
                return conj_sx;
            } else {
                return ones;
            }
        }

        sctl::Vector<std::complex<Real>>& select_sy(Real dy){
            const Real eps = 0.00001;
            if (dy > eps) {
                return sy;
            } else if (dy < -eps) {
                return conj_sy;
            } else {
                return ones;
            }
        }

        sctl::Vector<std::complex<Real>>& select_sz(Real dz){
            const Real eps = 0.00001;
            if (dz > eps) {
                return sz;
            } else if (dz < -eps) {
                return conj_sz;
            } else {
                return ones;
            }
        }
    };

    template <typename Real>
    int periodic_shift(Real x_i, Real x_j, Real L, Real boxsize_i, Real boxsize_j) {
        if (x_i < boxsize_i && x_j > (L - boxsize_j)) {
            return 1; // i is at the left boundary, j is at the right boundary, shift x_i by L
        } else if (x_i > (L - boxsize_i) && x_j < boxsize_j) {
            return -1; // i is at the right boundary, j is at the left boundary, shift x_i by -L
        } else {
            return 0; // i and j are at the same side of the boundary
        }
    }

    template <typename Real>
    inline Real dist2(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2) {
        return std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2) + std::pow(z1 - z2, 2);
    }

    std::vector<std::vector<double>> read_particle_info(const std::string& filename);

    namespace EwaldConst {
        static constexpr double EWALD_F = 1.12837917;
        static constexpr double EWALD_P = 0.3275911;
        static constexpr double A1 = 0.254829592;
        static constexpr double A2 = -0.284496736;
        static constexpr double A3 = 1.421413741;
        static constexpr double A4 = -1.453152027;
        static constexpr double A5 = 1.061405429;
    }
    
    // the erfc approch from LAMMPS, with 6 absolute and 4 relative precision on [0, 3] (erfc(3) ~ 1e-4)
    // on this region, this function is 4~5 times faster than std::erfc (2~3ns vs 11ns)
    template <typename Real>
    inline Real my_erfc(Real x) {
        Real expm2 = std::exp(-x * x);
        Real t = 1.0 / (1.0 + EwaldConst::EWALD_P * std::abs(x));
        Real res = t * (EwaldConst::A1 + t * (EwaldConst::A2 + t * (EwaldConst::A3 + t * (EwaldConst::A4 + t * EwaldConst::A5)))) * expm2;
        return res;
    }

    template <typename Real>
    inline Real my_mod(Real x, Real L) {
        while (x < 0) {
            x += L;
        }
        while (x >= L) {
            x -= L;
        }
        return x;
    }

    inline void remove_particle(sctl::Vector<sctl::Long> &particles, sctl::Long i_particle) {
        // std::cout << "remove_particle: " << i_particle << std::endl;
        // std::cout << "initial: N = " << particles.Dim() << ", " << particles << std::endl;
        for (int i = 0; i < particles.Dim(); ++i) {
            if (particles[i] == i_particle) {
                for (int j = i; j < particles.Dim() - 1; ++j) {
                    particles[j] = particles[j + 1];
                }
                particles.ReInit(particles.Dim() - 1);
                break;
            }
        }
        // std::cout << "final: N = " << particles.Dim() << ", " << particles << std::endl;
    }
}

#endif