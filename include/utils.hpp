#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include <complex>

#include <hpdmk.h>
#include <sctl.hpp>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

#include <shiftmat.hpp>
#include <vecops.hpp>
#include <pswf.hpp>

namespace hpdmk {

    inline void get_prolate_params(int n_digits, double& c, double& lambda, double& C0, int& n_diff, std::vector<double>& coefs) {
        if (n_digits == 3) {
            c = 7.2462000846862793;
            n_diff = 6;
            // delta_k0 = 0.6620 * M_PI;

            coefs = {1.627823522210361e-01,  -4.553645597616490e-01, 4.171687104204163e-01, -7.073638602709915e-02, -8.957845614474928e-02, 2.617986644718201e-02, 9.633427876507601e-03};
            std::reverse(coefs.begin(), coefs.end());
        } else if (n_digits == 6) {
            c = 13.739999771118164;
            n_diff = 12;
            // delta_k0 = 0.6686 * M_PI;
            coefs = {5.482525801351582e-02,  -2.616592110444692e-01, 4.862652666337138e-01, -3.894296348642919e-01, 1.638587821812791e-02,  1.870328434198821e-01,-8.714171086568978e-02, -3.927020727017803e-02, 3.728187607052319e-02, 3.153734425831139e-03,  -8.651313377285847e-03, 1.725110090795567e-04, 1.034762385284044e-03};
            std::reverse(coefs.begin(), coefs.end());
        } else if (n_digits == 9) {
            c = 20.736000061035156;
            n_diff = 19;
            // delta_k0 = 0.6625 * M_PI;
            coefs = {1.835718730962269e-02,  -1.258015846164503e-01, 3.609487248584408e-01,  -5.314579651112283e-01, 3.447559412892380e-01,  9.664692318551721e-02,  -3.124274531849053e-01, 1.322460720579388e-01, 9.773007866584822e-02,  -1.021958831082768e-01, -3.812847450976566e-03, 3.858117355875043e-02, -8.728545924521301e-03, -9.401196355382909e-03, 4.024549377076924e-03,  1.512806105865091e-03, -9.576734877247042e-04, -1.303457547418901e-04, 1.100385844683190e-04};
            std::reverse(coefs.begin(), coefs.end());
        } else if (n_digits == 12) {
            c = 27.870000839233398;
            n_diff = 26;
            // delta_k0 = 0.6677 * M_PI;
            coefs = {6.262472576363448e-03,  -5.605742936112479e-02, 2.185890864792949e-01,  -4.717350304955679e-01, 5.669680214206270e-01,  -2.511606878849214e-01, -2.744523658778361e-01, 4.582527599363415e-01, -1.397724810121539e-01, -2.131762135835757e-01, 1.995489373508990e-01,  1.793390341864239e-02, -1.035055132403432e-01, 3.035606831075176e-02,  3.153931762550532e-02,  -2.033178627450288e-02, -5.406682731236552e-03, 7.543645573618463e-03,  1.437788047407851e-05,  -1.928370882351732e-03, 2.891658777328665e-04,  3.332996162099811e-04,  -8.397699195938912e-05, -3.015837377517983e-05, 9.640642701924662e-06};
            std::reverse(coefs.begin(), coefs.end());
        } else {
            throw std::runtime_error("digits is not supported"); // redundant, but just in case
        }

        lambda = prolate0_lambda(c);
        C0 = prolate0_int_eval(c, 1.0);
    }
    
    inline bool isleaf(sctl::Tree<3>::NodeAttr node_attr) {
        return bool(node_attr.Leaf);
    }

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

    template <typename Real>
    void random_init(sctl::Vector<Real>& vec, Real min, Real max) {
        std::mt19937 generator;
        std::uniform_real_distribution<Real> distribution(min, max);
        for (int i = 0; i < vec.Dim(); ++i) {
            vec[i] = distribution(generator);
        }
    }

    template <typename Real>
    void unify_charge(sctl::Vector<Real>& charge) {
        Real total_charge = std::accumulate(charge.begin(), charge.end(), 0.0);
        charge -= total_charge / charge.Dim();
        assert(std::abs(std::accumulate(charge.begin(), charge.end(), 0.0)) < 1e-12);
    }
}

#endif