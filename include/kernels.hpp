#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <sctl.hpp>
#include <cmath>
#include <complex>

#include <hpdmk.h>
#include <utils.hpp>

namespace hpdmk {

    template <typename Real>
    Real gaussian_window(Real k2, Real sigma) {
        if (k2 == 0)
            return 0;
        else {
            Real window = 4 * M_PI * std::exp(- k2 * sigma * sigma / 4) / k2;
            return window;
        }
    }

    template <typename Real>
    CubicTensor<Real> gaussian_window_matrix(Real sigma, Real delta_k, Real n_k, Real k_max) {
        // interaction matrix for level 1, erf(r / sigma_2) / r
        int d = 2 * n_k + 1;
        CubicTensor<Real> window(d, sctl::Vector<Real>(std::pow(d, 3)));
        for (int i = 0; i < 2 * n_k + 1; ++i) {
            Real k_x = (i - n_k) * delta_k;
            for (int j = 0; j < 2 * n_k + 1; ++j) {
                Real k_y = (j - n_k) * delta_k;
                for (int k = 0; k < 2 * n_k + 1; ++k) {
                    Real k_z = (k - n_k) * delta_k;
                    Real k2 = k_x * k_x + k_y * k_y + k_z * k_z;
                    window.value(i, j, k) = gaussian_window<Real>(k2, sigma);
                }
            }
        }

        // std::cout << "minimum value: " << gaussian_window<Real>(k_max * k_max, sigma) << std::endl;
        return window;
    }

    template <typename Real>
    Real gaussian_difference_real(Real r, Real sigma_l, Real sigma_lp1) {
        Real difference = (std::erf(r / sigma_lp1) - std::erf(r / sigma_l)) / r;
        return difference;
    }

    template <typename Real>
    Real gaussian_difference(Real k2, Real sigma_l, Real sigma_lp1) {
        if (k2 == 0)
            return M_PI * (sigma_l * sigma_l - sigma_lp1 * sigma_lp1);
        else {
            Real window = 4 * M_PI * (std::exp(- k2 * sigma_lp1 * sigma_lp1 / 4) - std::exp(- k2 * sigma_l * sigma_l / 4)) / k2;
            return window;
        }
    }

    template <typename Real>
    CubicTensor<Real> gaussian_difference_matrix(Real sigma_l, Real sigma_lp1, Real delta_k, Real n_k, Real k_max) {
        int d = 2 * n_k + 1;
        CubicTensor<Real> D(d, sctl::Vector<Real>(std::pow(d, 3)));
        for (int i = 0; i < 2 * n_k + 1; ++i) {
            Real k_x = (i - n_k) * delta_k;
            for (int j = 0; j < 2 * n_k + 1; ++j) {
                Real k_y = (j - n_k) * delta_k;
                for (int k = 0; k < 2 * n_k + 1; ++k) {
                    Real k_z = (k - n_k) * delta_k;

                    Real k2 = k_x * k_x + k_y * k_y + k_z * k_z;
                    D.value(i, j, k) = gaussian_difference<Real>(k2, sigma_l, sigma_lp1);
                }
            }
        }

        // std::cout << "minimum value: " << gaussian_difference<Real>(k_max * k_max, sigma_l, sigma_lp1) << std::endl;

        return D;
    }

    template <typename Real>
    Real gaussian_residual(Real dr, Real sigma_l) {
        Real residual = std::erfc(dr / sigma_l) / dr;
        return residual;
    }
}

#endif