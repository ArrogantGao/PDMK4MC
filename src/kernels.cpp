#include <sctl.hpp>
#include <cmath>
#include <complex>

#include <hpdmk.h>
#include <utils.hpp>
#include <kernels.hpp>

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
    Rank3Tensor<Real> gaussian_window_matrix(Real sigma, Real delta_k, Real n_k, Real k_max) {
        // interaction matrix for level 1, erf(r / sigma_2) / r
        int d = 2 * n_k + 1;
        Real k_max_2 = k_max * k_max;
        Rank3Tensor<Real> window(d, d, n_k + 1);
        for (int i = 0; i < 2 * n_k + 1; ++i) {
            Real k_x = (i - n_k) * delta_k;
            for (int j = 0; j < 2 * n_k + 1; ++j) {
                Real k_y = (j - n_k) * delta_k;

                // only consider kz >= 0 due to symmetry
                // if kz > 0, double the value
                for (int k = 0; k < n_k + 1; ++k) {
                    Real k_z = k * delta_k;
                    Real k2 = k_x * k_x + k_y * k_y + k_z * k_z;
                    if (k2 <= k_max_2) {
                        if (k == 0) {
                            window(i, j, k) = gaussian_window<Real>(k2, sigma);
                        } else {
                            window(i, j, k) = 2 * gaussian_window<Real>(k2, sigma);
                        }
                    } else {
                        window(i, j, k) = 0;
                    }
                }
            }
        }

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
    Rank3Tensor<Real> gaussian_difference_matrix(Real sigma_l, Real sigma_lp1, Real delta_k, Real n_k, Real k_max) {

        int d = 2 * n_k + 1;
        Real k_max_2 = k_max * k_max;
        Real kx, ky, kz, k2;
        Rank3Tensor<Real> D(d, d, n_k + 1);

        for (int i = 0; i < 2 * n_k + 1; ++i) {
            for (int j = 0; j < 2 * n_k + 1; ++j) {
                for (int k = 0; k < n_k + 1; ++k) {

                    kx = (i - n_k) * delta_k;
                    ky = (j - n_k) * delta_k;
                    kz = k * delta_k;
                    k2 = kx * kx + ky * ky + kz * kz;

                    if (k2 <= k_max_2) {
                        if (k == 0) {
                            D(i, j, k) = gaussian_difference<Real>(k2, sigma_l, sigma_lp1);
                        } else {
                            D(i, j, k) = 2 * gaussian_difference<Real>(k2, sigma_l, sigma_lp1);
                        }
                    } else {
                        D(i, j, k) = 0;
                    }
                }
            }
        }

        return D;
    }

    template <typename Real>
    Real gaussian_residual(Real dr, Real sigma_l) {
        Real residual = std::erfc(dr / sigma_l) / dr;
        return residual;
    }

    template Rank3Tensor<double> hpdmk::gaussian_window_matrix<double>(double, double, double, double);
    template Rank3Tensor<float> hpdmk::gaussian_window_matrix<float>(float, float, float, float);
    template Rank3Tensor<double> hpdmk::gaussian_difference_matrix<double>(double, double, double, double, double);
    template Rank3Tensor<float> hpdmk::gaussian_difference_matrix<float>(float, float, float, float, float);
    template double hpdmk::gaussian_residual<double>(double, double);
    template float hpdmk::gaussian_residual<float>(float, float);
    template double hpdmk::gaussian_difference_real<double>(double, double, double);
    template float hpdmk::gaussian_difference_real<float>(float, float, float);
    template double hpdmk::gaussian_difference<double>(double, double, double);
    template float hpdmk::gaussian_difference<float>(float, float, float);
    template double hpdmk::gaussian_window<double>(double, double);
    template float hpdmk::gaussian_window<float>(float, float);
}