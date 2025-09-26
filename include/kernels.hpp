#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <sctl.hpp>
#include <cmath>
#include <complex>

#include <hpdmk.h>
#include <utils.hpp>
#include <pswf.hpp>


namespace hpdmk {

    template <typename Real>
    inline Real window_kernel(Real k2, PolyFun<Real> &fourier_poly, Real sigma){
        if (k2 == 0)
            return 0;
        else {
            Real k = sqrt(k2);
            Real k_scaled = k * sigma;
            Real window = fourier_poly.eval(k_scaled) / k2;
            return window;
        }
    }

    template <typename Real>
    inline Real difference_kernel(Real k2, PolyFun<Real> &fourier_poly, Real sigma_l, Real sigma_lp1){
        Real k = sqrt(k2);
        Real k_scaled_l = k * sigma_l;
        Real k_scaled_lp1 = k * sigma_lp1;

        int order = fourier_poly.order;
        double diff = fourier_poly.coeffs[order - 3];

        Real window;
        if (k2 == 0){
            window = diff * (sigma_lp1 * sigma_lp1 - sigma_l * sigma_l);
        } else {
            window = (fourier_poly.eval(k_scaled_lp1) - fourier_poly.eval(k_scaled_l)) / k2;
        }

        return window;
    }

    template <typename Real>
    inline Real difference_kernel_direct(Real r, PolyFun<Real> &real_poly, Real cutoff_l, Real cutoff_lp1){
        Real difference = - (real_poly.eval(r / cutoff_lp1) - real_poly.eval(r / cutoff_l)) / r;
        return difference;
    }
 
    template <typename Real>
    inline Real residual_kernel(Real r, PolyFun<Real> &real_poly, Real cutoff){
        if (r == 0)
            return 0;
        else {
            Real residual = real_poly.eval(r / cutoff) / r;
            return residual;
        }
    }

    template <typename Real>
    Rank3Tensor<Real> window_matrix(PolyFun<Real> &fourier_poly, Real sigma, Real delta_k, Real n_k, Real k_max) {
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
                            window(i, j, k) = window_kernel<Real>(k2, fourier_poly, sigma);
                        } else {
                            window(i, j, k) = 2 * window_kernel<Real>(k2, fourier_poly, sigma);
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
    Rank3Tensor<Real> difference_matrix(PolyFun<Real> &fourier_poly, Real sigma_l, Real sigma_lp1, Real delta_k, Real n_k, Real k_max) {

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
                            D(i, j, k) = difference_kernel<Real>(k2, fourier_poly, sigma_l, sigma_lp1);
                        } else {
                            D(i, j, k) = 2 * difference_kernel<Real>(k2, fourier_poly, sigma_l, sigma_lp1);
                        }
                    } else {
                        D(i, j, k) = 0;
                    }
                }
            }
        }

        return D;
    }
}

#endif