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
    sctl::Vector<Real> window_matrix(PolyFun<Real> &fourier_poly, Real sigma, Real delta_k, Real n_k) {
        // interaction matrix for level 1, erf(r / sigma_2) / r
        int d = 2 * n_k + 1;
        Real k_x, k_y, k_z, k2;

        sctl::Vector<Real> window(d * d * (n_k + 1));

        for (int k = 0; k < n_k + 1; k++) {
            for (int j = 0; j < d; j++) {
                for (int i = 0; i < d; i++) {
                    k_x = (i - n_k) * delta_k;
                    k_y = (j - n_k) * delta_k;
                    k_z = (k - n_k) * delta_k;
                    k2 = k_x * k_x + k_y * k_y + k_z * k_z;
                    window[k * d * d + j * d + i] = window_kernel<Real>(k2, fourier_poly, sigma);
                }
            }
        }

        return window;
    }

    template <typename Real>
    sctl::Vector<Real> difference_matrix(PolyFun<Real> &fourier_poly, Real sigma_l, Real sigma_lp1, Real delta_k, Real n_k) {

        int d = 2 * n_k + 1;
        Real kx, ky, kz, k2;
        sctl::Vector<Real> D(d * d * (n_k + 1));

        for (int k = 0; k < n_k + 1; ++k) {
            for (int j = 0; j < d; ++j) {
                for (int i = 0; i < d; ++i) {
                    kx = (i - n_k) * delta_k;   
                    ky = (j - n_k) * delta_k;
                    kz = (k - n_k) * delta_k;
                    k2 = kx * kx + ky * ky + kz * kz;
                    D[k * d * d + j * d + i] = difference_kernel<Real>(k2, fourier_poly, sigma_l, sigma_lp1);
                }
            }
        }

        return D;
    }
}

#endif