#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <sctl.hpp>
#include <cmath>
#include <complex>

#include <hpdmk.h>
#include <utils.hpp>

namespace hpdmk {

    template <typename Real>
    Real gaussian_window(Real k2, Real sigma);

    template <typename Real>
    Rank3Tensor<Real> gaussian_window_matrix(Real sigma, Real delta_k, Real n_k, Real k_max);

    template <typename Real>
    Real gaussian_difference_real(Real r, Real sigma_l, Real sigma_lp1);

    template <typename Real>
    Real gaussian_difference(Real k2, Real sigma_l, Real sigma_lp1);

    template <typename Real>
    Rank3Tensor<Real> gaussian_difference_matrix(Real sigma_l, Real sigma_lp1, Real delta_k, Real n_k, Real k_max);

    template <typename Real>
    Real gaussian_residual(Real dr, Real sigma_l);
}

#endif