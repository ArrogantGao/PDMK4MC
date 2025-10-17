// direct energy calculation, for testing
#include <hpdmk.h>
#include <tree.hpp>
#include <kernels.hpp>
#include <utils.hpp>
#include <pswf.hpp>

#include <vector>
#include <array>
#include <cmath>
#include <complex>
#include <algorithm>

#include <sctl.hpp>
#include <mpi.h>


namespace hpdmk {

    template <typename Real>
    Real HPDMKPtTree<Real>::eval_energy_window_direct() {
        Real energy = 0;

        Real delta_k0 = delta_k[0];
        int n_k0 = n_window;
        Real sigma = sigmas[2];

        Real kx, ky, kz, k2;
        Real x, y, z, q;
        std::complex<Real> rho_k;
        for (int i = 0; i < 2 * n_k0 + 1; ++i) {
            kx = (i - n_k0) * delta_k0;
            for (int j = 0; j < 2 * n_k0 + 1; ++j) {
                ky = (j - n_k0) * delta_k0;
                for (int k = 0; k < 2 * n_k0 + 1; ++k) {
                    kz = (k - n_k0) * delta_k0;
                    k2 = kx * kx + ky * ky + kz * kz;
                    rho_k = 0;
                    for (int n = 0; n < charge_sorted.Dim(); ++n) {
                        x = r_src_sorted[n * 3];
                        y = r_src_sorted[n * 3 + 1];
                        z = r_src_sorted[n * 3 + 2];
                        q = charge_sorted[n];

                        rho_k += std::exp(std::complex<Real>(0, kx * x + ky * y + kz * z)) * q;
                    }
                    energy += std::real(rho_k * std::conj(rho_k)) * window_kernel<Real>(k2, fourier_poly, sigma);
                }
            }
        }

        energy *= 1 / (2 * std::pow(2*M_PI, 3)) * std::pow(delta_k0, 3);

        // zeroth order term
        Real self_energy = Q * prolate0_eval(c, 0) / (2 * boxsize[2] * C0);
        energy -= self_energy;

        return energy;
    }

    // template <typename Real>
    // Real HPDMKPtTree<Real>::difference_energy_direct_i(int i_depth, sctl::Long i_node) {
    //     Real energy = 0;

    //     for (int i = 0; i < node_particles[i_node].Dim() - 1; ++i) {
    //         for (int j = i + 1; j < node_particles[i_node].Dim(); ++j) {
    //             int i_particle = node_particles[i_node][i];
    //             int j_particle = node_particles[i_node][j];
    //             Real xi = r_src_sorted[i_particle * 3];
    //             Real yi = r_src_sorted[i_particle * 3 + 1];
    //             Real zi = r_src_sorted[i_particle * 3 + 2];
    //             Real xj = r_src_sorted[j_particle * 3];
    //             Real yj = r_src_sorted[j_particle * 3 + 1];
    //             Real zj = r_src_sorted[j_particle * 3 + 2];
    //             Real r_ij = std::sqrt(dist2(xi, yi, zi, xj, yj, zj));
    //             energy += charge_sorted[i_particle] * charge_sorted[j_particle] * difference_kernel_direct<Real>(r_ij, real_poly, boxsize[i_depth], boxsize[i_depth + 1]);
    //         }
    //     }
    //     return energy;
    // }

    template <typename Real>
    Real HPDMKPtTree<Real>::eval_energy_diff_direct() {
        Real energy = 0;

        auto &node_attr = this->GetNodeAttr();

        for (int l = 2; l < max_depth; ++l) {
            for (sctl::Long i_node : level_indices[l]) {
                if (!isleaf(node_attr[i_node]) && node_particles[i_node].Dim() > 0) {
                    for (auto i_particle : node_particles[i_node]) {
                        Real xi = r_src_sorted[i_particle * 3];
                        Real yi = r_src_sorted[i_particle * 3 + 1];
                        Real zi = r_src_sorted[i_particle * 3 + 2];
                        for (int j = 0; j < charge_sorted.Dim(); ++j) {
                            if (j == i_particle) continue;
                            for (int mx = -1; mx <= 1; mx++) {
                                for (int my = -1; my <= 1; my++) {
                                    for (int mz = -1; mz <= 1; mz++) {
                                        Real xj = r_src_sorted[j * 3] + mx * L;
                                        Real yj = r_src_sorted[j * 3 + 1] + my * L;
                                        Real zj = r_src_sorted[j * 3 + 2] + mz * L;
                                        Real r_ij = std::sqrt(dist2(xi, yi, zi, xj, yj, zj));
                                        energy += charge_sorted[i_particle] * charge_sorted[j] * difference_kernel_direct<Real>(r_ij, real_poly, boxsize[l], boxsize[l + 1]) / 2;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return energy;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::eval_energy_res_direct() {
        Real energy = 0;

        auto &node_attr = this->GetNodeAttr();

        // for the l-th level
        for (int l = 2; l < max_depth; ++l) {
            for (sctl::Long i_node : level_indices[l]) {
                if (isleaf(node_attr[i_node]) && node_particles[i_node].Dim() > 0) {
                    for (auto i_particle : node_particles[i_node]) {
                        Real xi = r_src_sorted[i_particle * 3];
                        Real yi = r_src_sorted[i_particle * 3 + 1];
                        Real zi = r_src_sorted[i_particle * 3 + 2];
                        for (int j = 0; j < charge_sorted.Dim(); ++j) {
                            if (j == i_particle) continue;
                            for (int mx = -1; mx <= 1; mx++) {
                                for (int my = -1; my <= 1; my++) {
                                    for (int mz = -1; mz <= 1; mz++) {
                                        Real xj = r_src_sorted[j * 3] + mx * L;
                                        Real yj = r_src_sorted[j * 3 + 1] + my * L;
                                        Real zj = r_src_sorted[j * 3 + 2] + mz * L;
                                        Real r_ij = std::sqrt(dist2(xi, yi, zi, xj, yj, zj));
                                        energy += charge_sorted[i_particle] * charge_sorted[j] * residual_kernel<Real>(r_ij, real_poly, boxsize[l]) / 2;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return energy;
    }

    // template <typename Real>
    // Real HPDMKPtTree<Real>::residual_energy_shift_direct(int i_depth, Real x, Real y, Real z, Real q) {
    //     Real E = 0;
        
    //     for (int i = 0; i < charge_sorted.Dim(); ++i) {
    //         for (int mx = -1; mx <= 1; mx++) {
    //             for (int my = -1; my <= 1; my++) {
    //                 for (int mz = -1; mz <= 1; mz++) {
    //                     Real xi = r_src_sorted[i * 3] + mx * L;
    //                     Real yi = r_src_sorted[i * 3 + 1] + my * L;
    //                     Real zi = r_src_sorted[i * 3 + 2] + mz * L;
    //                     Real r_ij = std::sqrt(dist2(xi, yi, zi, x, y, z));
    //                     E += charge_sorted[i] * residual_kernel<Real>(r_ij, real_poly, boxsize[i_depth]);
    //                 }
    //             }
    //         }
    //     }

    //     return q * E;
    // }

    // template <typename Real>
    // Real HPDMKPtTree<Real>::difference_energy_shift_direct(int i_depth, int i_particle, Real x, Real y, Real z) {
    //     Real E = 0;
        
    //     auto &node_attr = this->GetNodeAttr();

    //     for (int l = 2; l < i_depth; ++l) {
    //         Real E_l = 0;
    //         for (int j = 0; j < charge_sorted.Dim(); ++j) {
    //             if (j == i_particle) continue;
    //             for (int mx = -1; mx <= 1; mx++) {
    //                 for (int my = -1; my <= 1; my++) {
    //                     for (int mz = -1; mz <= 1; mz++) {
    //                         Real xj = r_src_sorted[j * 3] + mx * L;
    //                         Real yj = r_src_sorted[j * 3 + 1] + my * L;
    //                         Real zj = r_src_sorted[j * 3 + 2] + mz * L;
    //                         Real r_ij = std::sqrt(dist2(x, y, z, xj, yj, zj));
    //                         E_l += charge_sorted[j] * difference_kernel_direct<Real>(r_ij, real_poly, boxsize[l], boxsize[l + 1]);
    //                     }
    //                 }
    //             }
    //         }
    //         std::cout << "l: " << l << ", E_l: " << E_l << std::endl;
    //         E += E_l;
    //     }
        
    //     return E;
    // }

    template struct HPDMKPtTree<float>;
    template struct HPDMKPtTree<double>;
}