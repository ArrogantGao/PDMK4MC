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

    // window energy
    template <typename Real>
    Real HPDMKPtTree<Real>::eval_energy_window(){
        Real energy = 0;

        auto &outgoing_pw_root = outgoing_pw[root()];
        auto &window_mat = interaction_mat[0];

        int dims = (2 * n_window + 1) * (2 * n_window + 1) * (n_window + 1);
        energy = tridot_nrc(dims, &outgoing_pw_root[0], &window_mat[0], &outgoing_pw_root[0]) / (2 * std::pow(2*M_PI, 3)) * std::pow(delta_k[0], 3);

        // self energy
        Real r_c = boxsize[2];
        Real self_energy = Q * prolate0_eval(c, 0) / (2 * r_c * C0);
        energy -= self_energy;

        return energy;
    }

    // difference energy
    template <typename Real>
    Real HPDMKPtTree<Real>::eval_energy_diff(){
        Real energy = 0;
        auto &node_attr = this->GetNodeAttr();
        int dims = (2 * n_diff + 1) * (2 * n_diff + 1) * (n_diff + 1);

        Real energy_oo, energy_self, energy_oi, Q_i;

        for (int l = 2; l < max_depth - 1; ++l) {
            auto& diff_mat = interaction_mat[l];
            Real boxsize_l = boxsize[l];
            Real boxsize_lp1 = boxsize[l + 1];
            Real delta_kl = delta_k[l];

            Real C_l = 1 / (2 * std::pow(2*M_PI, 3)) * std::pow(delta_kl, 3);
            Real S_l = prolate0_eval(c, 0) * (1 / (2 * boxsize_lp1 * C0) - 1 / (2 * boxsize_l * C0));

            for (sctl::Long i_node : level_indices[l]) {
                if (!isleaf(node_attr[i_node]) && node_particles[i_node].Dim() > 0) {
                    auto& outgoing_pw_i = outgoing_pw[i_node];
                    auto& incoming_pw_i = incoming_pw[i_node];

                    energy_oo = C_l * tridot_nrc(dims, &outgoing_pw_i[0], &diff_mat[0], &outgoing_pw_i[0]);
                    energy_oi = C_l * tridot_nrn(dims, &outgoing_pw_i[0], &diff_mat[0], &incoming_pw_i[0]);
                    
                    Q_i = 0;
                    for (auto i_particle : node_particles[i_node]) {
                        Q_i += charge_sorted[i_particle] * charge_sorted[i_particle];
                    }
                    energy_self = Q_i * S_l;

                    energy += energy_oo + energy_oi - energy_self;
                }
            }
        }
        
        return energy;
    }

    // W + D_0 + D_1
    // template <typename Real>
    // Real HPDMKPtTree<Real>::window_energy() {
    //     Real energy = 0;

    //     auto &root_coeffs = plane_wave_coeffs[root()];
    //     auto &window = interaction_matrices[0];

    //     assert(root_coeffs.Dim() == window.Dim());

    //     // #pragma omp parallel for reduction(+:energy)
    //     for (int i = 0; i < root_coeffs.Dim(); ++i) {
    //         if (window[i] != 0) {
    //             energy += std::real(root_coeffs[i] * window[i] * std::conj(root_coeffs[i]));
    //         }
    //     }

    //     energy *= 1 / (2 * std::pow(2*M_PI, 3)) * std::pow(delta_k[0], 3);

    //     // self energy
    //     Real r_c = boxsize[2];
    //     Real self_energy = Q * prolate0_eval(c, 0) / (2 * r_c * C0);
    //     energy -= self_energy;

    //     return energy;
    // }

    // // D_l for non-leaf nodes with depth >= 2
    // template <typename Real>
    // Real HPDMKPtTree<Real>::difference_energy() {
    //     Real energy = 0;
    //     auto &node_attr = this->GetNodeAttr();

    //     for (int l = 2; l < max_depth - 1; ++l) {
    //         for (sctl::Long i_node : level_indices[l]) {
    //             // non-leaf nodes, with particles inside
    //             if (!isleaf(node_attr[i_node]) && node_particles[i_node].Dim() > 0) {
    //                 energy += difference_energy_i(l, i_node);

    //                 // non-leaf nodes should have 26 colleague neighbors
    //                 assert(neighbors[i_node].colleague.Dim() == 26);
    //                 for (auto i_nbr : neighbors[i_node].colleague) {
    //                     if (node_particles[i_nbr].Dim() > 0) {
    //                         energy += difference_energy_ij(l, i_node, i_nbr);
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     return energy;
    // }

    // template <typename Real>
    // Real HPDMKPtTree<Real>::difference_energy_i(int i_depth, sctl::Long i_node) {
    //     Real energy = 0;

    //     auto &D_l = interaction_matrices[i_depth];
    //     auto &node_coeffs = plane_wave_coeffs[i_node];

    //     // #pragma omp parallel for reduction(+:energy)
    //     for (int i = 0; i < D_l.Dim(); ++i) {
    //         // if (D_l[i] != 0) {
    //             energy += std::real(node_coeffs[i] * D_l[i] * std::conj(node_coeffs[i]));
    //         // }
    //     }

    //     energy *= 1 / (2 * std::pow(2*M_PI, 3)) * std::pow(delta_k[i_depth], 3);

    //     // self energy
    //     Real boxsize_l = boxsize[i_depth];
    //     Real boxsize_lp1 = boxsize[i_depth + 1];
    //     Real Q_node = 0;
    //     for (auto i_particle : node_particles[i_node]) {
    //         Q_node += charge_sorted[i_particle] * charge_sorted[i_particle];
    //     }
    //     Real self_energy = Q_node * prolate0_eval(c, 0) * (1 / (2 * boxsize_lp1 * C0) - 1 / (2 * boxsize_l * C0));
    //     energy -= self_energy;

    //     return energy;
    // }

    // template <typename Real>
    // Real HPDMKPtTree<Real>::difference_energy_ij(int i_depth, sctl::Long i_node, sctl::Long j_node) {
    //     Real energy = 0;
        
    //     auto shift_ij = node_shift(i_node, j_node);
    //     Real delta_ki = delta_k[i_depth];
    //     int n_ki = n_k[i_depth];

    //     auto &D_l = interaction_matrices[i_depth];
    //     auto &node_coeffs_i = plane_wave_coeffs[i_node];
    //     auto &node_coeffs_j = plane_wave_coeffs[j_node];

    //     std::complex<Real> exp_ik_shiftx = std::exp(std::complex<Real>(0, 1) * delta_ki * shift_ij[0]);
    //     std::complex<Real> exp_ik_shifty = std::exp(std::complex<Real>(0, 1) * delta_ki * shift_ij[1]);
    //     std::complex<Real> exp_ik_shiftz = std::exp(std::complex<Real>(0, 1) * delta_ki * shift_ij[2]);

    //     int d = 2 * n_ki + 1;

    //     // #pragma omp parallel for
    //     for (int i = 0; i < d; ++i) {
    //         int n = i - n_ki;
    //         kx_cache[i] = std::pow(exp_ik_shiftx, n);
    //         ky_cache[i] = std::pow(exp_ik_shifty, n);
    //     }

    //     for (int i = 0; i < n_ki + 1; ++i) {
    //         kz_cache[i] = std::pow(exp_ik_shiftz, i);
    //     }

    //     // #pragma omp parallel for reduction(+:energy)
    //     for (int i = 0; i < d; ++i) {
    //         auto t1 = kx_cache[i];
    //         for (int j = 0; j < d; ++j) {
    //             auto t2 = t1 * ky_cache[j];
    //             for (int k = 0; k < n_ki + 1; ++k) {
    //                 if (D_l(i, j, k) != 0) {
    //                     energy += std::real(node_coeffs_i(i, j, k) * D_l(i, j, k) * std::conj(node_coeffs_j(i, j, k)) * t2 * kz_cache[k]);
    //                 }
    //             }
    //         }
    //     }

    //     energy *= 1 / (2 * std::pow(2*M_PI, 3)) * std::pow(delta_ki, 3);

    //     return energy;
    // }

    // energy of the residual term, for leaf nodes
    template <typename Real>
    Real HPDMKPtTree<Real>::eval_energy_res() {
        Real energy = 0;

        auto &node_attr = this->GetNodeAttr();

        for (int l = 2; l < max_depth; ++l) {
            for (sctl::Long i_node : level_indices[l]) {
                if (isleaf(node_attr[i_node]) && node_particles[i_node].Dim() > 0) {

                    // self interaction
                    energy += eval_energy_res_i(l, i_node);

                    // coarse-grained neighbors
                    for (auto i_nbr : neighbors[i_node].coarsegrain) {
                        if (node_particles[i_nbr].Dim() > 0) {
                            energy += eval_energy_res_ij(l, i_node, i_nbr);
                        }
                    }

                    // colleague neighbors
                    for (auto i_nbr : neighbors[i_node].colleague) {
                        if (node_particles[i_nbr].Dim() > 0) {
                            energy += eval_energy_res_ij(l, i_node, i_nbr);
                        }
                    }
                }
            }
        }

        return energy;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::eval_energy_res_i(int i_depth, sctl::Long i_node) {
        Real energy = 0;

        for (int i = 0; i < node_particles[i_node].Dim() - 1; ++i) {
            int i_particle = node_particles[i_node][i];

            Real xi = r_src_sorted[i_particle * 3];
            Real yi = r_src_sorted[i_particle * 3 + 1];
            Real zi = r_src_sorted[i_particle * 3 + 2];

            for (int j = i + 1; j < node_particles[i_node].Dim(); ++j) {
                int j_particle = node_particles[i_node][j];

                Real xj = r_src_sorted[j_particle * 3];
                Real yj = r_src_sorted[j_particle * 3 + 1];
                Real zj = r_src_sorted[j_particle * 3 + 2];
                Real r_ij = std::sqrt(dist2(xi, yi, zi, xj, yj, zj));
                if (r_ij <= boxsize[i_depth]) {
                    energy += charge_sorted[i_particle] * charge_sorted[j_particle] * residual_kernel<Real>(r_ij, real_poly, boxsize[i_depth]);
                }
            }
        }

        return energy;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::eval_energy_res_ij(int i_depth, sctl::Long i_node, sctl::Long j_node) {
        Real energy = 0;

        auto shift_ij = node_shift(i_node, j_node);

        Real center_xi = centers[i_node * 3];
        Real center_yi = centers[i_node * 3 + 1];
        Real center_zi = centers[i_node * 3 + 2];

        Real center_xj = centers[j_node * 3];
        Real center_yj = centers[j_node * 3 + 1];
        Real center_zj = centers[j_node * 3 + 2];

        for (auto i_particle : node_particles[i_node]) {
            Real xi = r_src_sorted[i_particle * 3] - center_xi - shift_ij[0];
            Real yi = r_src_sorted[i_particle * 3 + 1] - center_yi - shift_ij[1];
            Real zi = r_src_sorted[i_particle * 3 + 2] - center_zi - shift_ij[2];
            for (auto j_particle : node_particles[j_node]) {
                Real xj = r_src_sorted[j_particle * 3] - center_xj;
                Real yj = r_src_sorted[j_particle * 3 + 1] - center_yj;
                Real zj = r_src_sorted[j_particle * 3 + 2] - center_zj;

                Real r_ij = std::sqrt(dist2(xi, yi, zi, xj, yj, zj));
                if (r_ij <= boxsize[i_depth]) {
                    energy += charge_sorted[i_particle] * charge_sorted[j_particle] * residual_kernel<Real>(r_ij, real_poly, boxsize[i_depth]) / 2;
                }
            }
        }

        return energy;
    }
    
    template struct HPDMKPtTree<float>;
    template struct HPDMKPtTree<double>;
}