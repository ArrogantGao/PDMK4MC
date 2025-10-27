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

#include <vecops.hpp>
#include <sctl.hpp>
#include <mpi.h>


namespace hpdmk {

    // window energy
    template <typename Real>
    Real HPDMKPtTree<Real>::eval_energy_window(){
        Real energy = 0;

        auto &outgoing_pw_root = outgoing_pw[root()];
        auto &window_mat = interaction_mat[0];

        const int dims = (2 * n_window + 1) * (2 * n_window + 1) * (n_window + 1);

        energy = vec_doudot<Real>(dims, &outgoing_pw_root[0], &window_mat[0]);
        energy *= 1 / (2 * std::pow(2*M_PI, 3)) * std::pow(delta_k[0], 3);

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

        Real energy_self, energy_oi, Q_i;

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

                    // energy_oi = C_l * tridot_nrn(dims, &outgoing_pw_i[0], &diff_mat[0], &incoming_pw_i[0]);
                    energy_oi = C_l * vec_tridot<Real, false, false>(dims, &outgoing_pw_i[0], &incoming_pw_i[0], &diff_mat[0]);
                    
                    Q_i = 0;
                    for (auto i_particle : node_particles[i_node]) {
                        Q_i += charge_sorted[i_particle] * charge_sorted[i_particle];
                    }
                    energy_self = Q_i * S_l;

                    energy += energy_oi - energy_self;
                }
            }
        }
        
        return energy;
    }

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
                    energy += charge_sorted[i_particle] * charge_sorted[j_particle] * residual_kernel<Real>(r_ij, C0, c, boxsize[i_depth]);
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
                    energy += charge_sorted[i_particle] * charge_sorted[j_particle] * residual_kernel<Real>(r_ij, C0, c, boxsize[i_depth]) / 2;
                }
            }
        }

        return energy;
    }
    
    template struct HPDMKPtTree<float>;
    template struct HPDMKPtTree<double>;
}