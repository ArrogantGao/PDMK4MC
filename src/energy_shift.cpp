#include <hpdmk.h>
#include <tree.hpp>
#include <kernels.hpp>
#include <utils.hpp>
#include <pswf.hpp>
#include <direct_eval.hpp>

#include <vector>
#include <array>
#include <cmath>
#include <complex>
#include <algorithm>

#include <sctl.hpp>
#include <mpi.h>

namespace hpdmk {
    template <typename Real>
    Real HPDMKPtTree<Real>::eval_shift_energy(sctl::Long i_unsorted, Real dx, Real dy, Real dz) {

        sctl::Long i_particle = indices_invmap[i_unsorted];

        Real q = charge_sorted[i_particle];
        Real x_o = r_src_sorted[i_particle * 3];
        Real y_o = r_src_sorted[i_particle * 3 + 1];
        Real z_o = r_src_sorted[i_particle * 3 + 2];

        // std::cout << "original xyz in: " << x_o << ", " << y_o << ", " << z_o << std::endl;

        Real x_t = my_mod(x_o + dx, L);
        Real y_t = my_mod(y_o + dy, L);
        Real z_t = my_mod(z_o + dz, L);
        Real dr = std::sqrt(dx * dx + dy * dy + dz * dz);
        for (int mx = -1; mx <= 1; mx++) {
            for (int my = -1; my <= 1; my++) {
                for (int mz = -1; mz <= 1; mz++) {
                    Real t = std::sqrt(dist2(x_t, y_t, z_t, x_o + mx * L, y_o + my * L, z_o + mz * L));
                    dr = std::min(dr, t);
                }
            }
        }

        // std::cout << "target xyz in: " << x_t << ", " << y_t << ", " << z_t << std::endl;

        locate_particle(path_to_target, x_t, y_t, z_t);
        locate_particle(path_to_origin, x_o, y_o, z_o);

        // std::cout << "tree, origin, target depth: " << this->n_levels() << ", " << path_to_origin.Dim() - 1 << ", " << path_to_target.Dim() - 1 << std::endl;

        form_outgoing_pw_single(outgoing_pw_origin, path_to_origin, x_o, y_o, z_o, q);
        form_outgoing_pw_single(outgoing_pw_target, path_to_target, x_t, y_t, z_t, q);
        
        Real dE_window = eval_shift_energy_window();
        Real dE_difference = eval_shift_energy_diff(i_particle);
        Real dE_residual_target = eval_shift_energy_res_vec(i_particle, path_to_target, x_t, y_t, z_t, q);
        Real dE_residual_origin = eval_shift_energy_res_vec(i_particle, path_to_origin, x_o, y_o, z_o, q);

        Real dE_self = residual_kernel<Real>(dr, real_poly, boxsize[2]) * q * q;

        // Real dE_difference_direct = eval_shift_energy_diff_direct(i_particle, x_t, y_t, z_t, x_o, y_o, z_o);
        // std::cout << "diff, diff_ref, err: " << dE_difference << ", " << dE_difference_direct << ", " << dE_difference - dE_difference_direct << std::endl;

        // Real dE_rest_ref = eval_shift_energy_res(i_particle, path_to_target, x_t, y_t, z_t, q);
        // Real dE_reso_ref = eval_shift_energy_res(i_particle, path_to_origin, x_o, y_o, z_o, q);
        // std::cout << "rest_err, reso_err: " << dE_rest_ref - dE_residual_target << ", " << dE_reso_ref - dE_residual_origin << std::endl;

        Real dE_shift = dE_window + dE_difference + dE_residual_target - dE_residual_origin - dE_self;

        return dE_shift;
    }
    
    template <typename Real>
    Real HPDMKPtTree<Real>::eval_shift_energy_window() {
        Real dE_window = 0;

        auto &target_root_coeffs = outgoing_pw_target[0];
        auto &origin_root_coeffs = outgoing_pw_origin[0];
        auto &outgoing_root_coeffs = outgoing_pw[root()];
        auto &window = interaction_mat[0];


        const int dims = (2 * n_window + 1) * (2 * n_window + 1) * (n_window + 1);
        dE_window = 2.0 * vec_shift_window<Real>(dims, &target_root_coeffs[0], &origin_root_coeffs[0], &outgoing_root_coeffs[0], &window[0]);
        // for (int i = 0; i < dims; ++i) {
        //     dE_window += 2.0 * std::real((target_root_coeffs[i] - origin_root_coeffs[i]) * std::conj(outgoing_root_coeffs[i] - origin_root_coeffs[i])) * window[i];
        // }

        dE_window *= 1 / (2 * std::pow(2*M_PI, 3)) * std::pow(delta_k[0], 3);
        return dE_window;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::eval_shift_energy_diff(sctl::Long i_particle) {
        Real dE_difference_t = 0;
        Real dE_difference_o = 0;

        auto &node_mid = this->GetNodeMID();
        int n_kl = n_diff;
        int d = 2 * n_kl + 1;
        int dims = (2 * n_kl + 1) * (2 * n_kl + 1) * (n_kl + 1);

        for (int l = 2; l < path_to_origin.Dim() - 1; ++l) {
            Real dE_o = 0;

            auto &origin_coeff = outgoing_pw_origin[l];
            auto &D_l = interaction_mat[l];

            sctl::Long i_node = path_to_origin[l];
            assert(node_mid[i_node].Depth() == l);

            Real delta_kl = delta_k[l];

            auto &node_coeffs = incoming_pw[i_node];
            // for (int i = 0; i < dims; ++i) {
            //     dE_o += std::real(origin_coeff[i] * (node_coeffs[i] - std::conj(origin_coeff[i]))) * D_l[i];
            // }
            dE_o = vec_shift_diff_origin<Real>(dims, &origin_coeff[0], &node_coeffs[0], &D_l[0]);

            dE_difference_o += dE_o / (std::pow(2*M_PI, 3)) * std::pow(delta_kl, 3);
        }

        for (int l = 2; l < path_to_target.Dim() - 1; ++l) {
            Real dE_t = 0;

            auto &target_coeff = outgoing_pw_target[l];
            auto &D_l = interaction_mat[l];
            sctl::Long target_node = path_to_target[l];
            Real delta_kl = delta_k[l];
            auto &node_coeffs = incoming_pw[target_node];
            auto &shift_mat_l = shift_mat[l];

            dE_t = vec_tridot<Real, false, false>(dims, &target_coeff[0], &node_coeffs[0], &D_l[0]);

            dE_difference_t += dE_t / (std::pow(2*M_PI, 3)) * std::pow(delta_kl, 3);
        }

        // std::cout << "diff_planewave, origin, target: " << dE_difference_o << ", " << dE_difference_t << std::endl;

        return dE_difference_t - dE_difference_o;
    }


    template <typename Real>
    Real HPDMKPtTree<Real>::eval_shift_energy_res(sctl::Long i_particle, sctl::Vector<sctl::Long>& target_path, Real x, Real y, Real z, Real q) {
        Real dE_rt = 0;

        // only consider the leaf node that contains the target point
        auto i_depth = target_path.Dim() - 1;
        sctl::Long i_node = target_path[i_depth];

        auto &node_attr = this->GetNodeAttr();
        assert(isleaf(node_attr[i_node]));

        // self interaction
        if (node_particles[i_node].Dim() > 0) {
            dE_rt += eval_shift_energy_res_i(i_node, i_depth, i_particle, x, y, z, q);
        }

        // colleague interaction
        for (auto i_nbr : neighbors[i_node].colleague) {
            if (node_particles[i_nbr].Dim() > 0) {
                dE_rt += eval_shift_energy_res_ij(i_node, i_depth, i_nbr, i_particle, x, y, z, q);
            }
        }

        for (auto i_nbr : neighbors[i_node].coarsegrain) {
            if (node_particles[i_nbr].Dim() > 0) {
                dE_rt += eval_shift_energy_res_ij(i_node, i_depth, i_nbr, i_particle, x, y, z, q);
            }
        }

        return dE_rt;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::eval_shift_energy_res_i(sctl::Long i_node, int i_depth, sctl::Long i_particle, Real x, Real y, Real z, Real q) {
        Real potential = 0;

        for (int j = 0; j < node_particles[i_node].Dim(); ++j) {
            int j_particle = node_particles[i_node][j];

            if (j_particle != i_particle) {
                Real xj = r_src_sorted[j_particle * 3];
                Real yj = r_src_sorted[j_particle * 3 + 1];
                Real zj = r_src_sorted[j_particle * 3 + 2];
                Real r_ij = std::sqrt(dist2(x, y, z, xj, yj, zj));
                
                if (r_ij <= boxsize[i_depth]) {
                    potential += charge_sorted[j_particle] * residual_kernel<Real>(r_ij, real_poly, boxsize[i_depth]);
                }
            }
        }

        return q * potential;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::eval_shift_energy_res_ij(sctl::Long i_node, int i_depth, sctl::Long i_nbr, sctl::Long i_particle, Real x, Real y, Real z, Real q) {
        Real potential = 0;
                
        auto shift_ij = node_shift(i_node, i_nbr);

        Real center_xi = centers[i_node * 3];
        Real center_yi = centers[i_node * 3 + 1];
        Real center_zi = centers[i_node * 3 + 2];

        Real center_xj = centers[i_nbr * 3];
        Real center_yj = centers[i_nbr * 3 + 1];
        Real center_zj = centers[i_nbr * 3 + 2];

        Real xi = x - center_xi - shift_ij[0];
        Real yi = y - center_yi - shift_ij[1];
        Real zi = z - center_zi - shift_ij[2];

        for (auto j_particle : node_particles[i_nbr]) {
            if (j_particle != i_particle) {
                Real xj = r_src_sorted[j_particle * 3] - center_xj;
                Real yj = r_src_sorted[j_particle * 3 + 1] - center_yj;
                Real zj = r_src_sorted[j_particle * 3 + 2] - center_zj;

                Real r_ij = std::sqrt(dist2(xi, yi, zi, xj, yj, zj));
                if (r_ij <= boxsize[i_depth]) {
                    potential += charge_sorted[j_particle] * residual_kernel<Real>(r_ij, real_poly, boxsize[i_depth]);
                }
            }
        }


        return q * potential;
    }

    // evaluate the residual energy for the target points by direct_eval
    template <typename Real>
    Real HPDMKPtTree<Real>::eval_shift_energy_res_vec(sctl::Long i_particle, sctl::Vector<sctl::Long>& target_path, Real x, Real y, Real z, Real q) {
        Real dE_rt = 0;

        // only consider the leaf node that contains the target point
        auto i_depth = target_path.Dim() - 1;
        sctl::Long i_node = target_path[i_depth];

        auto &node_attr = this->GetNodeAttr();
        assert(isleaf(node_attr[i_node]));

        Real r_src_i[3] = {x, y, z};

        int n_trg = 0;
        sctl::Long j_particle;

        // self interaction
        if (node_particles[i_node].Dim() > 0) {
            for (int i = 0; i < node_particles[i_node].Dim(); ++i) {
                j_particle = node_particles[i_node][i];
                // if (j_particle == i_particle) continue;
                vec_trg[3 * (n_trg)] = r_src_sorted[j_particle * 3];
                vec_trg[3 * (n_trg) + 1] = r_src_sorted[j_particle * 3 + 1];
                vec_trg[3 * (n_trg) + 2] = r_src_sorted[j_particle * 3 + 2];
                q_trg[n_trg] = charge_sorted[j_particle];
                n_trg += 1;
            }
        }

        // colleague interaction
        for (auto i_nbr : neighbors[i_node].colleague) {
            if (node_particles[i_nbr].Dim() > 0) {
                auto shift_ij = node_shift(i_node, i_nbr);
                Real dcx = centers[i_node * 3] - centers[i_nbr * 3] ;
                Real dcy = centers[i_node * 3 + 1] - centers[i_nbr * 3 + 1];
                Real dcz = centers[i_node * 3 + 2] - centers[i_nbr * 3 + 2];
                for (int i = 0; i < node_particles[i_nbr].Dim(); ++i) {
                    j_particle = node_particles[i_nbr][i];
                    // if (j_particle == i_particle) continue;
                    vec_trg[3 * (n_trg)] = r_src_sorted[j_particle * 3] + dcx + shift_ij[0];
                    vec_trg[3 * (n_trg) + 1] = r_src_sorted[j_particle * 3 + 1] + dcy + shift_ij[1];
                    vec_trg[3 * (n_trg) + 2] = r_src_sorted[j_particle * 3 + 2] + dcz + shift_ij[2];
                    q_trg[n_trg] = charge_sorted[j_particle];
                    n_trg += 1;
                }
            }
        }

        for (auto i_nbr : neighbors[i_node].coarsegrain) {
            if (node_particles[i_nbr].Dim() > 0) {
                auto shift_ij = node_shift(i_node, i_nbr);
                Real dcx = centers[i_node * 3] - centers[i_nbr * 3] ;
                Real dcy = centers[i_node * 3 + 1] - centers[i_nbr * 3 + 1];
                Real dcz = centers[i_node * 3 + 2] - centers[i_nbr * 3 + 2];
                for (int i = 0; i < node_particles[i_nbr].Dim(); ++i) {
                    j_particle = node_particles[i_nbr][i];
                    // if (j_particle == i_particle) continue;
                    vec_trg[3 * (n_trg)] = r_src_sorted[j_particle * 3] + dcx + shift_ij[0];
                    vec_trg[3 * (n_trg) + 1] = r_src_sorted[j_particle * 3 + 1] + dcy + shift_ij[1];
                    vec_trg[3 * (n_trg) + 2] = r_src_sorted[j_particle * 3 + 2] + dcz + shift_ij[2];
                    q_trg[n_trg] = charge_sorted[j_particle];
                    n_trg += 1;
                }
            }
        }

        dE_rt = direct_eval<Real>(&r_src_i[0], &charge_sorted[i_particle], n_trg, &vec_trg[0], &q_trg[0], boxsize[i_depth], n_digits);

        return dE_rt;
    }

    template struct HPDMKPtTree<float>;
    template struct HPDMKPtTree<double>;
}