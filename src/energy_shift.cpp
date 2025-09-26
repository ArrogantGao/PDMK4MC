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
    Real HPDMKPtTree<Real>::energy_shift(sctl::Long i_particle, Real dx, Real dy, Real dz) {
        Real q = charge_sorted[i_particle];
        Real x_o = r_src_sorted[i_particle * 3];
        Real y_o = r_src_sorted[i_particle * 3 + 1];
        Real z_o = r_src_sorted[i_particle * 3 + 2];
        Real x_t = my_mod(x_o + dx, L);
        Real y_t = my_mod(y_o + dy, L);
        Real z_t = my_mod(z_o + dz, L);
        Real dr = std::sqrt(dx * dx + dy * dy + dz * dz);

        std::cout << "q: " << q << std::endl;
        std::cout << "x_t: " << x_t << ", y_t: " << y_t << ", z_t: " << z_t << std::endl;
        std::cout << "x_o: " << x_o << ", y_o: " << y_o << ", z_o: " << z_o << std::endl;

        locate_particle(path_to_target, x_t, y_t, z_t);
        locate_particle(path_to_origin, x_o, y_o, z_o);

        std::cout << "path_to_target: " << path_to_target.Dim() << std::endl;
        for (int i = 0; i < path_to_target.Dim(); ++i) {
            std::cout << "[" << i << "]: " << path_to_target[i] << std::endl;
            std::cout << "nbr[" << i << "]: " << neighbors[i].colleague << std::endl;
        }
        std::cout << "path_to_origin: " << path_to_origin.Dim() << std::endl;
        for (int i = 0; i < path_to_origin.Dim(); ++i) {
            std::cout << "[" << i << "]: " << path_to_origin[i] << std::endl;
        }

        init_target_planewave_coeffs(target_planewave_coeffs, path_to_target, x_t, y_t, z_t, q);
        init_target_planewave_coeffs(origin_planewave_coeffs, path_to_origin, x_o, y_o, z_o, q);

        Real dE_window = energy_window_shift(origin_planewave_coeffs, target_planewave_coeffs) / 2;
        Real dE_difference = energy_difference_shift(origin_planewave_coeffs, path_to_origin, target_planewave_coeffs, path_to_target);

        Real dE_residual_target = energy_residual_shift(i_particle, path_to_target, x_t, y_t, z_t, q);
        Real dE_residual_origin = energy_residual_shift(i_particle, path_to_origin, x_o, y_o, z_o, q);

        std::cout << "dE_window: " << dE_window << ", dE_difference: " << dE_difference << ", dE_residual_target: " << dE_residual_target << ", dE_residual_origin: " << dE_residual_origin << std::endl;

        Real res_l2_t = residual_energy_shift_direct(2, x_t, y_t, z_t, q) - q * q * residual_kernel<Real>(dr, real_poly, boxsize[2]);
        Real res_l2_o = residual_energy_shift_direct(2, x_o, y_o, z_o, q);

        Real dE_difference_origin_direct = q * energy_difference_shift_direct(path_to_origin.Dim() - 1, i_particle, x_o, y_o, z_o);
        Real dE_difference_target_direct = q * energy_difference_shift_direct(path_to_target.Dim() - 1, i_particle, x_t, y_t, z_t);

        Real dE_residual_target_direct = residual_energy_shift_direct(path_to_target.Dim() - 1, x_t, y_t, z_t, q) - q * q * residual_kernel<Real>(dr, real_poly, boxsize[path_to_target.Dim() - 1]);
        Real dE_residual_origin_direct = residual_energy_shift_direct(path_to_origin.Dim() - 1, x_o, y_o, z_o, q);

        std::cout << "diff_t_direct: " << dE_difference_target_direct << ", res_t_direct: " << dE_residual_target_direct << ", sum: " << dE_difference_target_direct + dE_residual_target_direct << std::endl;
        std::cout << "diff_o_direct: " << dE_difference_origin_direct << ", res_o_direct: " << dE_residual_origin_direct << ", sum: " << dE_difference_origin_direct + dE_residual_origin_direct << std::endl;

        std::cout << "residual_t_l2: " << res_l2_t << std::endl;
        std::cout << "residual_o_l2: " << res_l2_o << std::endl;

        std::cout << "dE_window + res_l2 = " << dE_window + res_l2_t - res_l2_o << std::endl;

        Real dE_shift = dE_window + dE_difference + dE_residual_target - dE_residual_origin;

        return dE_shift;
    }
    
    template <typename Real>
    Real HPDMKPtTree<Real>::energy_window_shift(std::vector<Rank3Tensor<std::complex<Real>>>& origin_coeffs, std::vector<Rank3Tensor<std::complex<Real>>>& target_coeffs) {
        Real dE_window = 0;

        auto &target_root_coeffs = target_coeffs[0];
        auto &origin_root_coeffs = origin_coeffs[0];
        auto &root_coeffs = plane_wave_coeffs[root()];
        auto &window = interaction_matrices[0];

        #pragma omp parallel for reduction(+:dE_window)
        for (int i = 0; i < target_root_coeffs.Dim(); ++i) {
            if (window[i] != 0) {
                dE_window += 2.0 * std::real((target_root_coeffs[i] - origin_root_coeffs[i]) * window[i] * std::conj(root_coeffs[i] - origin_root_coeffs[i]));
            }
        }

        #ifdef DEBUG
        // compute dE_window_direct
        Real dE_window_direct = 0;
        for (int i = 0; i < target_root_coeffs.Dim(); ++i) {
            if (window[i] != 0) {
                dE_window_direct += std::real((root_coeffs[i] - origin_root_coeffs[i] + target_root_coeffs[i]) * window[i] * std::conj(root_coeffs[i] - origin_root_coeffs[i] + target_root_coeffs[i])) - std::real(root_coeffs[i] * window[i] * std::conj(root_coeffs[i]));
            }
        }
        std::cout << "dE_window: " << dE_window << ", dE_window_direct: " << dE_window_direct << std::endl;
        #endif

        dE_window *= 1 / (std::pow(2*M_PI, 3)) * std::pow(delta_k[0], 3);
        return dE_window;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::energy_difference_shift(std::vector<Rank3Tensor<std::complex<Real>>>& origin_coeffs, sctl::Vector<sctl::Long>& origin_path, std::vector<Rank3Tensor<std::complex<Real>>>& target_coeffs, sctl::Vector<sctl::Long>& target_path) {
        Real dE_difference_t = 0;
        Real dE_difference_o = 0;

        auto &node_mid = this->GetNodeMID();

        for (int l = 2; l < origin_path.Dim() - 1; ++l) {
            Real dE_o = 0;

            auto &origin_coeff = origin_coeffs[l];
            auto &D_l = interaction_matrices[l];

            sctl::Long i_node = origin_path[l];
            assert(node_mid[i_node].Depth() == l);

            Real delta_kl = delta_k[l];
            int n_kl = n_k[l];

            auto &node_coeffs = plane_wave_coeffs[i_node];
            // self interaction
            if (node_particles[i_node].Dim() > 0) {
                #pragma omp parallel for reduction(+:dE_o)
                for (int i = 0; i < D_l.Dim(); ++i) {
                    // dE_o += std::real(origin_coeff[i] * D_l[i] * std::conj(std::complex<Real>(2.0, 0) * node_coeffs[i] - origin_coeff[i]));
                    dE_o += std::real(origin_coeff[i] * D_l[i] * std::conj(node_coeffs[i] - origin_coeff[i]));
                }
            }

            // colleague interaction
            assert(neighbors[i_node].colleague.Dim() == 26);
            for (auto i_nbr : neighbors[i_node].colleague) {
                if (node_particles[i_nbr].Dim() > 0) {

                    auto shift_ij = node_shift(i_node, i_nbr);
                    auto &node_coeffs_j = plane_wave_coeffs[i_nbr];

                    std::complex<Real> exp_ik_shiftx = std::exp(std::complex<Real>(0, 1) * delta_kl * shift_ij[0]);
                    std::complex<Real> exp_ik_shifty = std::exp(std::complex<Real>(0, 1) * delta_kl * shift_ij[1]);
                    std::complex<Real> exp_ik_shiftz = std::exp(std::complex<Real>(0, 1) * delta_kl * shift_ij[2]);

                    for (int i = 0; i < 2 * n_kl + 1; ++i) {
                        int n = i - n_kl;
                        kx_cache[i] = std::pow(exp_ik_shiftx, n);
                        ky_cache[i] = std::pow(exp_ik_shifty, n);
                    }

                    for (int i = 0; i < n_kl + 1; ++i) {
                        kz_cache[i] = std::pow(exp_ik_shiftz, i);
                    }   

                    int d = 2 * n_kl + 1;
                    // #pragma omp parallel for reduction(+:potential)
                    for (int i = 0; i < d; ++i) {
                        auto t1 = kx_cache[i];
                        for (int j = 0; j < d; ++j) {
                            auto t2 = t1 * ky_cache[j];
                            for (int k = 0; k < n_kl + 1; ++k) {
                                if (D_l(i, j, k) != 0) {
                                    dE_o += std::real(origin_coeff(i, j, k) * D_l(i, j, k) * std::conj(node_coeffs_j(i, j, k)) * t2 * kz_cache[k]);
                                }
                            }
                        }
                    }
                }
            }

            dE_difference_o += dE_o / (std::pow(2*M_PI, 3)) * std::pow(delta_kl, 3);
        }


        for (int l = 2; l < target_path.Dim() - 1; ++l) {
            Real dE_t = 0;

            auto &target_coeff = target_coeffs[l];
            auto &D_l = interaction_matrices[l];
            sctl::Long target_node = target_path[l];

            Real delta_kl = delta_k[l];
            int n_kl = n_k[l];

            auto &node_coeffs = plane_wave_coeffs[target_node];
            // self interaction
            if (node_particles[target_node].Dim() > 0) {
                if ((l < origin_path.Dim() - 1) && (origin_path[l] == target_node)) {
                    auto &origin_coeff = origin_coeffs[l];
                    #pragma omp parallel for reduction(+:dE_t)
                    for (int i = 0; i < D_l.Dim(); ++i) {
                        // dE_t += std::real(target_coeff[i] * D_l[i] * std::conj(std::complex<Real>(2.0, 0) * node_coeffs[i] - std::complex<Real>(2.0, 0) * origin_coeff[i] + target_coeff[i]));
                        dE_t += std::real(target_coeff[i] * D_l[i] * std::conj(node_coeffs[i] - origin_coeff[i]));
                    }
                }
                else {
                    #pragma omp parallel for reduction(+:dE_t)
                    for (int i = 0; i < D_l.Dim(); ++i) {
                        // dE_t += std::real(target_coeff[i] * D_l[i] * std::conj(std::complex<Real>(2.0, 0) * node_coeffs[i] + target_coeff[i]));
                        dE_t += std::real(target_coeff[i] * D_l[i] * std::conj(node_coeffs[i]));
                    }
                }
            }

            // colleague interaction
            assert(neighbors[target_node].colleague.Dim() == 26);
            for (auto i_nbr : neighbors[target_node].colleague) {
                if (node_particles[i_nbr].Dim() > 0) {

                    auto shift_ij = node_shift(target_node, i_nbr);
                    auto &node_coeffs_j = plane_wave_coeffs[i_nbr];

                    std::complex<Real> exp_ik_shiftx = std::exp(std::complex<Real>(0, 1) * delta_kl * shift_ij[0]);
                    std::complex<Real> exp_ik_shifty = std::exp(std::complex<Real>(0, 1) * delta_kl * shift_ij[1]);
                    std::complex<Real> exp_ik_shiftz = std::exp(std::complex<Real>(0, 1) * delta_kl * shift_ij[2]);

                    for (int i = 0; i < 2 * n_kl + 1; ++i) {
                        int n = i - n_kl;
                        kx_cache[i] = std::pow(exp_ik_shiftx, n);
                        ky_cache[i] = std::pow(exp_ik_shifty, n);
                    }

                    for (int i = 0; i < n_kl + 1; ++i) {
                        kz_cache[i] = std::pow(exp_ik_shiftz, i);
                    }   

                    int d = 2 * n_kl + 1;
                    if ((l < origin_path.Dim() - 1) && (origin_path[l] == i_nbr)) {
                        auto &origin_coeff = origin_coeffs[l];
                        for (int i = 0; i < d; ++i) {
                            auto t1 = kx_cache[i];
                            for (int j = 0; j < d; ++j) {
                                auto t2 = t1 * ky_cache[j];
                                for (int k = 0; k < n_kl + 1; ++k) {
                                    if (D_l(i, j, k) != 0) {
                                        dE_t += std::real(target_coeff(i, j, k) * D_l(i, j, k) * std::conj((node_coeffs_j(i, j, k) - origin_coeff(i, j, k))) * t2 * kz_cache[k]);
                                    }
                                }
                            }
                        }
                    }
                    else {
                        for (int i = 0; i < d; ++i) {
                            auto t1 = kx_cache[i];
                            for (int j = 0; j < d; ++j) {
                                auto t2 = t1 * ky_cache[j];
                                for (int k = 0; k < n_kl + 1; ++k) {
                                    if (D_l(i, j, k) != 0) {
                                        dE_t += std::real(target_coeff(i, j, k) * D_l(i, j, k) * std::conj(node_coeffs_j(i, j, k)) * t2 * kz_cache[k]);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            dE_difference_t += dE_t / (std::pow(2*M_PI, 3)) * std::pow(delta_kl, 3);
        }

        std::cout << "dE_difference_t: " << dE_difference_t << ", dE_difference_o: " << dE_difference_o << std::endl;

        return dE_difference_t - dE_difference_o;
    }


    template <typename Real>
    Real HPDMKPtTree<Real>::energy_residual_shift(sctl::Long i_particle, sctl::Vector<sctl::Long>& target_path, Real x, Real y, Real z, Real q) {
        Real dE_rt = 0;

        // only consider the leaf node that contains the target point
        auto i_depth = target_path.Dim() - 1;
        sctl::Long i_node = target_path[i_depth];

        auto &node_attr = this->GetNodeAttr();
        assert(isleaf(node_attr[i_node]));

        // self interaction
        if (node_particles[i_node].Dim() > 0) {
            dE_rt += energy_residual_shift_i(i_node, i_depth, i_particle, x, y, z, q);
        }

        std::cout << "number of colleague: " << neighbors[i_node].colleague.Dim() << std::endl;
        std::cout << "number of coarsegrain: " << neighbors[i_node].coarsegrain.Dim() << std::endl;

        // colleague interaction
        for (auto i_nbr : neighbors[i_node].colleague) {
            if (node_particles[i_nbr].Dim() > 0) {
                dE_rt += energy_residual_shift_ij(i_node, i_depth, i_nbr, i_particle, x, y, z, q);
            }
        }

        for (auto i_nbr : neighbors[i_node].coarsegrain) {
            if (node_particles[i_nbr].Dim() > 0) {
                dE_rt += energy_residual_shift_ij(i_node, i_depth, i_nbr, i_particle, x, y, z, q);
            }
        }

        return dE_rt;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::energy_residual_shift_i(sctl::Long i_node, int i_depth, sctl::Long i_particle, Real x, Real y, Real z, Real q) {
        Real potential = 0;

        #pragma omp parallel for reduction(+:potential)
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
    Real HPDMKPtTree<Real>::energy_residual_shift_ij(sctl::Long i_node, int i_depth, sctl::Long i_nbr, sctl::Long i_particle, Real x, Real y, Real z, Real q) {
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

        #pragma omp parallel for reduction(+:potential)
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

    template struct HPDMKPtTree<float>;
    template struct HPDMKPtTree<double>;
}