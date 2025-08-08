#include <hpdmk.h>
#include <tree.hpp>
#include <kernels.hpp>
#include <utils.hpp>

#include <vector>
#include <array>
#include <cmath>
#include <complex>
#include <algorithm>

#include <sctl.hpp>
#include <mpi.h>


namespace hpdmk {
    template <typename Real>
    Real HPDMKPtTree<Real>::energy() {
        Real energy = window_energy() + difference_energy() + residual_energy();
        return energy;
    }

    // W + D_0 + D_1
    template <typename Real>
    Real HPDMKPtTree<Real>::window_energy() {
        Real energy = 0;

        auto &root_coeffs = plane_wave_coeffs[root()];
        auto &window = interaction_matrices[0];

        assert(root_coeffs.Dim() == window.Dim());

        // #pragma omp parallel for reduction(+:energy)
        for (int i = 0; i < root_coeffs.Dim(); ++i) {
            energy += std::real(root_coeffs[i] * window[i] * std::conj(root_coeffs[i]));
        }

        energy *= 1 / (2 * std::pow(2*M_PI, 3)) * std::pow(delta_k[0], 3);

        // self energy
        Real sigma = sigmas[2];
        Real self_energy = Q / (std::sqrt(M_PI) * sigma);
        energy -= self_energy;

        return energy;
    }

    // D_l for non-leaf nodes with depth >= 2
    template <typename Real>
    Real HPDMKPtTree<Real>::difference_energy() {
        Real energy = 0;
        auto &node_attr = this->GetNodeAttr();

        for (int l = 2; l < max_depth - 1; ++l) {
            for (sctl::Long i_node : level_indices[l]) {
                // non-leaf nodes, with particles inside
                if (!isleaf(node_attr[i_node]) && node_particles[i_node].Dim() > 0) {
                    energy += difference_energy_i(l, i_node);

                    // non-leaf nodes should have 26 colleague neighbors
                    assert(neighbors[i_node].colleague.Dim() == 26);
                    for (auto i_nbr : neighbors[i_node].colleague) {
                        if (node_particles[i_nbr].Dim() > 0) {
                            energy += difference_energy_ij(l, i_node, i_nbr);
                        }
                    }
                }
            }
        }

        return energy;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::difference_energy_i(int i_depth, sctl::Long i_node) {
        Real energy = 0;

        auto &D_l = interaction_matrices[i_depth];
        auto &node_coeffs = plane_wave_coeffs[i_node];

        // #pragma omp parallel for reduction(+:energy)
        for (int i = 0; i < D_l.Dim(); ++i) {
            energy += std::real(node_coeffs[i] * D_l[i] * std::conj(node_coeffs[i]));
        }

        energy *= 1 / (2 * std::pow(2*M_PI, 3)) * std::pow(delta_k[i_depth], 3);

        // self energy
        Real sigma = sigmas[i_depth];
        Real sigma_lp1 = sigmas[i_depth + 1];
        Real Q_node = 0;
        for (auto i_particle : node_particles[i_node]) {
            Q_node += charge_sorted[i_particle] * charge_sorted[i_particle];
        }
        Real self_energy = Q_node * (1 / (std::sqrt(M_PI) * sigma_lp1) - 1 / (std::sqrt(M_PI) * sigma));
        energy -= self_energy;

        return energy;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::difference_energy_ij(int i_depth, sctl::Long i_node, sctl::Long j_node) {
        Real energy = 0;
        
        auto shift_ij = node_shift(i_node, j_node);
        Real delta_ki = delta_k[i_depth];
        int n_ki = n_k[i_depth];

        auto &D_l = interaction_matrices[i_depth];
        auto &node_coeffs_i = plane_wave_coeffs[i_node];
        auto &node_coeffs_j = plane_wave_coeffs[j_node];

        std::complex<Real> exp_ik_shiftx = std::exp(std::complex<Real>(0, 1) * delta_ki * shift_ij[0]);
        std::complex<Real> exp_ik_shifty = std::exp(std::complex<Real>(0, 1) * delta_ki * shift_ij[1]);
        std::complex<Real> exp_ik_shiftz = std::exp(std::complex<Real>(0, 1) * delta_ki * shift_ij[2]);

        int d = 2 * n_ki + 1;

        #pragma omp parallel for
        for (int i = 0; i < d; ++i) {
            int n = i - n_ki;
            kx_cache[i] = std::pow(exp_ik_shiftx, n);
            ky_cache[i] = std::pow(exp_ik_shifty, n);
            kz_cache[i] = std::pow(exp_ik_shiftz, n);
        }

        // #pragma omp parallel for reduction(+:energy)
        for (int i = 0; i < d; ++i) {
            auto t1 = kx_cache[i];
            for (int j = 0; j < d; ++j) {
                auto t2 = t1 * ky_cache[j];
                for (int k = 0; k < d; ++k) {
                    energy += std::real(node_coeffs_i[offset(i, j, k, d)] * D_l[offset(i, j, k, d)] * std::conj(node_coeffs_j[offset(i, j, k, d)]) * t2 * kz_cache[k]);
                }
            }
        }

        energy *= 1 / (2 * std::pow(2*M_PI, 3)) * std::pow(delta_ki, 3);

        return energy;
    }

    // energy of the residual term, for leaf nodes
    template <typename Real>
    Real HPDMKPtTree<Real>::residual_energy() {
        Real energy = 0;

        auto &node_attr = this->GetNodeAttr();

        for (int l = 2; l < max_depth; ++l) {
            for (sctl::Long i_node : level_indices[l]) {
                if (isleaf(node_attr[i_node]) && node_particles[i_node].Dim() > 0) {

                    // self interaction
                    energy += residual_energy_i(l, i_node);

                    // coarse-grained neighbors
                    for (auto i_nbr : neighbors[i_node].coarsegrain) {
                        if (node_particles[i_nbr].Dim() > 0) {
                            energy += residual_energy_ij(l, i_node, i_nbr);
                        }
                    }

                    // colleague neighbors
                    for (auto i_nbr : neighbors[i_node].colleague) {
                        if (node_particles[i_nbr].Dim() > 0) {
                            energy += residual_energy_ij(l, i_node, i_nbr);
                        }
                    }
                }
            }
        }

        return energy;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::residual_energy_i(int i_depth, sctl::Long i_node) {
        Real energy = 0;
        
        #pragma omp parallel for reduction(+:energy)
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
                    energy += charge_sorted[i_particle] * charge_sorted[j_particle] * gaussian_residual<Real>(r_ij, sigmas[i_depth]);
                }
            }
        }

        return energy;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::residual_energy_ij(int i_depth, sctl::Long i_node, sctl::Long j_node) {
        Real energy = 0;

        auto shift_ij = node_shift(i_node, j_node);

        Real center_xi = centers[i_node * 3];
        Real center_yi = centers[i_node * 3 + 1];
        Real center_zi = centers[i_node * 3 + 2];

        Real center_xj = centers[j_node * 3];
        Real center_yj = centers[j_node * 3 + 1];
        Real center_zj = centers[j_node * 3 + 2];

        #pragma omp parallel for reduction(+:energy)
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
                    energy += charge_sorted[i_particle] * charge_sorted[j_particle] * gaussian_residual<Real>(r_ij, sigmas[i_depth]) / 2;
                }
            }
        }

        return energy;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::difference_energy_direct_i(int i_depth, sctl::Long i_node) {
        Real energy = 0;

        for (int i = 0; i < node_particles[i_node].Dim() - 1; ++i) {
            for (int j = i + 1; j < node_particles[i_node].Dim(); ++j) {
                int i_particle = node_particles[i_node][i];
                int j_particle = node_particles[i_node][j];
                Real xi = r_src_sorted[i_particle * 3];
                Real yi = r_src_sorted[i_particle * 3 + 1];
                Real zi = r_src_sorted[i_particle * 3 + 2];
                Real xj = r_src_sorted[j_particle * 3];
                Real yj = r_src_sorted[j_particle * 3 + 1];
                Real zj = r_src_sorted[j_particle * 3 + 2];
                Real r_ij = std::sqrt(dist2(xi, yi, zi, xj, yj, zj));
                energy += charge_sorted[i_particle] * charge_sorted[j_particle] * gaussian_difference_real<Real>(r_ij, sigmas[i_depth], sigmas[i_depth + 1]);
            }
        }
        return energy;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::difference_energy_direct() {
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
                                        energy += charge_sorted[i_particle] * charge_sorted[j] * gaussian_difference_real<Real>(r_ij, sigmas[l], sigmas[l + 1]) / 2;
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
    Real HPDMKPtTree<Real>::residual_energy_direct() {
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
                                        energy += charge_sorted[i_particle] * charge_sorted[j] * gaussian_residual<Real>(r_ij, sigmas[l]) / 2;
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
    Real HPDMKPtTree<Real>::potential_target_window(Real x, Real y, Real z) {
        Real potential = 0;

        auto &target_root_coeffs = target_planewave_coeffs[0];
        auto &root_coeffs = plane_wave_coeffs[root()];
        auto &window = interaction_matrices[0];

        assert(target_root_coeffs.Dim() == root_coeffs.Dim());
        assert(target_root_coeffs.Dim() == window.Dim());

        #pragma omp parallel for reduction(+:potential)
        for (int i = 0; i < target_root_coeffs.Dim(); ++i) {
            potential += std::real(target_root_coeffs[i] * window[i] * std::conj(root_coeffs[i]));
        }

        potential *= 1 / (std::pow(2*M_PI, 3)) * std::pow(delta_k[0], 3);

        return potential;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::potential_target_difference(Real x, Real y, Real z) {
        Real potential_total = 0;
        Real potential = 0;

        auto &node_mid = this->GetNodeMID();

        for (int l = 2; l < path_to_target.Dim() - 1; ++l) {
            potential = 0;

            auto &target_coeffs = target_planewave_coeffs[l];
            auto &D_l = interaction_matrices[l];
            assert(target_coeffs.Dim() == D_l.Dim());

            sctl::Long i_node = path_to_target[l];
            assert(node_mid[i_node].Depth() == l);

            Real delta_kl = delta_k[l];
            int n_kl = n_k[l];

            auto &node_coeffs = plane_wave_coeffs[i_node];
            // self interaction
            if (node_particles[i_node].Dim() > 0) {
                #pragma omp parallel for reduction(+:potential)
                for (int i = 0; i < D_l.Dim(); ++i) {
                    potential += std::real(target_coeffs[i] * D_l[i] * std::conj(node_coeffs[i]));
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
                        kz_cache[i] = std::pow(exp_ik_shiftz, n);
                    }

                    int d = 2 * n_kl + 1;
                    #pragma omp parallel for reduction(+:potential)
                    for (int i = 0; i < d; ++i) {
                        auto t1 = kx_cache[i];
                        for (int j = 0; j < d; ++j) {
                            auto t2 = t1 * ky_cache[j];
                            for (int k = 0; k < d; ++k) {
                                potential += std::real(target_coeffs[offset(i, j, k, d)] * D_l[offset(i, j, k, d)] * std::conj(node_coeffs_j[offset(i, j, k, d)]) * t2 * kz_cache[k]);
                            }
                        }
                    }
                }
            }

            potential_total += potential / (std::pow(2*M_PI, 3)) * std::pow(delta_kl, 3);
        }
        
        return potential_total;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::potential_target_difference_direct(Real x, Real y, Real z) {
        Real potential = 0;

        auto i_depth = path_to_target.Dim() - 1;
        sctl::Long i_node = path_to_target[i_depth];

        auto &node_attr = this->GetNodeAttr();
        assert(isleaf(node_attr[i_node]));

        Real sigma_2 = sigmas[2];
        Real sigma_l = sigmas[i_depth];

        for (int j = 0; j < charge_sorted.Dim(); ++j) {
            for (int mx = -1; mx <= 1; mx++) {
                for (int my = -1; my <= 1; my++) {
                    for (int mz = -1; mz <= 1; mz++) {
                        Real xj = r_src_sorted[j * 3] + mx * L;
                        Real yj = r_src_sorted[j * 3 + 1] + my * L;
                        Real zj = r_src_sorted[j * 3 + 2] + mz * L;
                        Real r_ij = std::sqrt(dist2(x, y, z, xj, yj, zj));
                        potential += charge_sorted[j] * gaussian_difference_real<Real>(r_ij, sigma_2, sigma_l);
                    }
                }
            }
        }

        return potential;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::potential_target_residual_i(int i_depth, sctl::Long i_node, Real x, Real y, Real z) {
        Real potential = 0;

        #pragma omp parallel for reduction(+:potential)
        for (int j = 0; j < node_particles[i_node].Dim(); ++j) {
            int j_particle = node_particles[i_node][j];

            Real xj = r_src_sorted[j_particle * 3];
            Real yj = r_src_sorted[j_particle * 3 + 1];
            Real zj = r_src_sorted[j_particle * 3 + 2];
            Real r_ij = std::sqrt(dist2(x, y, z, xj, yj, zj));
            
            if (r_ij <= boxsize[i_depth]) {
                potential += charge_sorted[j_particle] * gaussian_residual<Real>(r_ij, sigmas[i_depth]);
            }
        }

        return potential;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::potential_target_residual_ij(int i_depth, sctl::Long i_node, sctl::Long j_node, Real x, Real y, Real z) {
        Real potential = 0;
        
        auto shift_ij = node_shift(i_node, j_node);

        Real center_xi = centers[i_node * 3];
        Real center_yi = centers[i_node * 3 + 1];
        Real center_zi = centers[i_node * 3 + 2];

        Real center_xj = centers[j_node * 3];
        Real center_yj = centers[j_node * 3 + 1];
        Real center_zj = centers[j_node * 3 + 2];

        Real xi = x - center_xi - shift_ij[0];
        Real yi = y - center_yi - shift_ij[1];
        Real zi = z - center_zi - shift_ij[2];

        #pragma omp parallel for reduction(+:potential)
        for (auto j_particle : node_particles[j_node]) {
            Real xj = r_src_sorted[j_particle * 3] - center_xj;
            Real yj = r_src_sorted[j_particle * 3 + 1] - center_yj;
            Real zj = r_src_sorted[j_particle * 3 + 2] - center_zj;

            Real r_ij = std::sqrt(dist2(xi, yi, zi, xj, yj, zj));
            if (r_ij <= boxsize[i_depth]) {
                potential += charge_sorted[j_particle] * gaussian_residual<Real>(r_ij, sigmas[i_depth]);
            }
        }

        return potential;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::potential_target_residual(Real x, Real y, Real z) {
        Real potential = 0;

        // only consider the leaf node that contains the target point
        auto i_depth = path_to_target.Dim() - 1;
        sctl::Long i_node = path_to_target[i_depth];

        auto &node_attr = this->GetNodeAttr();
        assert(isleaf(node_attr[i_node]));

        // self interaction
        if (node_particles[i_node].Dim() > 0) {
            potential += potential_target_residual_i(i_depth, i_node, x, y, z);
        }

        // colleague interaction
        for (auto i_nbr : neighbors[i_node].colleague) {    
            if (node_particles[i_nbr].Dim() > 0) {
                potential += potential_target_residual_ij(i_depth, i_node, i_nbr, x, y, z);
            }
        }

        return potential;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::potential_target_residual_direct(Real x, Real y, Real z) {
        Real potential = 0;

        auto i_depth = path_to_target.Dim() - 1;
        sctl::Long i_node = path_to_target[i_depth];

        auto &node_attr = this->GetNodeAttr();
        assert(isleaf(node_attr[i_node]));

        for (int j = 0; j < charge_sorted.Dim(); ++j) {
            for (int mx = -1; mx <= 1; mx++) {
                for (int my = -1; my <= 1; my++) {
                    for (int mz = -1; mz <= 1; mz++) {
                        Real xj = r_src_sorted[j * 3] + mx * L;
                        Real yj = r_src_sorted[j * 3 + 1] + my * L;
                        Real zj = r_src_sorted[j * 3 + 2] + mz * L;
                        Real r_ij = std::sqrt(dist2(x, y, z, xj, yj, zj));
                        potential += charge_sorted[j] * gaussian_residual<Real>(r_ij, sigmas[i_depth]);
                    }
                }
            }
        }

        return potential;
    }

    template <typename Real>
    Real HPDMKPtTree<Real>::potential_target(Real x, Real y, Real z) {
        Real potential = 0;

        Real potential_window = potential_target_window(x, y, z);
        Real potential_difference = potential_target_difference(x, y, z);
        Real potential_residual = potential_target_residual(x, y, z);

        potential = potential_window + potential_difference + potential_residual;

        return potential;
    }
    
    template struct HPDMKPtTree<float>;
    template struct HPDMKPtTree<double>;
}