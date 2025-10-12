#include <hpdmk.h>
#include <tree.hpp>
#include <kernels.hpp>
#include <utils.hpp>
#include <nudft.hpp>

#include <vector>
#include <array>
#include <cmath>
#include <complex>
#include <algorithm>

#include <sctl.hpp>
#include <mpi.h>

namespace hpdmk {    

    template <typename Real>
    void HPDMKPtTree<Real>::form_outgoing_pw() {
        // root node, always initialize with finufft
        auto &outgoing_pw_root = outgoing_pw[root()];

        //temp vector for x, y, z
        sctl::Long n_particles = charge_sorted.Dim();
        sctl::Vector<Real> x(n_particles), y(n_particles), z(n_particles), x_scaled(n_particles), y_scaled(n_particles), z_scaled(n_particles);
        sctl::Vector<std::complex<Real>> c(n_particles);

        for (int i = 0; i < n_particles; ++i) {
            x[i] = r_src_sorted[i * 3];
            y[i] = r_src_sorted[i * 3 + 1];
            z[i] = r_src_sorted[i * 3 + 2];
            c[i] = std::complex<Real>(charge_sorted[i], 0);
        }

        x_scaled = x * delta_k[0];
        y_scaled = y * delta_k[0];
        z_scaled = z * delta_k[0];

        nufft3d1(n_particles, &x_scaled[0], &y_scaled[0], &z_scaled[0], &c[0], -1, Real(params.nufft_eps), 2 * n_window + 1, 2 * n_window + 1, 2 * n_window + 1, &outgoing_pw_root[0]);

        auto node_attr = this->GetNodeAttr();
        for (int l = 2; l < max_depth - 1; ++l) {
            auto dk = delta_k[l];
            x_scaled = x * dk;
            y_scaled = y * dk;
            z_scaled = z * dk;

            for (auto i_node : level_indices[l]) {
                int num_particles_i = r_src_cnt_all[i_node];
                if (num_particles_i > 0) { // planewave of leaf nodes always need to be calculated
                    auto& i_node_particles = node_particles[i_node];
                    assert(num_particles_i == i_node_particles.Dim());
                    
                    int c_offset = charge_offset[i_node];
                    if (num_particles_i >= params.nufft_threshold) {
                        nufft3d1(num_particles_i, &x_scaled[c_offset], &y_scaled[c_offset], &z_scaled[c_offset], &c[c_offset], -1, Real(params.nufft_eps), 2 * n_diff + 1, 2 * n_diff + 1, 2 * n_diff + 1, &outgoing_pw[i_node][0]);
                    } else {
                        nudft3d1_halfplane(num_particles_i, &x_scaled[c_offset], &y_scaled[c_offset], &z_scaled[c_offset], &c[c_offset], -1, 2 * n_diff + 1, 2 * n_diff + 1, 2 * n_diff + 1, &outgoing_pw[i_node][0]);
                    }
                }
            }
        }
    }

    template <typename Real>
    void HPDMKPtTree<Real>::form_incoming_pw() {
        // only the nodes from level 2 
        auto node_attr = this->GetNodeAttr();

        for (int l = 2; l < max_depth - 1; ++l) {
            auto boxsize_l = boxsize[l];
            auto shift_mat_l = shift_mat[l];
            for (auto i_node : level_indices[l]) {
                // node with no-particles also need to be calculated
                // leaf nodes are skipped
                if (!isleaf(node_attr[i_node])) {
                    auto &incoming_pw_i = incoming_pw[i_node];
                    auto center_xi = centers[i_node * 3];
                    auto center_yi = centers[i_node * 3 + 1];
                    auto center_zi = centers[i_node * 3 + 2];

                    // clean the incoming pw
                    incoming_pw_i.SetZero();

                    assert(neighbors[i_node].colleague.Dim() == 26);
                    for (auto j_node : neighbors[i_node].colleague) {
                        auto &outgoing_pw_j = outgoing_pw[j_node];
                        auto center_xj = centers[j_node * 3];
                        auto center_yj = centers[j_node * 3 + 1];
                        auto center_zj = centers[j_node * 3 + 2];

                        // check if the two nodes need to be shifted along periodic boundary
                        int px, py, pz;
                        px = periodic_shift(center_xj, center_xi, L, boxsize_l, boxsize_l);
                        py = periodic_shift(center_yj, center_yi, L, boxsize_l, boxsize_l);
                        pz = periodic_shift(center_zj, center_zi, L, boxsize_l, boxsize_l);

                        // std::cout << "i: " << "(" << center_xi << ", " << center_yi << ", " << center_zi << ")" << ", j: " << "(" << center_xj << ", " << center_yj << ", " << center_zj << ")" << ", p: (" << px << ", " << py << ", " << pz << "), boxsize: " << boxsize_l << ", depth: "<< l << std::endl;

                        pw_shift(n_diff, incoming_pw_i, outgoing_pw_j, px, py, pz, shift_mat_l);
                    }
                }
            }
        }
    }

    // template <typename Real>
    // void HPDMKPtTree<Real>::init_planewave_coeffs() {

    //     // generate the root node coeffs
    //     sctl::Long root_node = root();
    //     init_planewave_coeffs_i(root_node, n_k[0], delta_k[0], k_max[0]);

    //     // from l = 2 to max_depth - 1
    //     for (int l = 2; l < max_depth - 1; ++l) {
    //         for (int j = 0; j < level_indices[l].Dim(); ++j) {
    //             sctl::Long i_node = level_indices[l][j];
    //             init_planewave_coeffs_i(i_node, n_k[l], delta_k[l], k_max[l]);
    //         }
    //     }
    // }

    // template <typename Real>
    // void HPDMKPtTree<Real>::init_planewave_coeffs_i(sctl::Long i_node, int n_k, Real delta_ki, Real k_max_i) {
    //     auto &node_attr = this->GetNodeAttr();
    //     auto &node_list = this->GetNodeLists();
    //     auto &node_mid = this->GetNodeMID();

    //     Real center_x = centers[i_node * 3];
    //     Real center_y = centers[i_node * 3 + 1];
    //     Real center_z = centers[i_node * 3 + 2];

    //     // set all coeffs to 0
    //     auto &coeffs = plane_wave_coeffs[i_node];
    //     coeffs *= 0;

    //     for (auto i_particle : node_particles[i_node]) {
    //         Real q = charge_sorted[i_particle];
    //         Real x = r_src_sorted[i_particle * 3] - center_x;
    //         Real y = r_src_sorted[i_particle * 3 + 1] - center_y;
    //         Real z = r_src_sorted[i_particle * 3 + 2] - center_z;

    //         auto exp_ikx = std::exp( - std::complex<Real>(0, 1) * delta_ki * x);
    //         auto exp_iky = std::exp( - std::complex<Real>(0, 1) * delta_ki * y);
    //         auto exp_ikz = std::exp( - std::complex<Real>(0, 1) * delta_ki * z);

    //         // #pragma omp parallel for
    //         for (int i = 0; i < n_k + 1; ++i) {    
    //             kx_cache[i] = std::pow(exp_ikx, i);
    //             ky_cache[i] = std::pow(exp_iky, i);
    //             kz_cache[i] = std::pow(exp_ikz, i);
    //         }

    //         apply_values<Real>(coeffs, kx_cache, ky_cache, kz_cache, n_k, delta_ki, k_max_i, q);
    //     }
    // }

    // template <typename Real>
    // void HPDMKPtTree<Real>::init_target_planewave_coeffs_i(Rank3Tensor<std::complex<Real>>& coeff, sctl::Long i_node, Real x, Real y, Real z, Real q) {
    //     auto &node_attr = this->GetNodeAttr();
    //     auto &node_mid = this->GetNodeMID();

    //     int i_depth = node_mid[i_node].Depth();
    //     int n_ki = n_k[i_depth];
    //     Real delta_ki = delta_k[i_depth];
    //     Real k_max_i = k_max[i_depth];

    //     Real center_x = centers[i_node * 3];
    //     Real center_y = centers[i_node * 3 + 1];
    //     Real center_z = centers[i_node * 3 + 2];

    //     std::complex<Real> exp_ikx = std::exp( - std::complex<Real>(0, 1) * delta_ki * (x - center_x));
    //     std::complex<Real> exp_iky = std::exp( - std::complex<Real>(0, 1) * delta_ki * (y - center_y));
    //     std::complex<Real> exp_ikz = std::exp( - std::complex<Real>(0, 1) * delta_ki * (z - center_z));

    //     // #pragma omp parallel for
    //     for (int i = 0; i < n_ki + 1; ++i) {
    //         kx_cache[i] = std::pow(exp_ikx, i);
    //         ky_cache[i] = std::pow(exp_iky, i);
    //         kz_cache[i] = std::pow(exp_ikz, i);
    //     }

    //     coeff *= 0; // clean the coeff

    //     apply_values<Real>(coeff, kx_cache, ky_cache, kz_cache, n_ki, delta_ki, k_max_i, q);
    // }

    // template <typename Real>
    // void HPDMKPtTree<Real>::init_target_planewave_coeffs(std::vector<Rank3Tensor<std::complex<Real>>>& coeffs, sctl::Vector<sctl::Long>& path, Real x, Real y, Real z, Real q) {
    //     auto &node_mid = this->GetNodeMID();
    //     auto &node_list = this->GetNodeLists();

    //     locate_particle(path, x, y, z);

    //     // 0-th level
    //     init_target_planewave_coeffs_i(coeffs[0], root(), x, y, z, q);

    //     // construct the plane wave coefficients from level 2 to level (depth of particle - 1)
    //     for (int i = 2; i < path.Dim() - 1; ++i) {
    //         init_target_planewave_coeffs_i(coeffs[i], path[i], x, y, z, q);
    //     }

    //     // the last level
    //     if (path.Dim() < max_depth) {
    //         init_target_planewave_coeffs_i(coeffs[path.Dim() - 1], path[path.Dim() - 1], x, y, z, q);
    //     }
    // }

    template struct HPDMKPtTree<float>;
    template struct HPDMKPtTree<double>;
}