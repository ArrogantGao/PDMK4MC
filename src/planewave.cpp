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
                    if (node_particles[i_node].Dim() > 0) {
                        auto &outgoing_pw_i = outgoing_pw[i_node];
                        int d = 2 * n_diff + 1;
                        for (int i = 0; i < d * d * (n_diff + 1); ++i) {
                            incoming_pw_i[i] = std::conj(outgoing_pw_i[i]);
                        }
                    } else {
                        incoming_pw_i.SetZero();
                    }

                    assert(neighbors[i_node].colleague.Dim() == 26);
                    for (auto j_node : neighbors[i_node].colleague) {
                        if (node_particles[j_node].Dim() > 0) {
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
    }

    template <typename Real>
    void HPDMKPtTree<Real>::form_outgoing_pw_single(sctl::Vector<sctl::Vector<std::complex<Real>>>& pw, sctl::Vector<sctl::Long>& path, Real x, Real y, Real z, Real q) {
        auto path_depth = path.Dim();

        // #pragma omp parallel for
        for (int i = 0; i < path_depth; ++i) {
            int n_k;
            Real d_k;
            if (i == 0){
                n_k = n_window;
                d_k = delta_k[0];
            } else if ((i == 1) || (i == max_depth - 1)){
                continue;
            } else{
                n_k = n_diff;
                d_k = delta_k[i];
            }

            int d = 2 * n_k + 1;
            auto &cache = phase_cache[i];
            auto &outgoing_pw_i = pw[i];

            nudft3d1_single_halfplane(d_k * x, d_k * y, d_k * z, complex<Real>(q, 0), -1, d, d, d, &cache[0], &cache[d], &cache[2 * d], &outgoing_pw_i[0]);
        }
    }

    template struct HPDMKPtTree<float>;
    template struct HPDMKPtTree<double>;
}