#include <hpdmk.h>
#include <tree.hpp>
#include <kernels.hpp>
#include <utils.hpp>
#include <nudft.hpp>
#include <upward_pass.hpp>

#include <vector>
#include <array>
#include <cmath>
#include <complex>
#include <algorithm>

#include <sctl.hpp>
#include <mpi.h>

namespace hpdmk {    

    template <class Tree>
    void outgoing_pw_proxy(Tree &tree, sctl::Vector<sctl::Vector<std::complex<typename Tree::float_type>>> &outgoing_pw) {
        upward_pass(tree, outgoing_pw);
    }

    template <class Tree>
    void outgoing_pw_root(Tree &tree, sctl::Vector<sctl::Vector<std::complex<typename Tree::float_type>>> &outgoing_pw) {

        using Real = typename Tree::float_type;

        auto &outgoing_pw_root = outgoing_pw[tree.root()];
        auto n_root = tree.n_window;
        auto d_root = 2 * n_root + 1;
        auto dk_root = tree.delta_k[0];
        
        sctl::Long n_particles = tree.charge_sorted.Dim();
        sctl::Vector<Real> x_scaled(n_particles), y_scaled(n_particles), z_scaled(n_particles);
        sctl::Vector<std::complex<Real>> c(n_particles);

        for (int i = 0; i < n_particles; ++i) {
            x_scaled[i] = tree.r_src_sorted[i * 3] * dk_root;
            y_scaled[i] = tree.r_src_sorted[i * 3 + 1] * dk_root;
            z_scaled[i] = tree.r_src_sorted[i * 3 + 2] * dk_root;
            c[i] = std::complex<Real>(tree.charge_sorted[i], 0);
        }
        
        nufft3d1(n_particles, &x_scaled[0], &y_scaled[0], &z_scaled[0], &c[0], -1, Real(std::pow(10.0, - tree.params.digits - 1)), d_root, d_root, d_root, &outgoing_pw_root[0]);
    }

    template <class Tree>
    void outgoing_pw_direct(Tree &tree, sctl::Vector<sctl::Vector<std::complex<typename Tree::float_type>>> &outgoing_pw) {
        // root node, always initialize with finufft
        auto &outgoing_pw_root = outgoing_pw[tree.root()];
        using Real = typename Tree::float_type;

        //temp vector for x, y, z
        sctl::Long n_particles = tree.charge_sorted.Dim();
        sctl::Vector<Real> x(n_particles), y(n_particles), z(n_particles), x_scaled(n_particles), y_scaled(n_particles), z_scaled(n_particles);
        sctl::Vector<std::complex<Real>> c(n_particles);

        for (int i = 0; i < n_particles; ++i) {
            x[i] = tree.r_src_sorted[i * 3];
            y[i] = tree.r_src_sorted[i * 3 + 1];
            z[i] = tree.r_src_sorted[i * 3 + 2];
            c[i] = std::complex<Real>(tree.charge_sorted[i], 0);
        }

        auto node_attr = tree.GetNodeAttr();
        for (int l = 2; l < tree.max_depth - 1; ++l) {
            auto dk = tree.delta_k[l];
            x_scaled = x * dk;
            y_scaled = y * dk;
            z_scaled = z * dk;

            for (auto i_node : tree.level_indices[l]) {
                int num_particles_i = tree.r_src_cnt_all[i_node];
                if (num_particles_i > 0) { // planewave of leaf nodes always need to be calculated
                    auto& i_node_particles = tree.node_particles[i_node];
                    assert(num_particles_i == i_node_particles.Dim());
                    
                    int c_offset = tree.charge_offset[i_node];
                    nudft3d1_halfplane(num_particles_i, &x_scaled[c_offset], &y_scaled[c_offset], &z_scaled[c_offset], &c[c_offset], -1, 2 * tree.n_diff + 1, 2 * tree.n_diff + 1, 2 * tree.n_diff + 1, &outgoing_pw[i_node][0]);
                }
            }
        }
    }

    template <typename Real>
    void HPDMKPtTree<Real>::form_outgoing_pw() {

        // initialize the outgoing planewave for the root node with nufft
        outgoing_pw_root(*this, outgoing_pw);

        if (params.init == DIRECT) {
            outgoing_pw_direct(*this, outgoing_pw);
        } else {
            outgoing_pw_proxy(*this, outgoing_pw);
        }
    }

    template <typename Real>
    void HPDMKPtTree<Real>::form_incoming_pw() {
        // only the nodes from level 2 
        auto node_attr = this->GetNodeAttr();

        const int dims = (2 * n_diff + 1) * (2 * n_diff + 1) * (n_diff + 1);

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

                        #pragma omp simd
                        for (int i = 0; i < dims; ++i) {
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

                            if (px == 0 && py == 0 && pz == 0) {
                                vec_addsub<Real, true, true>(dims, &incoming_pw_i[0], &outgoing_pw_j[0]);
                            } else {
                                auto shift_vec = shift_mat_l.select_shift_vec(px, py, pz);
                                vec_muladdsub<Real, true, true>(dims, &incoming_pw_i[0], &outgoing_pw_j[0], &shift_vec[0]);
                            }
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


template void hpdmk::outgoing_pw_direct(hpdmk::HPDMKPtTree<float> &tree, sctl::Vector<sctl::Vector<std::complex<float>>> &outgoing_pw);
template void hpdmk::outgoing_pw_direct(hpdmk::HPDMKPtTree<double> &tree, sctl::Vector<sctl::Vector<std::complex<double>>> &outgoing_pw);
template void hpdmk::outgoing_pw_proxy(hpdmk::HPDMKPtTree<float> &tree, sctl::Vector<sctl::Vector<std::complex<float>>> &outgoing_pw);
template void hpdmk::outgoing_pw_proxy(hpdmk::HPDMKPtTree<double> &tree, sctl::Vector<sctl::Vector<std::complex<double>>> &outgoing_pw);
template void hpdmk::outgoing_pw_root(hpdmk::HPDMKPtTree<float> &tree, sctl::Vector<sctl::Vector<std::complex<float>>> &outgoing_pw);
template void hpdmk::outgoing_pw_root(hpdmk::HPDMKPtTree<double> &tree, sctl::Vector<sctl::Vector<std::complex<double>>> &outgoing_pw);