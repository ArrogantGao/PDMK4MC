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
    void HPDMKPtTree<Real>::form_wavenumbers() {

        c = prolc180(params.eps);
        lambda = prolate0_lambda(c);
        C0 = prolate0_int_eval(c, 1.0);
        
        real_poly = approximate_real_poly<Real>(params.eps, params.prolate_order);
        fourier_poly = approximate_fourier_poly<Real>(params.eps, params.prolate_order);

        sigmas.ReInit(max_depth + 1);
        Real sigma_0 = L / c;
        sigmas[0] = sigma_0;
        for (int i = 1; i < max_depth + 1; ++i) {
            sigmas[i] = 0.5 * sigmas[i - 1];
        }

        // level 0 is not used, window + D_0 + D_1 are calculated together in level 1
        k_max.ReInit(max_depth + 1);
        // n_k.ReInit(max_depth + 1);
        delta_k.ReInit(max_depth + 1);

        n_window = std::floor(2 * c / M_PI);
        n_diff = std::floor(3 * c / M_PI);

        k_max[0] = k_max[1] = 4 * c / L; // special for level 1, the kernel is int_prolate0(r / (L/4)) / r
        delta_k[0] = delta_k[1] = 2 * M_PI / L; // level 1 is a periodic, discrete Fourier summation

        // for level 2 and above, the kernel is (int_prolate0(r / (L/2^(i + 1))) - int_prolate0(r / (L/2^i))) / r
        for (int i = 2; i < max_depth + 1; ++i) {
            k_max[i] = 2 * c / boxsize[i];
            delta_k[i] = 2 * M_PI / (3 * boxsize[i]);
            // n_k[i] = std::ceil(k_max[i] / delta_k[i]);
        }

        #ifdef DEBUG
            std::cout << "c: " << c << std::endl;
            std::cout << "lambda: " << lambda << std::endl;
            std::cout << "C0: " << C0 << std::endl;

            std::cout << "sigmas: " << sigmas << std::endl;

            std::cout << "boxsize: " << boxsize << std::endl;
            std::cout << "k_max: " << k_max << std::endl;
            std::cout << "delta_k: " << delta_k << std::endl;
        #endif
    }

    template <typename Real>
    void HPDMKPtTree<Real>::form_interaction_matrices() {
        // initialize the interaction matrices

        // W + D_0 + D_1
        auto window = window_matrix<Real>(fourier_poly, sigmas[2], delta_k[1], n_window);
        interaction_mat.PushBack(window);

        // dummy interaction matrix for level 1, not used
        interaction_mat.PushBack(sctl::Vector<Real>());

        // D_l
        // the finest level does not need to be calculated
        for (int l = 2; l < max_depth - 1; l++) {
            auto D_l = difference_matrix<Real>(fourier_poly, sigmas[l], sigmas[l + 1], delta_k[l], n_diff);
            interaction_mat.PushBack(D_l);
        }
    }

    template <typename Real>
    void HPDMKPtTree<Real>::form_shift_matrices() {
        // initialize the shift matrices

        ShiftMatrix<Real> dummy_0, dummy_1;
        shift_mat.PushBack(dummy_0);
        shift_mat.PushBack(dummy_1);

        for (int l = 2; l < max_depth - 1; l++) {
            shift_mat.PushBack(ShiftMatrix<Real>(n_diff, delta_k[l], L));
        }
    }

    // this function will search from the leaf nodes to the root nodes (i_node) of the subtree
    template <typename Real>
    void HPDMKPtTree<Real>::collect_particles(sctl::Long i_node) {
        auto &node_attr = this->GetNodeAttr();
        auto &node_list = this->GetNodeLists();
        if (isleaf(node_attr[i_node])) {
            for (int i = 0; i < r_src_cnt[i_node]; ++i) {
                node_particles[i_node].PushBack(charge_offset[i_node] + i);
            }
        } else {
            for (int i = 0; i < 8; ++i) {
                sctl::Long i_child = node_list[i_node].child[i];
                collect_particles(i_child);
                for (int j = 0; j < node_particles[i_child].Dim(); ++j) {
                    node_particles[i_node].PushBack(node_particles[i_child][j]);
                }
            }
        }
    }

    template <typename Real>
    void HPDMKPtTree<Real>::collect_neighbors(sctl::Long i_node) {
        auto &node_attr = this->GetNodeAttr();
        auto &node_list = this->GetNodeLists();

        // collect the colleague neighbors
        for (int i = 0; i < 27; ++i) {
            sctl::Long i_nbr = node_list[i_node].nbr[i];
            if (i_nbr != -1 && i_nbr != i_node)
                neighbors[i_node].colleague.PushBack(i_nbr); // colleague neighbors are the neighbors that are at the same depth
        }

        if (neighbors[i_node].colleague.Dim() < 26) {
            // if the node has less than 26 colleague neighbors, it has coarsegrain neighbors
            sctl::Long i_parent = node_list[i_node].parent;
            // collect the coarsegrain neighbors, which are the neighbors of its parent
            auto &parent_nbr = node_list[i_parent].nbr;

            auto shift_i2p = node_shift(i_parent, i_node);

            for (int i = 0; i < 27; ++i) {
                sctl::Long i_nbr = parent_nbr[i];
                if (i_nbr != -1 && i_nbr != i_parent && isleaf(node_attr[i_nbr])) {
                    auto shift_n2p = node_shift(i_parent, i_nbr);
                    if (shift_i2p[0] * shift_n2p[0] >= 0 && shift_i2p[1] * shift_n2p[1] >= 0 && shift_i2p[2] * shift_n2p[2] >= 0) {
                        neighbors[i_node].coarsegrain.PushBack(i_nbr);
                    }
                }
            }
        }
    }

    // shift from the center of node i_node to the center of node j_node (x_j - periodic_image(x_i))
    template <typename Real>
    sctl::Vector<Real> HPDMKPtTree<Real>::node_shift(sctl::Long i_node, sctl::Long j_node) {
        auto &node_mid = this->GetNodeMID();
        auto &node_attr = this->GetNodeAttr();
        auto &node_list = this->GetNodeLists();

        sctl::Long i_depth = node_mid[i_node].Depth();
        sctl::Long j_depth = node_mid[j_node].Depth();

        // difference of depth of the nodes i_node and j_node at most 1
        assert(std::abs(i_depth - j_depth) <= 1);

        sctl::Vector<Real> shift(3);

        Real x_i = centers[i_node * 3];
        Real y_i = centers[i_node * 3 + 1];
        Real z_i = centers[i_node * 3 + 2];
        Real x_j = centers[j_node * 3];
        Real y_j = centers[j_node * 3 + 1];
        Real z_j = centers[j_node * 3 + 2];

        // to see if i and j are at the oppsite side of the same boundary
        // if true, add L to the shift the i-th coordinates
        int px, py, pz;
        px = periodic_shift(x_i, x_j, L, boxsize[i_depth], boxsize[j_depth]);
        py = periodic_shift(y_i, y_j, L, boxsize[i_depth], boxsize[j_depth]);
        pz = periodic_shift(z_i, z_j, L, boxsize[i_depth], boxsize[j_depth]);

        shift[0] = x_j - (x_i + px * L);
        shift[1] = y_j - (y_i + py * L);
        shift[2] = z_j - (z_i + pz * L);

        return shift;
    }

    template <typename Real>
    void HPDMKPtTree<Real>::locate_particle(sctl::Vector<sctl::Long>& path, Real x, Real y, Real z) {
        auto &node_mid = this->GetNodeMID();
        auto &node_attr = this->GetNodeAttr();
        auto &node_list = this->GetNodeLists();

        path.ReInit(0);
        sctl::Long node_0 = root();
        path.PushBack(node_0);

        while (true) {
            if (isleaf(node_attr[node_0])) {
                return ;
            } else {
                int depth = node_mid[node_0].Depth();
                for (int i = 0; i < 8; ++i) {
                    sctl::Long i_child = node_list[node_0].child[i];
                    Real center_x = centers[i_child * 3];
                    Real center_y = centers[i_child * 3 + 1];
                    Real center_z = centers[i_child * 3 + 2];

                    Real shift_x = std::abs(x - center_x);
                    Real shift_y = std::abs(y - center_y);
                    Real shift_z = std::abs(z - center_z);

                    if (shift_x <= boxsize[depth + 1] / 2 && shift_y <= boxsize[depth + 1] / 2 && shift_z <= boxsize[depth + 1] / 2) {
                        path.PushBack(i_child);
                        node_0 = i_child;
                        break;
                    }
                }
            }
        }
    }

    template <typename Real>
    bool HPDMKPtTree<Real>::is_in_node(Real x, Real y, Real z, sctl::Long i_node) {
        auto &node_mid = this->GetNodeMID();

        sctl::Long i_depth = node_mid[i_node].Depth();
        Real center_x = centers[i_node * 3];
        Real center_y = centers[i_node * 3 + 1];
        Real center_z = centers[i_node * 3 + 2];

        Real shift_x = std::abs(x - center_x);
        Real shift_y = std::abs(y - center_y);
        Real shift_z = std::abs(z - center_z);

        return (shift_x <= boxsize[i_depth] / 2 && shift_y <= boxsize[i_depth] / 2 && shift_z <= boxsize[i_depth] / 2);
    }

    // in current implementation, mpi is not supported yet
    template <typename Real>
    HPDMKPtTree<Real>::HPDMKPtTree(const sctl::Comm &comm, const HPDMKParams &params_, const sctl::Vector<Real> &r_src, const sctl::Vector<Real> &charge) : sctl::PtTree<Real, 3>(comm), params(params_), n_digits(std::round(log10(1.0 / params_.eps) - 0.1)), L(params_.L){
        sctl::Vector<Real> normalized_r_src = r_src / Real(L); // normalize the source points to the unit box

        const int n_src = charge.Dim();

        constexpr bool balance21 = true; // Use "2-1" balancing for the tree, i.e. bordering boxes never more than one level away in depth
        constexpr int halo = 0;          // Only grab nearest neighbors as 'ghosts'
        constexpr bool use_periodic = true; // Use periodic boundary conditions

        #ifdef DEBUG
            std::cout << "adding particles" << std::endl;
            std::cout << "r_src: " << normalized_r_src.Dim() << std::endl;
            std::cout << "charge: " << charge.Dim() << std::endl;
        #endif

        this->AddParticles("hpdmk_src", normalized_r_src);
        this->AddParticleData("hpdmk_charge", "hpdmk_src", charge);

        indices_map.ReInit(n_src);
        for (int i = 0; i < n_src; ++i) {
            indices_map[i] = i;
        }
        this->AddParticleData("hpdmk_indices", "hpdmk_src", indices_map);

        #ifdef DEBUG
            std::cout << "generating tree with maximum number of particles per leaf: " << params.n_per_leaf << std::endl;
        #endif

        this->UpdateRefinement(normalized_r_src, params.n_per_leaf, balance21, use_periodic, halo);

        this->template Broadcast<Real>("hpdmk_src");
        this->template Broadcast<Real>("hpdmk_charge");

        // sort the sorted source points and charges
        r_src_sorted.ReInit(n_src * 3);
        charge_sorted.ReInit(n_src);
        indices_map_sorted.ReInit(n_src);

        Q = 0;
        for (int i = 0; i < n_src; ++i) {
            Q += charge[i] * charge[i];
        }

        int n_nodes = this->GetNodeMID().Dim();
        this->GetData(r_src_sorted, r_src_cnt, "hpdmk_src");
        this->GetData(charge_sorted, charge_cnt, "hpdmk_charge");
        this->GetData(indices_map_sorted, indices_map_cnt, "hpdmk_indices");

        // build the indices_invmap
        indices_invmap.ReInit(n_src);
        for (int i = 0; i < n_src; ++i) {
            indices_invmap[indices_map_sorted[i]] = i;
        }

        // rescale the r_src
        r_src_sorted *= L;

        #ifdef DEBUG
            std::cout << "tree generated with number of nodes: " << n_nodes << std::endl;
        #endif

        r_src_offset.ReInit(n_nodes);
        charge_offset.ReInit(n_nodes);
        r_src_offset[0] = 0;
        charge_offset[0] = 0;
        for (int i = 1; i < n_nodes; i++) {
            assert(r_src_cnt[i] == charge_cnt[i]); // check if the number of source points and charges are the same, they have to be the same
            r_src_offset[i] = r_src_offset[i - 1] + 3 * r_src_cnt[i - 1];
            charge_offset[i] = charge_offset[i - 1] + charge_cnt[i - 1];
        }

        auto &node_mid = this->GetNodeMID();
        auto &node_lst = this->GetNodeLists();
        auto &node_attr = this->GetNodeAttr();

        // initialize the level_indices
        max_depth = 0;
        for (int i_node = 0; i_node < n_nodes; ++i_node) {
            auto &node = node_mid[i_node];
            max_depth = std::max(int(node.Depth()), max_depth);
        }
        max_depth++; // fuck, why did I add 1 here? it is actually number of levels
        level_indices.ReInit(max_depth);

        if (max_depth <= 2)
            throw std::runtime_error("max depth is less than 2, too low for hpdmk");

        for (int i_node = 0; i_node < n_nodes; ++i_node) {
            auto &node = node_mid[i_node];
            level_indices[node.Depth()].PushBack(i_node);
        }

        // std::cout << "max_depth: " << max_depth << std::endl;
        // std::cout << "n_levels: " << n_levels() << std::endl;
        // std::cout << "level indices dim: " << level_indices.Dim() << std::endl;

        boxsize.ReInit(max_depth + 1);
        boxsize[0] = L;
        for (int i = 1; i < max_depth + 1; ++i)
            boxsize[i] = 0.5 * boxsize[i - 1];

        // std::cout << "boxsize[end]: " << boxsize[max_depth] << std::endl;

        Real scale = 1.0;
        centers.ReInit(n_nodes * 3);
        for (int i_level = 0; i_level < n_levels(); ++i_level) {
            for (auto i_node : level_indices[i_level]) {
                auto &node = node_mid[i_node];
                auto node_origin = node.template Coord<Real>();
                for (int i = 0; i < 3; ++i)
                    centers[i_node * 3 + i] = L * (node_origin[i] + 0.5 * scale);
            }
            scale *= 0.5;
        }

        form_wavenumbers();
        // std::cout << "wavenumbers formed" << std::endl;

        form_interaction_matrices();
        // std::cout << "interaction matrices formed" << std::endl;

        form_shift_matrices();
        // std::cout << "shift matrices formed" << std::endl;

        // store the indices of particles in each node
        node_particles.ReInit(n_nodes);
        collect_particles(root());
        // std::cout << "particles collected" << std::endl;

        r_src_cnt_all.ReInit(n_nodes);
        for (int i_node = 0; i_node < n_nodes; ++i_node) {
            auto n_i = node_particles[i_node].Dim();
            r_src_cnt_all[i_node] = n_i;
        }
        // std::cout << "r_src_cnt and charge_cnt updated" << std::endl;

        // initialize the neighbors, only coarse-grained neighbors and colleague neighbors are stored
        neighbors.ReInit(n_nodes);
        for (int l = 2; l < max_depth; ++l) {
            for (auto i_node : level_indices[l]) {
                collect_neighbors(i_node);
            }
        }
        // std::cout << "neighbors collected" << std::endl;

        // allocate the memory for the plane wave coefficients
        outgoing_pw.ReInit(n_nodes);
        incoming_pw.ReInit(n_nodes);

        outgoing_pw_origin.ReInit(max_depth);
        outgoing_pw_target.ReInit(max_depth);

        // the coeffs related to the root node, all N particles
        int d_window = 2 * n_window + 1;
        incoming_pw[root()] = sctl::Vector<std::complex<Real>>(d_window * d_window * d_window);
        outgoing_pw[root()] = sctl::Vector<std::complex<Real>>(d_window * d_window * d_window);
        outgoing_pw_origin[0] = sctl::Vector<std::complex<Real>>(d_window * d_window * d_window);
        outgoing_pw_target[0] = sctl::Vector<std::complex<Real>>(d_window * d_window * d_window);
        
        // from l = 2 to max_depth - 1, the finest level does not need to be calculated
        int d_diff = 2 * n_diff + 1;
        for (int l = 2; l < max_depth - 1; ++l) {
            for (auto i_node : level_indices[l]) {
                incoming_pw[i_node] = sctl::Vector<std::complex<Real>>(d_diff * d_diff * d_diff);
                outgoing_pw[i_node] = sctl::Vector<std::complex<Real>>(d_diff * d_diff * d_diff);
            }
            outgoing_pw_origin[l] = sctl::Vector<std::complex<Real>>(d_diff * d_diff * d_diff);
            outgoing_pw_target[l] = sctl::Vector<std::complex<Real>>(d_diff * d_diff * d_diff);
        }
    }

    // template <typename Real>
    // void HPDMKPtTree<Real>::update_shift(sctl::Long i_particle, Real dx, Real dy, Real dz) {
        
    //     // update the source points
    //     Real x_o = r_src_sorted[i_particle * 3];
    //     Real y_o = r_src_sorted[i_particle * 3 + 1];
    //     Real z_o = r_src_sorted[i_particle * 3 + 2];
    //     r_src_sorted[i_particle * 3] = my_mod(x_o + dx, L);
    //     r_src_sorted[i_particle * 3 + 1] = my_mod(y_o + dy, L);
    //     r_src_sorted[i_particle * 3 + 2] = my_mod(z_o + dz, L);

    //     //update the window level
    //     auto &root_coeffs = plane_wave_coeffs[root()];
    //     for (int i = 0; i < root_coeffs.Dim(); ++i) {
    //         root_coeffs[i] += target_planewave_coeffs[0][i] - origin_planewave_coeffs[0][i];
    //     }

    //     //update the difference levels
    //     for (int l = 2; l < path_to_origin.Dim() - 1; ++l) {
    //         auto node_origin = path_to_origin[l];
    //         auto &node_coeffs = plane_wave_coeffs[node_origin];
    //         for (int i = 0; i < node_coeffs.Dim(); ++i) {
    //             node_coeffs[i] -= origin_planewave_coeffs[l][i];
    //         }
    //     }
    //     if (path_to_origin.Dim() < max_depth) {
    //         auto node_origin = path_to_origin[path_to_origin.Dim() - 1];
    //         auto &node_coeffs = plane_wave_coeffs[node_origin];
    //         for (int i = 0; i < node_coeffs.Dim(); ++i) {
    //             node_coeffs[i] -= origin_planewave_coeffs[path_to_origin.Dim() - 1][i];
    //         }
    //     }

    //     for (int l = 2; l < path_to_target.Dim() - 1; ++l) {
    //         auto node_target = path_to_target[l];
    //         auto &node_coeffs = plane_wave_coeffs[node_target];
    //         for (int i = 0; i < node_coeffs.Dim(); ++i) {
    //             node_coeffs[i] += target_planewave_coeffs[l][i];
    //         }
    //     }
    //     if (path_to_target.Dim() < max_depth) {
    //         auto node_target = path_to_target[path_to_target.Dim() - 1];
    //         auto &node_coeffs = plane_wave_coeffs[node_target];
    //         for (int i = 0; i < node_coeffs.Dim(); ++i) {
    //             node_coeffs[i] += target_planewave_coeffs[path_to_target.Dim() - 1][i];
    //         }
    //     }

    //     // update particle lists
    //     for (int l = 2; l < path_to_origin.Dim(); ++l) {
    //         auto node_origin = path_to_origin[l];
    //         auto &node_particles_origin = node_particles[node_origin];
    //         remove_particle(node_particles_origin, i_particle);
    //     }
    //     for (int l = 2; l < path_to_target.Dim(); ++l) {
    //         auto node_target = path_to_target[l];
    //         auto &node_particles_target = node_particles[node_target];
    //         node_particles_target.PushBack(i_particle);
    //     }
    // }

    template struct HPDMKPtTree<float>;
    template struct HPDMKPtTree<double>;
}