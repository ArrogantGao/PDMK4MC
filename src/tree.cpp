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
    inline bool isleaf(sctl::Tree<3>::NodeAttr node_attr) {
        if (node_attr.Leaf) {
            return true;
        } else {
            return false;
        }
    }

    template <typename Real>
    void HPDMKPtTree<Real>::init_wavenumbers() {
        sigmas.ReInit(max_depth + 1);
        Real sigma_0 = L / std::sqrt(std::log(1.0 / params.eps));
        sigmas[0] = sigma_0;
        for (int i = 1; i < max_depth + 1; ++i) {
            sigmas[i] = 0.5 * sigmas[i - 1];
        }

        // level 0 is not used, window + D_0 + D_1 are calculated together in level 1
        k_max.ReInit(max_depth + 1);
        n_k.ReInit(max_depth + 1);
        delta_k.ReInit(max_depth + 1);

        k_max[0] = k_max[1] = 8 * std::log(1.0 / params.eps) / L; // special for level 1, the kernel is erf(r / sigma_2) / r
        delta_k[0] = delta_k[1] = 2 * M_PI / L; // level 1 is a periodic, discrete Fourier summation
        n_k[0] = n_k[1] = std::ceil(k_max[1] / delta_k[1]);

        // for level 2 and above, the kernel is (erf(r / sigma_lp1) - erf(r / sigma_l)) / r
        for (int i = 2; i < max_depth + 1; ++i) {
            k_max[i] = 4 / boxsize[i] * std::log(1.0 / params.eps);
            delta_k[i] = 4 * M_PI / (3 * boxsize[i]);
            n_k[i] = std::ceil(k_max[i] / delta_k[i]);
        }

        #ifdef DEBUG
            std::cout << "sigmas: " << sigmas << std::endl;

            std::cout << "boxsize: " << boxsize << std::endl;
            std::cout << "k_max: " << k_max << std::endl;
            std::cout << "n_k: " << n_k << std::endl;
            std::cout << "delta_k: " << delta_k << std::endl;

            for (int i = 2; i < max_depth; ++i) {
                Real ef = std::exp(- sigmas[i] * sigmas[i] * k_max[i] * k_max[i] / 4) - std::exp(- sigmas[i + 1] * sigmas[i + 1] * k_max[i] * k_max[i] / 4) / (k_max[i] * k_max[i]) * 4 * M_PI;
                Real er = (std::erf(boxsize[i] / sigmas[i]) - std::erf(boxsize[i] / sigmas[i + 1])) / boxsize[i];
                std::cout << "level " << i << ": ef: " << ef << ", er: " << er << std::endl;
            }
        #endif
    }

    template <typename Real>
    void HPDMKPtTree<Real>::init_interaction_matrices() {
        // initialize the interaction matrices

        // W + D_0 + D_1, erf(r / sigma_2) / r 
        // 4 * pi * exp( -k^2 * sigma_2^2 / 4) / k^2
        auto window = gaussian_window_matrix<Real>(sigmas[2], delta_k[1], n_k[1], k_max[1]);
        interaction_matrices.push_back(window);

        // dummy interaction matrix for level 1, not used
        interaction_matrices.push_back(CubicTensor<Real>(1, sctl::Vector<Real>(0)));

        // D_l, (erf(r / sigma_lp1) - erf(r / sigma_l)) / r
        // 4 * pi * (exp( -k^2 * sigma_lp1^2 / 4) - exp( -k^2 * sigma_l^2 / 4)) / k^2
        // the finest level does not need to be calculated
        for (int l = 2; l < max_depth - 1; ++l) {
            auto D_l = gaussian_difference_matrix<Real>(sigmas[l], sigmas[l + 1], delta_k[l], n_k[l], k_max[l]);
            interaction_matrices.push_back(D_l);
        }
    }

    // this function will search from the leaf nodes to the root nodes (i_node) of the subtree
    template <typename Real>
    void HPDMKPtTree<Real>::collect_particles(sctl::Long i_node) {
        auto &node_attr = this->GetNodeAttr();
        auto &node_list = this->GetNodeLists();
        if (isleaf(node_attr[i_node])) {
            for (int i = 0; i < r_src_cnt[i_node]; ++i) {
                node_particles[i_node].PushBack(r_src_offset[i_node] + i);
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

        // collect the collegue neighbors
        for (int i = 0; i < 27; ++i) {
            sctl::Long i_nbr = node_list[i_node].nbr[i];
            if (i_nbr != -1 && i_nbr != i_node)
                neighbors[i_node].collegue.PushBack(i_nbr); // collegue neighbors are the neighbors that are at the same depth
        }

        if (neighbors[i_node].collegue.Dim() < 26) {
            // if the node has less than 26 collegue neighbors, it has coarsegrain neighbors
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

        #ifdef DEBUG
            std::cout << "generating tree with maximum number of particles per leaf: " << params.n_per_leaf << std::endl;
        #endif

        this->UpdateRefinement(normalized_r_src, params.n_per_leaf, balance21, use_periodic, halo);

        this->template Broadcast<Real>("hpdmk_src");
        this->template Broadcast<Real>("hpdmk_charge");

        // sort the sorted source points and charges
        r_src_sorted.ReInit(n_src * 3);
        charge_sorted.ReInit(n_src);

        Q = 0;
        for (int i = 0; i < n_src; ++i) {
            Q += charge[i] * charge[i];
        }

        int n_nodes = this->GetNodeMID().Dim();
        this->GetData(r_src_sorted, r_src_cnt, "hpdmk_src");
        this->GetData(charge_sorted, charge_cnt, "hpdmk_charge");

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
            r_src_offset[i] = r_src_offset[i - 1] + r_src_cnt[i - 1];
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
        max_depth++;
        level_indices.ReInit(max_depth);

        if (max_depth <= 2)
            throw std::runtime_error("max depth is less than 2, too low for hpdmk");

        for (int i_node = 0; i_node < n_nodes; ++i_node) {
            auto &node = node_mid[i_node];
            level_indices[node.Depth()].PushBack(i_node);
        }

        boxsize.ReInit(max_depth + 1);
        boxsize[0] = L;
        for (int i = 1; i < max_depth + 1; ++i)
            boxsize[i] = 0.5 * boxsize[i - 1];

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

        #ifdef DEBUG
            std::cout << "max depth: " << max_depth << std::endl;
            // check number of nodes and particles in each level
            for (int i_level = 0; i_level < max_depth; ++i_level) {
                int num_nodes = level_indices[i_level].Dim();
                int num_particles = 0;
                for (auto i_node : level_indices[i_level]) {
                    auto &node = node_mid[i_node];
                    num_particles += r_src_cnt[i_node];
                }
                std::cout << "level " << i_level << ": num_nodes: " << num_nodes << ", num_particles: " << num_particles << std::endl;
            }
        #endif

        init_wavenumbers();

        init_interaction_matrices();

        // store the indices of particles in each node
        node_particles.ReInit(n_nodes);
        collect_particles(root());

        #ifdef DEBUG
            // check the indices of particles in each node
            std::cout << "checking the indices of particles in each node" << std::endl;
            for (int i_node = 0; i_node < n_nodes; ++i_node) {
                auto &ids = node_particles[i_node];
                int depth = int(node_mid[i_node].Depth());
                for (int i = 0; i < ids.Dim(); ++i) {
                    Real x = r_src_sorted[ids[i] * 3];
                    Real y = r_src_sorted[ids[i] * 3 + 1];
                    Real z = r_src_sorted[ids[i] * 3 + 2];
                    assert(std::abs(x - centers[i_node * 3]) <= boxsize[depth] / 2);
                    assert(std::abs(y - centers[i_node * 3 + 1]) <= boxsize[depth] / 2);
                    assert(std::abs(z - centers[i_node * 3 + 2]) <= boxsize[depth] / 2);
                }
            }
            std::cout << "done" << std::endl;
        #endif

        // initialize the neighbors, only coarse-grained neighbors and collegue neighbors are stored
        neighbors.resize(n_nodes);
        for (int l = 2; l < max_depth; ++l) {
            for (auto i_node : level_indices[l]) {
                collect_neighbors(i_node);
            }
        }

        // allocate the memory for the plane wave coefficients
        plane_wave_coeffs.resize(n_nodes);

        // the coeffs related to the root node, all N particles
        plane_wave_coeffs[root()] = CubicTensor<std::complex<Real>>(2 * n_k[0] + 1, sctl::Vector<std::complex<Real>>(std::pow(2 * n_k[0] + 1, 3)));
        
        // from l = 2 to max_depth - 1, the finest level does not need to be calculated
        for (int l = 2; l < max_depth - 1; ++l) {
            for (auto i_node : level_indices[l]) {
                plane_wave_coeffs[i_node] = CubicTensor<std::complex<Real>>(2 * n_k[l] + 1, sctl::Vector<std::complex<Real>>(std::pow(2 * n_k[l] + 1, 3)));
            }
        }
    }

    template struct HPDMKPtTree<float>;
    template struct HPDMKPtTree<double>;

    template <typename Real>
    void HPDMKPtTree<Real>::init_planewave_coeffs() {
        // generate the root node coeffs
        sctl::Long root_node = root();
        init_planewave_coeffs_i(root_node, n_k[0], delta_k[0]);

        // from l = 2 to max_depth - 1
        for (int l = 2; l < max_depth - 1; ++l) {
            for (int j = 0; j < level_indices[l].Dim(); ++j) {
                sctl::Long i_node = level_indices[l][j];
                init_planewave_coeffs_i(i_node, n_k[l], delta_k[l]);
            }
        }
    }

    template <typename Real>
    void HPDMKPtTree<Real>::init_planewave_coeffs_i(sctl::Long i_node, int n_k, Real delta_k) {
        auto &node_attr = this->GetNodeAttr();
        auto &node_list = this->GetNodeLists();
        auto &node_mid = this->GetNodeMID();

        Real center_x = centers[i_node * 3];
        Real center_y = centers[i_node * 3 + 1];
        Real center_z = centers[i_node * 3 + 2];

        sctl::Vector<std::complex<Real>> kx_cache(2 * n_k + 1);
        sctl::Vector<std::complex<Real>> ky_cache(2 * n_k + 1);
        sctl::Vector<std::complex<Real>> kz_cache(2 * n_k + 1);

        // set all coeffs to 0
        plane_wave_coeffs[i_node].tensor *= 0;

        for (auto i_particle : node_particles[i_node]) {
            Real q = charge_sorted[i_particle];
            Real x = r_src_sorted[i_particle * 3] - center_x;
            Real y = r_src_sorted[i_particle * 3 + 1] - center_y;
            Real z = r_src_sorted[i_particle * 3 + 2] - center_z;

            auto exp_ikx = std::exp( - std::complex<Real>(0, 1) * delta_k * x);
            auto exp_iky = std::exp( - std::complex<Real>(0, 1) * delta_k * y);
            auto exp_ikz = std::exp( - std::complex<Real>(0, 1) * delta_k * z);

            #pragma omp parallel for
            for (int i = 0; i < 2 * n_k + 1; ++i) {
                int n = i - n_k;    
                kx_cache[i] = std::pow(exp_ikx, n);
                ky_cache[i] = std::pow(exp_iky, n);
                kz_cache[i] = std::pow(exp_ikz, n);
            }

            #pragma omp parallel for
            for (int i = 0; i < 2 * n_k + 1; ++i) {
                for (int j = 0; j < 2 * n_k + 1; ++j) {
                    for (int k = 0; k < 2 * n_k + 1; ++k) {
                        plane_wave_coeffs[i_node].value(i, j, k) += q * kx_cache[i] * ky_cache[j] * kz_cache[k];
                    }
                }
            }
        }
    }

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

        assert(root_coeffs.tensor.Dim() == window.tensor.Dim());

        #pragma omp parallel for reduction(+:energy)
        for (int i = 0; i < root_coeffs.tensor.Dim(); ++i) {
            energy += std::real(root_coeffs.tensor[i] * window.tensor[i] * std::conj(root_coeffs.tensor[i]));
        }

        energy *= 1 / (2 * std::pow(2*M_PI, 3)) * std::pow(delta_k[0], 3);

        // self energy
        Real sigma = sigmas[2];
        Real self_energy = Q / (std::sqrt(M_PI) * sigma);
        energy -= self_energy;

        #ifdef DEBUG
            std::cout << "window energy: " << energy << std::endl;
        #endif

        return energy;
    }

    // D_l for none-leaf nodes with depth >= 2
    template <typename Real>
    Real HPDMKPtTree<Real>::difference_energy() {
        Real energy = 0;

        return energy;
    }

    // energy of the residual term, for leaf nodes
    template <typename Real>
    Real HPDMKPtTree<Real>::residual_energy() {
        Real energy = 0;

        return energy;
    }
}