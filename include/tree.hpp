#ifndef TREE_HPP
#define TREE_HPP

#include <hpdmk.h>
#include <vector>
#include <array>
#include <cmath>
#include <complex>

#include <sctl.hpp>
#include <mpi.h>

#include <utils.hpp>
#include <kernels.hpp>
#include <pswf.hpp>

namespace hpdmk {

    typedef struct NodeNeighbors {
        sctl::Vector<sctl::Long> coarsegrain;
        sctl::Vector<sctl::Long> colleague;

        NodeNeighbors() {}
        NodeNeighbors(sctl::Vector<sctl::Long> coarsegrain, sctl::Vector<sctl::Long> colleague)
            : coarsegrain(coarsegrain), colleague(colleague) {}
    } NodeNeighbors;

    template <typename Real>
    struct HPDMKPtTree : public sctl::PtTree<Real, 3> {
        using float_type = Real;
        const HPDMKParams params;
        int n_digits;
        Real L;
        
        // cnt of src and charge should be the same, but in dmk they are set to be different
        sctl::Vector<Real> r_src_sorted;
        sctl::Vector<sctl::Long> r_src_cnt, r_src_offset, r_src_cnt_all; // number of source points and offset of source points in each node

        Real Q;
        sctl::Vector<Real> charge_sorted;
        sctl::Vector<sctl::Long> charge_cnt, charge_offset; // number of charges and offset of charges in each node

        sctl::Vector<Real> indices_map, indices_map_sorted, indices_invmap; // map the indices of the source points to the indices of the sorted source points
        sctl::Vector<sctl::Long> indices_map_cnt, indices_map_offset; // map the indices of the sorted source points to the indices of the source points

        // parameters for the PSWF kernel
        double c, lambda, C0;
        PolyFun<Real> real_poly, fourier_poly; // PSWF approximation functions for real and reciprocal space

        sctl::Vector<Real> delta_k, k_max; // delta k and the cutoff at each level
        int n_window, n_diff; // all difference kernels shares the same number of modes
        // sctl::Vector<sctl::Long> n_k; // number of Fourier modes needed at each level, total should be (2 * n_k[i] + 1) ^ 3
        sctl::Vector<Real> sigmas; // sigma for each level


        int max_depth; // maximum depth of the tree
        sctl::Vector<sctl::Vector<sctl::Long>> level_indices; // store the indices of tree nodes in each level
        sctl::Vector<Real> boxsize; // store the size of the box
        sctl::Vector<Real> centers; // store the center location of each node, inner vector is [x, y, z]

        sctl::Vector<sctl::Vector<sctl::Long>> node_particles; // store the indices of particles in each node, at most NlogN indices are stored
        sctl::Vector<NodeNeighbors> neighbors; // store the neighbors of each node
        
        sctl::Vector<sctl::Vector<Real>> interaction_mat; // store the interaction matrices for each level
        sctl::Vector<ShiftMatrix<Real>> shift_mat; // shift matrices, stored as vectors of phace factors in xyz directions
        sctl::Vector<sctl::Vector<std::complex<Real>>> incoming_pw, outgoing_pw;

        sctl::Vector<sctl::Long> path_to_origin, path_to_target;
        sctl::Vector<sctl::Vector<std::complex<Real>>> outgoing_pw_origin, outgoing_pw_target, phase_cache;

        HPDMKPtTree(const sctl::Comm &comm, const HPDMKParams &params_, const sctl::Vector<Real> &r_src, const sctl::Vector<Real> &charge);

        sctl::Long root() { return 0; }
        int n_levels() const { return level_indices.Dim(); }
        std::size_t n_boxes() const { return this->GetNodeMID().Dim(); }

        void collect_particles(sctl::Long i_node);
        void collect_neighbors(sctl::Long i_node);
        bool is_in_node(Real x, Real y, Real z, sctl::Long i_node);

        bool is_colleague(sctl::Long i_node, sctl::Long j_node);
        
        Real *r_src_ptr(sctl::Long i_node) {
            assert(r_src_cnt_all[i_node]);
            return &r_src_sorted[r_src_offset[i_node]];
        }
        Real *charge_ptr(sctl::Long i_node) {
            assert(r_src_cnt_all[i_node]);
            return &charge_sorted[charge_offset[i_node]];
        }

        // shift from the center of node i_node to the center of node j_node (x_j - periodic_image(x_i))
        sctl::Vector<Real> node_shift(sctl::Long i_node, sctl::Long j_node);

        void form_wavenumbers();
        void form_interaction_matrices();
        void form_shift_matrices();

        void form_outgoing_pw();
        void form_incoming_pw();

        Real eval_energy() {return eval_energy_window() + eval_energy_diff() + eval_energy_res();}
        Real eval_energy_window();
        Real eval_energy_diff();
        Real eval_energy_res();

        // residual energy evaluation for self interaction and neighbor interaction
        Real eval_energy_res_i(int i_depth, sctl::Long i_node);
        Real eval_energy_res_ij(int i_depth, sctl::Long i_node, sctl::Long j_node);

        // direct energy evaluation
        Real eval_energy_window_direct();
        Real eval_energy_diff_direct();
        Real eval_energy_res_direct();

        void locate_particle(sctl::Vector<sctl::Long>& path, Real x, Real y, Real z); // locate the node that the target point is in
        void form_outgoing_pw_single(sctl::Vector<sctl::Vector<std::complex<Real>>>& pw,sctl::Vector<sctl::Long>& path, Real x, Real y, Real z, Real q);

        Real eval_shift_energy(sctl::Long i_unsorted, Real dx, Real dy, Real dz);
        Real eval_shift_energy_window();
        Real eval_shift_energy_diff(sctl::Long i_particle);
        Real eval_shift_energy_res(sctl::Long i_particle, sctl::Vector<sctl::Long>& target_path, Real x, Real y, Real z, Real q);

        Real eval_shift_energy_res_i(sctl::Long i_node, int i_depth, sctl::Long i_particle, Real x, Real y, Real z, Real q);
        Real eval_shift_energy_res_ij(sctl::Long i_node, int i_depth, sctl::Long i_nbr, sctl::Long i_particle, Real x, Real y, Real z, Real q);

        void update_shift(sctl::Long i_particle_unsorted, Real dx, Real dy, Real dz); // if the shift is accepted, update the plane wave coefficients and the structure of the tree

        // void init_target_planewave_coeffs(std::vector<Rank3Tensor<std::complex<Real>>>& coeffs, sctl::Vector<sctl::Long>& path, Real x, Real y, Real z, Real q); // initialize the plane wave coefficients for the target point
        // void init_target_planewave_coeffs_i(Rank3Tensor<std::complex<Real>>& coeff, sctl::Long i_node, Real x, Real y, Real z, Real q); // initialize the plane wave coefficients for the target point

        // Real energy_shift(sctl::Long i_particle, Real dx, Real dy, Real dz);

        // Real window_energy_shift(std::vector<Rank3Tensor<std::complex<Real>>>& origin_coeffs, std::vector<Rank3Tensor<std::complex<Real>>>& target_coeffs); // calculate the potential at the target point using window function
        // Real difference_energy_shift(std::vector<Rank3Tensor<std::complex<Real>>>& origin_coeffs, sctl::Vector<sctl::Long>& origin_path, std::vector<Rank3Tensor<std::complex<Real>>>& target_coeffs, sctl::Vector<sctl::Long>& target_path); // calculate the potential at the target point using difference kernel

        // Real residual_energy_shift(sctl::Long i_particle, sctl::Vector<sctl::Long>& target_path, Real dx, Real dy, Real dz, Real q); // calculate the potential at the target point using residual kernel
        // Real residual_energy_shift_i(sctl::Long i_node, int i_depth, sctl::Long i_particle, Real x, Real y, Real z, Real q); // calculate the potential at the target point using residual kernel
        // Real residual_energy_shift_ij(sctl::Long i_node, int i_depth, sctl::Long i_nbr, sctl::Long i_particle, Real x, Real y, Real z, Real q); // calculate the potential at the target point using residual kernel

        // Real difference_energy_shift_direct(int i_depth, int i_particle, Real x, Real y, Real z);
        // Real residual_energy_shift_direct(int i_depth, Real x, Real y, Real z, Real q);
    };
}

#endif
