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

        const HPDMKParams params;
        int n_digits;
        Real L;
        
        // cnt of src and charge should be the same, but in dmk they are set to be different
        sctl::Vector<Real> r_src_sorted;
        sctl::Vector<sctl::Long> r_src_cnt, r_src_offset; // number of source points and offset of source points in each node

        Real Q;
        sctl::Vector<Real> charge_sorted;
        sctl::Vector<sctl::Long> charge_cnt, charge_offset; // number of charges and offset of charges in each node

        // parameters for the PSWF kernel
        double c, lambda, C0;
        PolyFun<Real> real_poly, fourier_poly; // PSWF approximation functions for real and reciprocal space

        sctl::Vector<Real> delta_k, k_max; // delta k and the cutoff at each level
        sctl::Vector<sctl::Long> n_k; // number of Fourier modes needed at each level, total should be (2 * n_k[i] + 1) ^ 3
        sctl::Vector<Real> sigmas; // sigma for each level


        int max_depth; // maximum depth of the tree
        sctl::Vector<sctl::Vector<int>> level_indices; // store the indices of tree nodes in each level
        sctl::Vector<Real> boxsize; // store the size of the box
        sctl::Vector<Real> centers; // store the center location of each node, inner vector is [x, y, z]
        
        std::vector<Rank3Tensor<Real>> interaction_matrices; // store the interaction matrices for each level

        sctl::Vector<std::complex<Real>> kx_cache;
        sctl::Vector<std::complex<Real>> ky_cache;
        sctl::Vector<std::complex<Real>> kz_cache;

        std::vector<Rank3Tensor<std::complex<Real>>> plane_wave_coeffs; // store the plane wave coefficients for each node

        HPDMKPtTree(const sctl::Comm &comm, const HPDMKParams &params_, const sctl::Vector<Real> &r_src, const sctl::Vector<Real> &charge);

        int n_levels() const { return level_indices.Dim(); }
        std::size_t n_boxes() const { return this->GetNodeMID().Dim(); }

        sctl::Vector<sctl::Vector<sctl::Long>> node_particles; // store the indices of particles in each node, at most NlogN indices are stored
        std::vector<NodeNeighbors> neighbors; // store the neighbors of each node

        // shift from the center of node i_node to the center of node j_node (x_j - periodic_image(x_i))
        sctl::Vector<Real> node_shift(sctl::Long i_node, sctl::Long j_node);

        void init_wavenumbers();

        void init_interaction_matrices();

        sctl::Long root() { return level_indices[0][0]; }
        void collect_particles(sctl::Long i_node);

        void collect_neighbors(sctl::Long i_node);

        bool is_in_node(Real x, Real y, Real z, sctl::Long i_node);

        void init_planewave_coeffs();
        void init_planewave_coeffs_i(sctl::Long i_node, int n_k, Real delta_ki, Real k_max_i);

        // after init_planewave_coeffs, the energy can be calculated
        Real energy();

        Real window_energy();

        Real difference_energy(); // the difference kernel energy
        Real difference_energy_i(int i_depth, sctl::Long i_node); // self interaction energy of a single node
        Real difference_energy_ij(int i_depth, sctl::Long i_node, sctl::Long j_node); // interaction between two nodes i and j at the same depth

        Real residual_energy();
        Real residual_energy_i(int i_depth, sctl::Long i_node); // self interaction energy of a single node
        Real residual_energy_ij(int i_depth, sctl::Long i_node, sctl::Long j_node); // interaction energy between two nodes i and j

        Real window_energy_direct();
        Real residual_energy_direct();
        Real difference_energy_direct();
        Real difference_energy_direct_i(int i_depth, sctl::Long i_node);
        Real difference_energy_direct_ij(int i_depth, sctl::Long i_node, sctl::Long j_node);


        sctl::Vector<sctl::Long> path_to_target;
        void locate_target(Real x, Real y, Real z); // locate the node that the target point is in

        std::vector<Rank3Tensor<std::complex<Real>>> target_planewave_coeffs; // cache the plane wave coefficients for the target points
        void init_planewave_coeffs_target(Real x, Real y, Real z); // initialize the plane wave coefficients for the target point
        void init_planewave_coeffs_target_i(sctl::Long i_node, Real x, Real y, Real z); // initialize the plane wave coefficients for the target point

        Real potential_target(Real x, Real y, Real z); // calculate the potential at the target point
        Real potential_target_window(Real x, Real y, Real z); // calculate the potential at the target point using window function
        Real potential_target_difference(Real x, Real y, Real z); // calculate the potential at the target point using difference kernel

        Real potential_target_residual(Real x, Real y, Real z); // calculate the potential at the target point using residual kernel
        Real potential_target_residual_i(int i_depth, sctl::Long i_node, Real x, Real y, Real z); 
        Real potential_target_residual_ij(int i_depth, sctl::Long i_node, sctl::Long j_node, Real x, Real y, Real z);

        Real potential_target_window_direct(Real x, Real y, Real z); // calculate the potential at the target point of the window kernel
        Real potential_target_difference_direct(Real x, Real y, Real z); // calculate the potential at the target point using difference kernel
        Real potential_target_residual_direct(Real x, Real y, Real z); // calculate the potential at the target point using residual kernel
    };
}

#endif