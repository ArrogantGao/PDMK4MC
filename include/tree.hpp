#ifndef TREE_HPP
#define TREE_HPP

#include <hpdmk.h>
#include <vector>
#include <array>
#include <cmath>
#include <complex>

#include <sctl.hpp>
#include <mpi.h>

namespace hpdmk {

    typedef struct NodeNeighbors {
        sctl::Vector<sctl::Long> coarsegrain;
        sctl::Vector<sctl::Long> collegue;

        NodeNeighbors() {}
        NodeNeighbors(sctl::Vector<sctl::Long> coarsegrain, sctl::Vector<sctl::Long> collegue)
            : coarsegrain(coarsegrain), collegue(collegue) {}
    } NodeNeighbors;

    template <typename Real> 
    struct NodePlaneWaveCoeffs {
        sctl::Vector<std::complex<Real>> pw_kfx;
        sctl::Vector<std::complex<Real>> pw_kfy;
        sctl::Vector<std::complex<Real>> pw_kfz;

        sctl::Vector<std::complex<Real>> pw_kcx;
        sctl::Vector<std::complex<Real>> pw_kcy;
        sctl::Vector<std::complex<Real>> pw_kcz;

        sctl::Vector<std::complex<Real>> pw_kcgx;
        sctl::Vector<std::complex<Real>> pw_kcgy;
        sctl::Vector<std::complex<Real>> pw_kcgz;
    };

    template <typename Real>
    struct InteractionMatrix {
        int d; // dimension of the interaction matrix
        sctl::Vector<Real> interaction_matrix; // the i-th matrix is of d[i] * d[i] * d[i] size

        InteractionMatrix(int d, sctl::Vector<Real> interaction_matrix) : d(d), interaction_matrix(interaction_matrix) {}

        inline Real offset(int i, int j, int k) {
            return i * d * d + j * d + k;
        }
    };

    template <typename Real>
    struct HPDMKPtTree : public sctl::PtTree<Real, 3> {

        const HPDMKParams params;
        int n_digits;
        Real L;
        
        // cnt of src and charge should be the same, but in dmk they are set to be different
        sctl::Vector<Real> r_src_sorted;
        sctl::Vector<sctl::Long> r_src_cnt, r_src_offset; // number of source points and offset of source points in each node

        sctl::Vector<Real> charge_sorted;
        sctl::Vector<sctl::Long> charge_cnt, charge_offset; // number of charges and offset of charges in each node

        sctl::Vector<Real> delta_k, k_max; // delta k and the cutoff at each level
        sctl::Vector<sctl::Long> n_k; // number of Fourier modes needed at each level, total should be (2 * n_k[i] + 1) ^ 3
        sctl::Vector<Real> sigmas; // sigma for each level


        int max_depth; // maximum depth of the tree
        sctl::Vector<sctl::Vector<int>> level_indices; // store the indices of tree nodes in each level
        sctl::Vector<Real> boxsize; // store the size of the box
        sctl::Vector<Real> centers; // store the center location of each node, inner vector is [x, y, z]
        
        std::vector<InteractionMatrix<Real>> interaction_matrices; // store the interaction matrices for each level

        std::vector<NodePlaneWaveCoeffs<Real>> plane_wave_coeffs; // store the plane wave coefficients for each node

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

        void init_planewave_data();
    };
}

#endif