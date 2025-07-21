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
        sctl::Vector<int> fine;
        sctl::Vector<int> coarse;
        sctl::Vector<int> collegue;

        NodeNeighbors(sctl::Vector<int> fine, sctl::Vector<int> coarse, sctl::Vector<int> collegue) : fine(fine), coarse(coarse), collegue(collegue) {
        }

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
        sctl::Vector<std::complex<Real>> interaction_matrix; // the i-th matrix is of d[i] * d[i] * d[i] size
        
        InteractionMatrix(int d, sctl::Vector<std::complex<Real>> interaction_matrix) : d(d), interaction_matrix(interaction_matrix) {}

        std::complex<Real> offset(int i, int j, int k) {
            return interaction_matrix[i * d * d + j * d + k];
        }
    };

    template <typename Real>
    struct HPDMKPtTree : public sctl::PtTree<Real, 3> {

        const HPDMKParams params;
        int n_digits;
        Real L;

        sctl::Vector<Real> r_indices;
        sctl::Vector<Real> r_src;
        sctl::Vector<Real> charge;

        sctl::Vector<sctl::Long> r_src_cnt; // number of source points in each node
        sctl::Vector<sctl::Long> r_src_offset; // offset of source points in each node

        sctl::Vector<sctl::Vector<int>> level_indices; // store the indices of tree nodes in each level
        std::vector<NodeNeighbors> neighbors; // store the neighbors of each node

        sctl::Vector<Real> boxsize; // store the size of the box
        sctl::Vector<Real> centers; // store the center location of each node, inner vector is [x, y, z]

        std::vector<NodePlaneWaveCoeffs<Real>> plane_wave_coeffs; // store the plane wave coefficients for each node
        std::vector<InteractionMatrix<Real>> interaction_matrices; // store the interaction matrices for each level

        HPDMKPtTree(const sctl::Comm &comm, const HPDMKParams &params_, const sctl::Vector<Real> &r_src, const sctl::Vector<Real> &charge);

        int n_levels() const { return level_indices.Dim(); }
        std::size_t n_boxes() const { return this->GetNodeMID().Dim(); }

        void init_planewave_data();
    };
}

#endif