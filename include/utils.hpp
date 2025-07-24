#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include <complex>

#include <hpdmk.h>
#include <sctl.hpp>

namespace hpdmk {
    
    inline bool isleaf(sctl::Tree<3>::NodeAttr node_attr) {
        if (node_attr.Leaf) {
            return true;
        } else {
            return false;
        }
    }

    template <typename Real>
    int periodic_shift(Real x_i, Real x_j, Real L, Real boxsize_i, Real boxsize_j) {
        if (x_i < boxsize_i && x_j > (L - boxsize_j)) {
            return 1; // i is at the left boundary, j is at the right boundary, shift x_i by L
        } else if (x_i > (L - boxsize_i) && x_j < boxsize_j) {
            return -1; // i is at the right boundary, j is at the left boundary, shift x_i by -L
        } else {
            return 0; // i and j are at the same side of the boundary
        }
    }

    template <typename T> // a cubic tensor is a rank-3 tensor of size d * d * d
    struct CubicTensor {
        int d; // dimension of the cubic tensor
        sctl::Vector<T> tensor;

        CubicTensor(int d, sctl::Vector<T> tensor) : d(d), tensor(tensor) {}
        CubicTensor() : d(0), tensor(sctl::Vector<T>(0)) {}

        inline int offset(int i, int j, int k) {
            return i * d * d + j * d + k;
        }

        inline T &value(int i, int j, int k) {
            return tensor[offset(i, j, k)];
        }
    };

    template <typename Real>
    Real dist2(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2) {
        return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
    }
}

#endif