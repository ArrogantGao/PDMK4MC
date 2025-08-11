#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include <complex>

#include <hpdmk.h>
#include <sctl.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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

    inline int offset(int i, int j, int k, int d) {
        return i * d * d + j * d + k;
    }

    template <typename Real>
    inline Real dist2(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2) {
        return std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2) + std::pow(z1 - z2, 2);
    }

    template <typename Real>
    struct Rank3Tensor {
        int dx, dy, dz;
        int n;
        sctl::Vector<Real> tensor;

        Rank3Tensor(int dx, int dy, int dz) : dx(dx), dy(dy), dz(dz), n(dx * dy * dz), tensor(sctl::Vector<Real>(n)) {}

        inline int offset(int i, int j, int k) {
            return i * dy * dz + j * dz + k;
        }

        inline Real& operator()(int i, int j, int k) {
            return tensor[offset(i, j, k)];
        }

        inline Real& operator[](int idx) {
            return tensor[idx];
        }

        inline int Dim() {
            return n;
        }
    };

    std::vector<std::vector<double>> read_particle_info(const std::string& filename);

    namespace EwaldConst {
        static constexpr double EWALD_F = 1.12837917;
        static constexpr double EWALD_P = 0.3275911;
        static constexpr double A1 = 0.254829592;
        static constexpr double A2 = -0.284496736;
        static constexpr double A3 = 1.421413741;
        static constexpr double A4 = -1.453152027;
        static constexpr double A5 = 1.061405429;
    }
    
    // the erfc approch from LAMMPS, with 6 absolute and 4 relative precision on [0, 3] (erfc(3) ~ 1e-4)
    // on this region, this function is 4~5 times faster than std::erfc (2~3ns vs 11ns)
    template <typename Real>
    inline Real my_erfc(Real x) {
        Real expm2 = std::exp(-x * x);
        Real t = 1.0 / (1.0 + EwaldConst::EWALD_P * std::abs(x));
        Real res = t * (EwaldConst::A1 + t * (EwaldConst::A2 + t * (EwaldConst::A3 + t * (EwaldConst::A4 + t * EwaldConst::A5)))) * expm2;
        return res;
    }
}

#endif