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
    void HPDMKPtTree<Real>::init_planewave_coeffs() {

        // generate the root node coeffs
        sctl::Long root_node = root();
        init_planewave_coeffs_i(root_node, n_k[0], delta_k[0], k_max[0]);

        // from l = 2 to max_depth - 1
        for (int l = 2; l < max_depth - 1; ++l) {
            for (int j = 0; j < level_indices[l].Dim(); ++j) {
                sctl::Long i_node = level_indices[l][j];
                init_planewave_coeffs_i(i_node, n_k[l], delta_k[l], k_max[l]);
            }
        }
    }

    template <typename Real>
    void HPDMKPtTree<Real>::init_planewave_coeffs_i(sctl::Long i_node, int n_k, Real delta_ki, Real k_max_i) {
        auto &node_attr = this->GetNodeAttr();
        auto &node_list = this->GetNodeLists();
        auto &node_mid = this->GetNodeMID();

        Real center_x = centers[i_node * 3];
        Real center_y = centers[i_node * 3 + 1];
        Real center_z = centers[i_node * 3 + 2];

        // set all coeffs to 0
        auto &coeffs = plane_wave_coeffs[i_node];
        coeffs *= 0;

        for (auto i_particle : node_particles[i_node]) {
            Real q = charge_sorted[i_particle];
            Real x = r_src_sorted[i_particle * 3] - center_x;
            Real y = r_src_sorted[i_particle * 3 + 1] - center_y;
            Real z = r_src_sorted[i_particle * 3 + 2] - center_z;

            auto exp_ikx = std::exp( - std::complex<Real>(0, 1) * delta_ki * x);
            auto exp_iky = std::exp( - std::complex<Real>(0, 1) * delta_ki * y);
            auto exp_ikz = std::exp( - std::complex<Real>(0, 1) * delta_ki * z);

            // #pragma omp parallel for
            for (int i = 0; i < n_k + 1; ++i) {    
                kx_cache[i] = std::pow(exp_ikx, i);
                ky_cache[i] = std::pow(exp_iky, i);
                kz_cache[i] = std::pow(exp_ikz, i);
            }

            apply_values<Real>(coeffs, kx_cache, ky_cache, kz_cache, n_k, delta_ki, k_max_i, q);
        }
    }

    template <typename Real>
    void HPDMKPtTree<Real>::init_planewave_coeffs_target_i(sctl::Long i_node, Real x, Real y, Real z) {
        auto &node_attr = this->GetNodeAttr();
        auto &node_mid = this->GetNodeMID();

        int i_depth = node_mid[i_node].Depth();
        int n_ki = n_k[i_depth];
        Real delta_ki = delta_k[i_depth];
        Real k_max_i = k_max[i_depth];

        Real center_x = centers[i_node * 3];
        Real center_y = centers[i_node * 3 + 1];
        Real center_z = centers[i_node * 3 + 2];

        std::complex<Real> exp_ikx = std::exp( - std::complex<Real>(0, 1) * delta_ki * (x - center_x));
        std::complex<Real> exp_iky = std::exp( - std::complex<Real>(0, 1) * delta_ki * (y - center_y));
        std::complex<Real> exp_ikz = std::exp( - std::complex<Real>(0, 1) * delta_ki * (z - center_z));

        // #pragma omp parallel for
        for (int i = 0; i < n_ki + 1; ++i) {
            kx_cache[i] = std::pow(exp_ikx, i);
            ky_cache[i] = std::pow(exp_iky, i);
            kz_cache[i] = std::pow(exp_ikz, i);
        }

        auto &coeffs = target_planewave_coeffs[i_depth];
        coeffs *= 0;

        apply_values<Real>(coeffs, kx_cache, ky_cache, kz_cache, n_ki, delta_ki, k_max_i, 1.0);
    }

    template <typename Real>
    void HPDMKPtTree<Real>::init_planewave_coeffs_target(Real x, Real y, Real z) {
        auto &node_mid = this->GetNodeMID();
        auto &node_list = this->GetNodeLists();

        locate_target(x, y, z);

        // 0-th level
        init_planewave_coeffs_target_i(root(), x, y, z);

        // construct the plane wave coefficients from level 2 to level (depth of particle - 1)
        for (int i = 2; i < path_to_target.Dim() - 1; ++i) {
            init_planewave_coeffs_target_i(path_to_target[i], x, y, z);
        }
    }

    template struct HPDMKPtTree<float>;
    template struct HPDMKPtTree<double>;
}