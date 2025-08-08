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

        // set all coeffs to 0
        auto &coeffs = plane_wave_coeffs[i_node];
        coeffs *= 0;

        for (auto i_particle : node_particles[i_node]) {
            Real q = charge_sorted[i_particle];
            Real x = r_src_sorted[i_particle * 3] - center_x;
            Real y = r_src_sorted[i_particle * 3 + 1] - center_y;
            Real z = r_src_sorted[i_particle * 3 + 2] - center_z;

            auto exp_ikx = std::exp( - std::complex<Real>(0, 1) * delta_k * x);
            auto exp_iky = std::exp( - std::complex<Real>(0, 1) * delta_k * y);
            auto exp_ikz = std::exp( - std::complex<Real>(0, 1) * delta_k * z);

            // #pragma omp parallel for
            for (int i = 0; i < 2 * n_k + 1; ++i) {
                int n = i - n_k;    
                kx_cache[i] = std::pow(exp_ikx, n);
                ky_cache[i] = std::pow(exp_iky, n);
                kz_cache[i] = std::pow(exp_ikz, n);
            }

            int d = 2 * n_k + 1;
            // #pragma omp parallel for
            for (int i = 0; i < d; ++i) {
                std::complex<Real> t1 = q * kx_cache[i];
                for (int j = 0; j < d; ++j) {
                    std::complex<Real> t2 = ky_cache[j] * t1;
                    for (int k = 0; k < d; ++k) {
                        // rho(kx, ky, kz) = sum(q_i * exp(i * (kx * x_i + ky * y_i + kz * z_i)))
                        coeffs[offset(i, j, k, d)] += t2 * kz_cache[k];
                    }
                }
            }
        }
    }

    template <typename Real>
    void HPDMKPtTree<Real>::init_planewave_coeffs_target_i(sctl::Long i_node, Real x, Real y, Real z) {
        auto &node_attr = this->GetNodeAttr();
        auto &node_mid = this->GetNodeMID();

        int i_depth = node_mid[i_node].Depth();
        int n_ki = n_k[i_depth];
        Real delta_ki = delta_k[i_depth];

        Real center_x = centers[i_node * 3];
        Real center_y = centers[i_node * 3 + 1];
        Real center_z = centers[i_node * 3 + 2];

        std::complex<Real> exp_ikx = std::exp( - std::complex<Real>(0, 1) * delta_ki * (x - center_x));
        std::complex<Real> exp_iky = std::exp( - std::complex<Real>(0, 1) * delta_ki * (y - center_y));
        std::complex<Real> exp_ikz = std::exp( - std::complex<Real>(0, 1) * delta_ki * (z - center_z));

        // #pragma omp parallel for
        for (int i = 0; i < 2 * n_ki + 1; ++i) {
            int n = i - n_ki;
            kx_cache[i] = std::pow(exp_ikx, n);
            ky_cache[i] = std::pow(exp_iky, n);
            kz_cache[i] = std::pow(exp_ikz, n);
        }

        auto &coeffs = target_planewave_coeffs[i_depth];
        coeffs *= 0;

        int d = 2 * n_ki + 1;
        // #pragma omp parallel for
        for (int i = 0; i < d; ++i) {
            std::complex<Real> t1 = kx_cache[i];
            for (int j = 0; j < d; ++j) {
                std::complex<Real> t2 = ky_cache[j] * t1;
                for (int k = 0; k < d; ++k) {
                    coeffs[offset(i, j, k, d)] += t2 * kz_cache[k];
                }
            }
        }
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
            // std::cout << "constructing plane wave coefficients at level " << i << " for node " << path_to_target[i] << " with depth " << int(node_mid[path_to_target[i]].Depth()) << std::endl;
            init_planewave_coeffs_target_i(path_to_target[i], x, y, z);
        }
    }

    template struct HPDMKPtTree<float>;
    template struct HPDMKPtTree<double>;
}