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

    template struct HPDMKPtTree<float>;
    template struct HPDMKPtTree<double>;
}