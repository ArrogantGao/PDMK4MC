#include <hpdmk.h>
#include <tree.hpp>

#include <vector>
#include <array>
#include <cmath>
#include <complex>

#include <sctl.hpp>
#include <mpi.h>

namespace hpdmk {

    template <typename Real>
    HPDMKPtTree<Real>::HPDMKPtTree(const sctl::Comm &comm, const HPDMKParams &params_, const sctl::Vector<Real> &r_src, const sctl::Vector<Real> &charge) : sctl::PtTree<Real, 3>(comm), params(params_), n_digits(std::round(log10(1.0 / params_.eps) - 0.1)), L(params_.L), r_src(r_src), charge(charge) {
        sctl::Vector<Real> normalized_r_src = r_src / Real(L); // normalize the source points to the unit box

        const int n_src = charge.Dim();

        r_indices.ReInit(n_src);
        for (int i = 0; i < n_src; i++) {
            r_indices[i] = i;
        }

        constexpr bool balance21 = true; // Use "2-1" balancing for the tree, i.e. bordering boxes never more than one level away in depth
        constexpr int halo = 0;          // Only grab nearest neighbors as 'ghosts'
        constexpr bool use_periodic = true; // Use periodic boundary conditions

        std::cout << "adding particles" << std::endl;

        std::cout << "r_src: " << normalized_r_src.Dim() << std::endl;
        std::cout << "charge: " << charge.Dim() << std::endl;
        std::cout << "r_indices: " << r_indices.Dim() << std::endl;

        this->AddParticles("hpdmk_src", normalized_r_src);
        this->AddParticleData("hpdmk_charge", "hpdmk_src", charge);
        this->AddParticleData("hpdmk_r_indices", "hpdmk_src", r_indices);

        std::cout << "generating tree" << std::endl;

        this->UpdateRefinement(r_src, params.n_per_leaf, balance21, use_periodic, halo);

        this->template Broadcast<Real>("hpdmk_src");
        this->template Broadcast<Real>("hpdmk_charge");

        std::cout << "tree generated" << std::endl;

        int n_nodes = this->GetNodeMID().Dim();
        std::cout << "number of nodes: " << n_nodes << std::endl;

        this->GetData(r_indices, r_src_cnt, "hpdmk_r_indices");
        r_src_offset.ReInit(n_nodes);
        r_src_offset[0] = 0;
        for (int i = 1; i < n_nodes; i++) {
            r_src_offset[i] = r_src_offset[i - 1] + r_src_cnt[i - 1];
        }


    }

    template struct HPDMKPtTree<float>;
    template struct HPDMKPtTree<double>;
}