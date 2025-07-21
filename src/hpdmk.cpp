#include <hpdmk.h>
#include <iostream>
#include <tree.hpp>
#include <sctl.hpp>

namespace hpdmk {
    template <typename Real>
    inline void hpdmk_tree_create(MPI_Comm comm, HPDMKParams params, int n_src, const Real *r_src, const Real *charge) {
        const sctl::Comm sctl_comm(comm);

        sctl::Vector<Real> r_src_vec(n_src * 3, const_cast<Real *>(r_src), false);
        sctl::Vector<Real> charge_vec(n_src, const_cast<Real *>(charge), false);

        hpdmk::HPDMKPtTree<Real> tree(sctl_comm, params, r_src_vec, charge_vec);
    }
}

extern "C" {
    void hpdmk_tree_create(MPI_Comm comm, HPDMKParams params, int n_src, const double *r_src, const double *charge) {
        hpdmk::hpdmk_tree_create<double>(comm, params, n_src, r_src, charge);
    }
}