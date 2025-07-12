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
    template <typename Real, int DIM>
    struct DMKPtTree : public sctl::PtTree<Real, DIM> {
        
    };
}

#endif