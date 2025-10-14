#ifndef UPWARD_PASS_HPP
#define UPWARD_PASS_HPP

#include <complex>
#include <mdspan.hpp>
#include <sctl.hpp>

namespace hpdmk {

template <typename T, int DIM>
using ndview = std::experimental::mdspan<T, std::experimental::dextents<size_t, DIM>, std::experimental::layout_left>;

template <typename T>
void transform(int add_flag, const ndview<const T, 3> &fin, const ndview<const T, 2> &umat_, const ndview<T, 3> &fout,
               sctl::Vector<T> &workspace);

template <typename T>
void transform(int nvec, int add_flag, const ndview<const T, 4> &fin, const ndview<const T, 2> &umat,
               const ndview<T, 4> &fout, sctl::Vector<T> &workspace);

template <typename T>
void proxycharge2pw(const ndview<const T, 4> &proxy_coeffs, const ndview<const std::complex<T>, 2> &poly2pw,
                    const ndview<std::complex<T>, 4> &pw_expansion, sctl::Vector<T> &workspace);

template <typename T>
void charge2proxycharge(const ndview<const T, 2> &r_src_, const ndview<const T, 2> &charge_,
                        const ndview<const T, 1> &center, T scale_factor, const ndview<T, 4> &coeffs,
                        sctl::Vector<T> &workspace);

template <typename T>
sctl::Vector<std::complex<T>> calc_prox_to_pw(T boxsize, T hpw, int n_pw, int n_order);

template <class Tree>
sctl::Vector<sctl::Vector<std::complex<typename Tree::float_type>>> upward_pass(Tree &tree, int n_order);

} // namespace hpdmk

#endif
