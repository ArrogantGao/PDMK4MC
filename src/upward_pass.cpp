#include <tree.hpp>
#include <upward_pass.hpp>

template sctl::Vector<sctl::Vector<std::complex<float>>> hpdmk::upward_pass(hpdmk::HPDMKPtTree<float> &tree,
                                                                            int n_order);
template sctl::Vector<sctl::Vector<std::complex<double>>> hpdmk::upward_pass(hpdmk::HPDMKPtTree<double> &tree,
                                                                             int n_order);
