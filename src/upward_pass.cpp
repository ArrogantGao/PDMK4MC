#include <tree.hpp>
#include <upward_pass.hpp>

template sctl::Vector<float> hpdmk::upward_pass(hpdmk::HPDMKPtTree<float> &tree, int n_order);
template sctl::Vector<double> hpdmk::upward_pass(hpdmk::HPDMKPtTree<double> &tree, int n_order);
