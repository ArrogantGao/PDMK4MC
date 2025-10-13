#ifndef UPWARD_PASS_HPP
#define UPWARD_PASS_HPP

#include <chebychev.hpp>
#include <gemm.hpp>
#include <mdspan.hpp>
#include <omp.h>
#include <sctl.hpp>

namespace hpdmk {

template <typename T, int DIM>
using ndview = std::experimental::mdspan<T, std::experimental::dextents<size_t, DIM>, std::experimental::layout_left>;

template <class Tree>
sctl::Vector<sctl::Vector<int>> get_level_indices(const Tree &tree) {
    const auto &node_mid = tree.GetNodeMID();
    sctl::Vector<sctl::Vector<int>> level_indices;

    level_indices.ReInit(SCTL_MAX_DEPTH);
    int8_t max_depth = 0;
    for (int i_node = 0; i_node < node_mid.Dim(); ++i_node) {
        auto &node = node_mid[i_node];
        level_indices[node.Depth()].PushBack(i_node);
        max_depth = std::max(node.Depth(), max_depth);
    }
    max_depth++;

    level_indices.ReInit(max_depth);
    return level_indices;
}

template <typename T>
void transform(int add_flag, const ndview<const T, 3> &fin, const ndview<const T, 2> &umat_, const ndview<T, 3> &fout,
               sctl::Vector<T> &workspace) {
    using hpdmk::gemm::gemm;
    const int nin = fin.extent(0);
    const int nout = fout.extent(0);
    const int nin2 = nin * nin;
    const int noutnin = nout * nin;
    const int nout2 = nout * nout;

    ndview<const T, 2> umat(umat_.data_handle(), nout * nin, 3);
    workspace.ReInit(2 * nin * nin * nout + nout * nout * nin);
    ndview<T, 3> ff(&workspace[0], nin, nin, nout);
    ndview<T, 3> fft(ff.data_handle() + ff.size(), nout, nout, nin);
    ndview<T, 3> ff2(fft.data_handle() + fft.size(), nout, nout, nin);

    // transform in z
    gemm('n', 't', nin2, nout, nin, T{1.0}, fin.data_handle(), nin2, umat.data_handle() + 2 * nout * nin, nout, T{0.0},
         ff.data_handle(), nin2);

    for (int k = 0; k < nin; ++k)
        for (int j = 0; j < nout; ++j)
            for (int i = 0; i < nin; ++i)
                fft(i, j, k) = ff(k, i, j);

    // transform in y
    gemm('n', 'n', nout, noutnin, nin, T{1.0}, umat.data_handle() + nout * nin, nout, fft.data_handle(), nin, T{0.0},
         ff2.data_handle(), nout);

    // transform in x
    gemm('n', 't', nout, nout2, nin, T{1.0}, umat.data_handle(), nout, ff2.data_handle(), nout2, T(add_flag),
         fout.data_handle(), nout);
}

template <typename T>
void transform(int nvec, int add_flag, const ndview<const T, 4> &fin, const ndview<const T, 2> &umat,
               const ndview<T, 4> &fout, sctl::Vector<T> &workspace) {
    const int nin = fin.extent(0);
    const int nout = fout.extent(0);
    const int block_in = nin * nin * nin, block_out = nout * nout * nout;
    for (int i = 0; i < nvec; ++i) {
        ndview<const T, 3> fin_view(fin.data_handle() + i * block_in, nin, nin, nin);
        ndview<T, 3> fout_view(fout.data_handle() + i * block_out, nout, nout, nout);

        transform(add_flag, fin_view, umat, fout_view, workspace);
    }
}

template <typename T>
void proxycharge2pw(const ndview<const T, 4> &proxy_coeffs, const ndview<const std::complex<T>, 2> &poly2pw,
                    const ndview<std::complex<T>, 4> &pw_expansion, sctl::Vector<T> &workspace) {
    using hpdmk::gemm::gemm;
    const int n_order = proxy_coeffs.extent(0);
    const int n_charge_dim = proxy_coeffs.extent(3);
    const int n_pw = pw_expansion.extent(0);
    const int n_pw2 = pw_expansion.extent(2);
    const int n_proxy_coeffs = sctl::pow<3>(n_order);
    const int n_pw_coeffs = n_pw * n_pw * n_pw2;

    workspace.ReInit(2 * (n_order * n_order * n_pw2 + n_order * n_pw2 * n_order + n_pw * n_pw2 * n_order +
                          n_order * n_order * n_order));
    std::complex<T> *workspace_ptr = (std::complex<T> *)(&workspace[0]);
    ndview<std::complex<T>, 3> ff(workspace_ptr, n_order, n_order, n_pw2);
    ndview<std::complex<T>, 3> fft(ff.data_handle() + ff.size(), n_order, n_pw2, n_order);
    ndview<std::complex<T>, 1> ff2(fft.data_handle() + fft.size(), n_pw * n_pw2 * n_order);
    ndview<std::complex<T>, 1> proxy_coeffs_complex(ff2.data_handle() + ff2.size(), n_order * n_order * n_order);

    for (int i_dim = 0; i_dim < n_charge_dim; ++i_dim) {
        for (int i = 0; i < n_proxy_coeffs; ++i)
            proxy_coeffs_complex[i] = proxy_coeffs.data_handle()[i + i_dim * n_proxy_coeffs];

        // transform in z
        gemm('n', 't', n_order * n_order, n_pw2, n_order, {1.0, 0.0}, &proxy_coeffs_complex[0], n_order * n_order,
             poly2pw.data_handle(), n_pw, {0.0, 0.0}, ff.data_handle(), n_order * n_order);

        for (int m1 = 0; m1 < n_order; ++m1)
            for (int k3 = 0; k3 < n_pw2; ++k3)
                for (int m2 = 0; m2 < n_order; ++m2)
                    fft(m2, k3, m1) = ff(m1, m2, k3);

        // transform in y
        gemm('n', 'n', n_pw, n_pw2 * n_order, n_order, {1.0, 0.0}, poly2pw.data_handle(), n_pw, fft.data_handle(),
             n_order, {0.0, 0.0}, ff2.data_handle(), n_pw);

        // transform in x
        gemm('n', 't', n_pw, n_pw * n_pw2, n_order, {1.0, 0.0}, poly2pw.data_handle(), n_pw, ff2.data_handle(),
             n_pw * n_pw2, {0.0, 0.0}, &pw_expansion(0, 0, 0, i_dim), n_pw);
    }
}

template <typename T>
void charge2proxycharge(const ndview<const T, 2> &r_src_, const ndview<const T, 2> &charge_,
                        const ndview<const T, 1> &center, T scale_factor, const ndview<T, 4> &coeffs,
                        sctl::Vector<T> &workspace) {
    using MatrixMap = Eigen::Map<Eigen::MatrixX<T>>;
    using CMatrixMap = Eigen::Map<const Eigen::MatrixX<T>>;

    const int n_dim = 3;
    const int order = coeffs.extent(0);
    const int n_charge_dim = coeffs.extent(3);
    const int n_src = r_src_.extent(1);

    workspace.ReInit(4 * n_src * order + n_src * order * order);
    MatrixMap dz(&workspace[0], n_src, order);
    MatrixMap dyz(&workspace[n_src * order], n_src, order * order);
    MatrixMap poly_x(&workspace[n_src * order + n_src * order * order], order, n_src);
    MatrixMap poly_y(&workspace[2 * n_src * order + n_src * order * order], order, n_src);
    MatrixMap poly_z(&workspace[3 * n_src * order + n_src * order * order], order, n_src);

    CMatrixMap r_src(r_src_.data_handle(), n_dim, n_src);
    CMatrixMap charge(charge_.data_handle(), n_charge_dim, n_src);

    auto calc_polynomial = hpdmk::chebyshev::get_polynomial_calculator<T>(order);
    for (int i_src = 0; i_src < n_src; ++i_src) {
        calc_polynomial(scale_factor * (r_src(0, i_src) - center(0)), &poly_x(0, i_src));
        calc_polynomial(scale_factor * (r_src(1, i_src) - center(1)), &poly_y(0, i_src));
        calc_polynomial(scale_factor * (r_src(2, i_src) - center(2)), &poly_z(0, i_src));
    }

    for (int i_dim = 0; i_dim < n_charge_dim; ++i_dim) {
        for (int k = 0; k < order; ++k)
            for (int m = 0; m < n_src; ++m)
                dz(m, k) = charge(i_dim, m) * poly_z(k, m);

        for (int k = 0; k < order; ++k)
            for (int j = 0; j < order; ++j)
                for (int m = 0; m < n_src; ++m)
                    dyz(m, j + k * order) = poly_y(j, m) * dz(m, k);

        MatrixMap(&coeffs(i_dim * order * order * order, 0, 0, 0), order, order * order) += poly_x * dyz;
    }
}

template <class Tree>
sctl::Vector<typename Tree::float_type> upward_pass(Tree &tree, int n_order) {
    using Real = typename Tree::float_type;
    constexpr int dim = Tree::Dim();
    static_assert(dim == 3, "Only 3D is supported");
    constexpr int n_vec = 1;
    const std::size_t n_coeffs = n_vec * sctl::pow<dim>(n_order);
    const std::size_t n_boxes = tree.GetNodeMID().Dim();
    constexpr int n_children = 1u << dim;
    const auto &node_lists = tree.GetNodeLists();
    const auto &node_attr = tree.GetNodeAttr();
    const auto &node_mid = tree.GetNodeMID();

    const auto start_level = tree.level_indices.Dim() - 1;
    const auto scale_factor = [&tree](int i_level) { return 2.0 / tree.boxsize[i_level]; };

    const auto r_src_ptr = [&tree](int i_box) { return &tree.r_src_sorted[tree.r_src_offset[i_box]]; };
    const auto r_src_view = [&tree, &node_attr, &r_src_ptr](int i_box) {
        assert(node_attr[i_box].Leaf);
        return ndview<Real, 2>(r_src_ptr(i_box), dim, tree.r_src_cnt[i_box]);
    };

    const auto charge_ptr = [&tree](int i_box) { return &tree.charge_sorted[tree.charge_offset[i_box]]; };
    const auto charge_view = [&tree, &node_attr, &charge_ptr, &n_vec](int i_box) {
        assert(node_attr[i_box].Leaf);
        return ndview<Real, 2>(charge_ptr(i_box), n_vec, tree.charge_cnt[i_box]);
    };
    const auto center_view = [&tree](int i_box) { return ndview<const Real, 1>(&tree.centers[i_box * dim], dim); };

    sctl::Vector<Real> proxy_coeffs(n_boxes * n_coeffs);
    auto proxy_view_upward = [&proxy_coeffs, &n_coeffs, &n_order](int i_box) {
        return ndview<Real, dim + 1>(&proxy_coeffs[i_box * n_coeffs], n_order, n_order, n_order, n_vec);
    };

    sctl::Vector<Real> c2p, p2c;
    std::tie(c2p, p2c) = hpdmk::chebyshev::get_c2p_p2c_matrices<Real>(dim, n_order);
    sctl::Vector<sctl::Vector<Real>> workspaces(n_coeffs * n_boxes);

#pragma omp parallel
#pragma omp single
    workspaces.ReInit(omp_get_num_threads());

#pragma omp parallel
    {
        auto &workspace = workspaces[omp_get_thread_num()];

#pragma omp for schedule(dynamic)
        for (auto i_box : tree.level_indices[start_level]) {
            if (!node_attr[i_box].Leaf)
                continue;

            charge2proxycharge<Real>(r_src_view(i_box), charge_view(i_box), center_view(i_box),
                                     scale_factor(start_level), proxy_view_upward(i_box), workspace);
        }
    }

#pragma omp parallel
    {
        auto &workspace = workspaces[omp_get_thread_num()];

        for (int i_level = start_level - 1; i_level >= 0; --i_level) {
#pragma omp for schedule(dynamic)
            for (auto parent_box : tree.level_indices[i_level]) {
                auto &children = node_lists[parent_box].child;
                for (int i_child = 0; i_child < n_children; ++i_child) {
                    const int child_box = children[i_child];
                    if (child_box < 0)
                        continue;

                    if (node_attr[child_box].Leaf) {
                        charge2proxycharge<Real>(r_src_view(child_box), charge_view(child_box), center_view(parent_box),
                                                 scale_factor(i_level), proxy_view_upward(parent_box), workspace);
                    } else {
                        ndview<const Real, 2> c2p_view(&c2p[i_child * dim * n_order * n_order], n_order, dim);
                        transform<Real>(n_vec, true, proxy_view_upward(child_box), c2p_view,
                                        proxy_view_upward(parent_box), workspace);
                    }
                }
            }
        }
    }

    return proxy_coeffs;
}
} // namespace hpdmk

#endif
