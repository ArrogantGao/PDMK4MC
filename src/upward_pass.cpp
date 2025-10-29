#include <chebychev.hpp>
#include <gemm.hpp>
#include <omp.h>
#include <tree.hpp>
#include <upward_pass.hpp>

namespace hpdmk {

int get_poly_order(int ndigits) {
    if (ndigits <= 3)
        return 18;
    // FIXME: Add more orders as needed
    throw std::runtime_error("Polynomial order for requested precision not implemented");
}

// double get_PSWF_difference_kernel_hpw(double boxsize) { return M_PI * 2.0 / 3.0 / boxsize; }

template <typename T>
void tensorprod_transform(int add_flag, const ndview<const T, 3> &fin, const ndview<const T, 2> &umat_,
                          const ndview<T, 3> &fout, sctl::Vector<T> &workspace) {
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
void tensorprod_transform(int nvec, int add_flag, const ndview<const T, 4> &fin, const ndview<const T, 2> &umat,
                          const ndview<T, 4> &fout, sctl::Vector<T> &workspace) {
    const int nin = fin.extent(0);
    const int nout = fout.extent(0);
    const int block_in = nin * nin * nin, block_out = nout * nout * nout;
    for (int i = 0; i < nvec; ++i) {
        ndview<const T, 3> fin_view(fin.data_handle() + i * block_in, nin, nin, nin);
        ndview<T, 3> fout_view(fout.data_handle() + i * block_out, nout, nout, nout);

        tensorprod_transform(add_flag, fin_view, umat, fout_view, workspace);
    }
}

template <typename T>
void proxycharge2pw(const ndview<const T, 4> &proxy_coeffs, const ndview<const std::complex<T>, 2> &poly2pw,
                    const ndview<std::complex<T>, 4> &pw_expansion, sctl::Vector<T> &workspace) {
    using hpdmk::gemm::gemm;
    const int n_order = proxy_coeffs.extent(0);
    const int n_charge_dim = proxy_coeffs.extent(3);
    const int n_pw = pw_expansion.extent(0);
    const int n_pw2 = (n_pw + 1) / 2;
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

template <typename T>
sctl::Vector<std::complex<T>> calc_prox_to_pw(T boxsize, T hpw, int n_pw, int n_order) {
    sctl::Vector<std::complex<T>> prox2pw_vec(n_pw * n_order);
    sctl::Vector<std::complex<T>> pw2prox_vec(n_pw * n_order);

    using matrix_t = Eigen::MatrixX<std::complex<T>>;
    const T dsq = 0.5 * boxsize;
    const auto xs = hpdmk::chebyshev::get_cheb_nodes(n_order, -1.0, 1.0);

    Eigen::Map<matrix_t> prox2pw(&prox2pw_vec[0], n_pw, n_order);
    Eigen::Map<matrix_t> pw2poly(&pw2prox_vec[0], n_pw, n_order);

    matrix_t tmp(n_pw, n_order);
    const int shift = n_pw / 2;
    for (int i = 0; i < n_order; ++i) {
        const T factor = xs[i] * dsq * hpw;
        for (int j = 0; j < n_pw; ++j)
            tmp(j, i) = exp(std::complex<T>{0, T(j - shift) * factor});
    }

    const auto &[vmat, umat_lu] = chebyshev::get_vandermonde_and_LU<T>(n_order);
    // Can't use umat_lu.solve() because eigen doesn't support LU with mixed complex/real types
    const Eigen::MatrixX<T> umat = umat_lu.inverse();
    pw2poly = tmp * umat.transpose();

    for (int i = 0; i < n_order * n_pw; ++i)
        prox2pw(i) = std::conj(pw2poly(i));

    return prox2pw_vec;
}

template <typename Real>
void decenter_phase(const ndview<const Real, 1> &center, Real boxsize,
                    const ndview<std::complex<Real>, 4> &pw_expansion, sctl::Vector<Real> &workspace) {
    const auto n_pw = pw_expansion.extent(0);
    const auto dim = 3;
    workspace.ReInit(dim * n_pw * 2);
    ndview<std::complex<Real>, 2> shift_correction(reinterpret_cast<std::complex<Real> *>(&workspace[0]), dim, n_pw);
    const int shift = n_pw / 2;
    const Real factor(-2.0 * M_PI / 3.0 / boxsize);
    for (int i_dim = 0; i_dim < dim; ++i_dim)
        for (int i = 0; i < n_pw; ++i)
            shift_correction(i_dim, i) = std::exp(std::complex<Real>(0, factor * center[i_dim] * (i - shift)));

    for (int i = 0; i < n_pw; ++i) {
        for (int j = 0; j < n_pw; ++j) {
            auto shift_common = shift_correction(0, i) * shift_correction(1, j);
            for (int k = 0; k < n_pw; ++k) {
                pw_expansion(i, j, k, 0) = pw_expansion(i, j, k, 0) * shift_common * shift_correction(2, k);
            }
        }
    }
}

template <class Tree>
void upward_pass(Tree &tree, sctl::Vector<sctl::Vector<std::complex<typename Tree::float_type>>> &outgoing_pw) {
    // Some various convenience variables
    using Real = typename Tree::float_type;
    constexpr int dim = Tree::Dim();
    static_assert(dim == 3, "Only 3D is supported");
    constexpr int n_vec = 1;
    const int n_order = get_poly_order(tree.n_digits);
    const std::size_t n_proxy_coeffs = n_vec * sctl::pow<dim>(n_order);
    const int n_pw_diff = 2 * tree.n_diff + 1;
    const std::size_t n_boxes = tree.GetNodeMID().Dim();
    constexpr int n_children = 1u << dim;
    const auto &node_lists = tree.GetNodeLists();
    const auto &node_attr = tree.GetNodeAttr();
    const auto &node_mid = tree.GetNodeMID();

    // Our result container
    outgoing_pw.ReInit(n_boxes);

    // Very large temporary!
    std::vector<std::vector<Real>> proxy_coeffs(n_boxes);

    // Tiny temporary workspaces for each thread
    sctl::Vector<sctl::Vector<Real>> workspaces;

    // Transformation matrices. Only need c2p (child to parent)
    sctl::Vector<Real> c2p, p2c;
    std::tie(c2p, p2c) = hpdmk::chebyshev::get_c2p_p2c_matrices<Real>(dim, n_order);
    sctl::Vector<sctl::Vector<std::complex<Real>>> poly2pws;

    // Polynomial -> planewave coefficient translation matrices
    poly2pws.ReInit(tree.level_indices.Dim());
    for (int i_level = 2; i_level < tree.level_indices.Dim(); ++i_level) {
        // const Real hpw = get_PSWF_difference_kernel_hpw(tree.boxsize[i_level]);
        const Real hpw = tree.delta_k[i_level];
        poly2pws[i_level] = calc_prox_to_pw(tree.boxsize[i_level], hpw, n_pw_diff, n_order);
    }

    // Various convenience functions/views
    const auto scale_factor = [&tree](int i_level) { return 2.0 / tree.boxsize[i_level]; };
    const auto r_src_view = [&tree](int i_box) {
        return ndview<Real, 2>(tree.r_src_ptr(i_box), dim, tree.r_src_cnt_all[i_box]);
    };
    const auto charge_view = [&tree, &n_vec](int i_box) {
        return ndview<Real, 2>(tree.charge_ptr(i_box), n_vec, tree.r_src_cnt_all[i_box]);
    };
    const auto center_view = [&tree](int i_box) { return ndview<const Real, 1>(&tree.centers[i_box * dim], dim); };
    auto proxy_view = [&proxy_coeffs, &n_proxy_coeffs, &n_order](int i_box) {
        return ndview<Real, dim + 1>(&proxy_coeffs[i_box][0], n_order, n_order, n_order, n_vec);
    };
    auto outgoing_pw_view = [&outgoing_pw, &n_vec, &n_pw_diff](int i_box) {
        return ndview<std::complex<Real>, 4>(&outgoing_pw[i_box][0], n_pw_diff, n_pw_diff, n_pw_diff, n_vec);
    };
    auto poly2pw_view = [&poly2pws, &n_pw_diff, &n_order](int i_level) {
        return ndview<std::complex<Real>, 2>(&poly2pws[i_level][0], n_pw_diff, n_order);
    };
    auto c2p_view = [&c2p, dim, n_order](int i_child) {
        return ndview<const Real, 2>(&c2p[i_child * dim * n_order * n_order], n_order, dim);
    };

#pragma omp parallel
#pragma omp single
    workspaces.ReInit(omp_get_num_threads());

    const auto lowest_nonleaf_level = tree.level_indices.Dim() - 2;
#pragma omp parallel
    {
        auto &workspace = workspaces[omp_get_thread_num()];

#pragma omp for schedule(dynamic)
        for (auto i_box : tree.level_indices[lowest_nonleaf_level]) {
            if (!tree.r_src_cnt_all[i_box])
                continue;
            proxy_coeffs[i_box].resize(n_proxy_coeffs);
            charge2proxycharge<Real>(r_src_view(i_box), charge_view(i_box), center_view(i_box),
                                     scale_factor(lowest_nonleaf_level), proxy_view(i_box), workspace);
            outgoing_pw[i_box].ReInit(n_pw_diff * n_pw_diff * n_pw_diff * n_vec);
            proxycharge2pw<Real>(proxy_view(i_box), poly2pw_view(lowest_nonleaf_level), outgoing_pw_view(i_box),
                                 workspace);
            decenter_phase<Real>(center_view(i_box), tree.boxsize[lowest_nonleaf_level], outgoing_pw_view(i_box),
                                 workspace);
        }
    }

#pragma omp parallel
    {
        auto &workspace = workspaces[omp_get_thread_num()];

        for (int i_level = lowest_nonleaf_level - 1; i_level >= 2; --i_level) {
#pragma omp for schedule(dynamic)
            for (auto parent_box : tree.level_indices[i_level]) {
                auto &children = node_lists[parent_box].child;
                for (int i_child = 0; i_child < n_children; ++i_child) {
                    const int child_box = children[i_child];
                    if (child_box < 0 || !tree.r_src_cnt_all[child_box])
                        continue;

                    proxy_coeffs[parent_box].resize(n_proxy_coeffs);
                    tensorprod_transform<Real>(n_vec, true, proxy_view(child_box), c2p_view(i_child),
                                               proxy_view(parent_box), workspace);
                }
                if (proxy_coeffs[parent_box].size()) {
                    outgoing_pw[parent_box].ReInit(n_pw_diff * n_pw_diff * n_pw_diff * n_vec);
                    proxycharge2pw<Real>(proxy_view(parent_box), poly2pw_view(i_level), outgoing_pw_view(parent_box),
                                         workspace);
                    decenter_phase<Real>(center_view(parent_box), tree.boxsize[i_level], outgoing_pw_view(parent_box),
                                         workspace);
                }
                for (int i_child = 0; i_child < n_children; ++i_child) {
                    const int child_box = children[i_child];
                    if (child_box < 0)
                        continue;
                    proxy_coeffs[child_box].clear();
                }
            }
        }
    }
}

} // namespace hpdmk

template void hpdmk::upward_pass(hpdmk::HPDMKPtTree<float> &tree, sctl::Vector<sctl::Vector<std::complex<float>>> &outgoing_pw);
template void hpdmk::upward_pass(hpdmk::HPDMKPtTree<double> &tree,
                                 sctl::Vector<sctl::Vector<std::complex<double>>> &outgoing_pw);
