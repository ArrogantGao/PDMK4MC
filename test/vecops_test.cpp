#include <hpdmk.h>
#include <complex>
#include <vector>
#include <vecops.hpp>
#include <random>

#include <gtest/gtest.h>

using namespace std;

#define M 256

void test_addsub() {
    vector<complex<double>> a(M);
    vector<complex<double>> b(M);

    vector<complex<double>> a_copy(M);
    
    auto bool_add = {true, false};
    auto bool_conjugate = {true, false};

    for (const bool add : bool_add) {
        for (const bool conjugate : bool_conjugate) {

            for (int i = 0; i < M; i++) {
                a[i] = complex<double>(float(i), float(i));
                b[i] = complex<double>(float(i), float(i));
                a_copy[i] = a[i];
            }

            if (add) {
                if (conjugate) {
                    hpdmk::vec_addsub<double, true, true>(M, a.data(), b.data());
                } else {
                    hpdmk::vec_addsub<double, true, false>(M, a.data(), b.data());
                }
            } else {
                if (conjugate) {
                    hpdmk::vec_addsub<double, false, true>(M, a.data(), b.data());
                } else {
                    hpdmk::vec_addsub<double, false, false>(M, a.data(), b.data());
                }
            }

            for (int i = 0; i < M; i++) {
                if (add) {
                    if (conjugate) {
                        EXPECT_EQ(a[i], a_copy[i] + conj(b[i]));
                    } else {
                        EXPECT_EQ(a[i], a_copy[i] + b[i]);
                    }
                } else {
                    if (conjugate) {
                        EXPECT_EQ(a[i], a_copy[i] - conj(b[i]));
                    } else {
                        EXPECT_EQ(a[i], a_copy[i] - b[i]);
                    }
                }
            }
        }
    }
}

void test_muladdsub(const bool add, const bool conjugate) {
    vector<complex<double>> a(M);
    vector<complex<double>> b(M);
    vector<complex<double>> c(M);

    vector<complex<double>> a_copy(M);

    for (int i = 0; i < M; i++) {
        a[i] = complex<double>(i, i);
        b[i] = complex<double>(i, i);
        c[i] = complex<double>(i, i);
        a_copy[i] = a[i];
    }

    if (add) {
        if (conjugate) {
            hpdmk::vec_muladdsub<double, true, true>(M, a.data(), b.data(), c.data());
        } else {
            hpdmk::vec_muladdsub<double, true, false>(M, a.data(), b.data(), c.data());
        }
    } else {
        if (conjugate) {
            hpdmk::vec_muladdsub<double, false, true>(M, a.data(), b.data(), c.data());
        } else {
            hpdmk::vec_muladdsub<double, false, false>(M, a.data(), b.data(), c.data());
        }
    }

    for (int i = 0; i < M; i++) {
        if (add) {
            if (conjugate) {
                EXPECT_EQ(a[i], a_copy[i] + conj(b[i] * c[i]));
            } else {
                EXPECT_EQ(a[i], a_copy[i] + (b[i] * c[i]));
            }
        } else {
            if (conjugate) {
                EXPECT_EQ(a[i], a_copy[i] - conj(b[i] * c[i]));
            } else {
                EXPECT_EQ(a[i], a_copy[i] - (b[i] * c[i]));
            }
        }
    }
}

void test_tridot(const bool conjugate_a, const bool conjugate_b) {
    vector<complex<double>> a(M);
    vector<complex<double>> b(M);
    vector<double> c(M);

    for (int i = 0; i < M; i++) {
        a[i] = complex<double>(i, i);
        b[i] = complex<double>(i, i);
        c[i] = i;
    }

    double res = 0;
    if (conjugate_a) {
        if (conjugate_b) {
            res = hpdmk::vec_tridot<double, true, true>(M, a.data(), b.data(), c.data());
        } else {
            res = hpdmk::vec_tridot<double, true, false>(M, a.data(), b.data(), c.data());
        }
    } else {
        if (conjugate_b) {
            res = hpdmk::vec_tridot<double, false, true>(M, a.data(), b.data(), c.data());
        } else {
            res = hpdmk::vec_tridot<double, false, false>(M, a.data(), b.data(), c.data());
        }
    }

    double res_direct = 0;
    for (int i = 0; i < M; i++) {
        if (conjugate_a) {
            if (conjugate_b) {
                res_direct += std::real(conj(a[i]) * conj(b[i])) * c[i];
            } else {
                res_direct += std::real(conj(a[i]) * b[i]) * c[i];
            }
        } else {
            if (conjugate_b) {
                res_direct += std::real(a[i] * conj(b[i])) * c[i];
            } else {
                res_direct += std::real(a[i] * b[i]) * c[i];
            }
        }
    }

    EXPECT_EQ(res, res_direct);
}

void test_shift_ops(){
    vector<complex<double>> target(M);
    vector<complex<double>> origin(M);
    vector<complex<double>> outgoing(M);
    vector<complex<double>> incoming(M);
    vector<complex<double>> shift_vec(M);
    vector<double> D(M);

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, 1);

    for (int i = 0; i < M; i++) {
        target[i] = complex<double>(distribution(generator), distribution(generator));
        origin[i] = complex<double>(distribution(generator), distribution(generator));
        incoming[i] = complex<double>(distribution(generator), distribution(generator));
        outgoing[i] = complex<double>(distribution(generator), distribution(generator));
        shift_vec[i] = complex<double>(distribution(generator), distribution(generator));
        D[i] = distribution(generator);
    }

    // res = real( (t-o) * conj(g-o) ) * W
    double res = hpdmk::vec_shift_window<double>(M, target.data(), origin.data(), outgoing.data(), D.data());
    double res_ref = 0;
    for (int i = 0; i < M; i++) {
        res_ref += std::real((target[i] - origin[i]) * conj(outgoing[i] - origin[i])) * D[i];
    }
    EXPECT_EQ(res, res_ref);

    // res = re(o * (i - conj(o)))  * D
    res = hpdmk::vec_shift_diff_origin<double>(M, origin.data(), incoming.data(), D.data());
    res_ref = 0;
    for (int i = 0; i < M; i++) {
        res_ref += std::real(origin[i] * (incoming[i] - conj(origin[i]))) * D[i];
    }
    EXPECT_EQ(res, res_ref);

    // res = real( t * (i - conj(o)) ) * D
    res = hpdmk::vec_shift_diff_target_same<double>(M, target.data(), origin.data(), incoming.data(), D.data());
    res_ref = 0;
    for (int i = 0; i < M; i++) {
        res_ref += std::real(target[i] * (incoming[i] - conj(origin[i]))) * D[i];
    }
    EXPECT_EQ(res, res_ref);

    // res = real( t * (i - conj(o * s)) ) * D
    res = hpdmk::vec_shift_diff_target_neib<double>(M, target.data(), origin.data(), incoming.data(), shift_vec.data(), D.data());
    res_ref = 0;
    for (int i = 0; i < M; i++) {
        res_ref += std::real(target[i] * (incoming[i] - conj(origin[i] * shift_vec[i]))) * D[i];
    }
    EXPECT_EQ(res, res_ref);
}


TEST(VecOpsTest, AddSub) {
    test_addsub();
}

TEST(VecOpsTest, MulAddSub) {
    test_muladdsub(true, true);
    test_muladdsub(true, false);
    test_muladdsub(false, true);
    test_muladdsub(false, false);
}

TEST(VecOpsTest, Tridot) {
    test_tridot(true, true);
    test_tridot(true, false);
    test_tridot(false, true);
    test_tridot(false, false);
}

TEST(VecOpsTest, ShiftOps) {
    test_shift_ops();
}