#ifndef VEC_OPS_HPP
#define VEC_OPS_HPP

#include <hpdmk.h>
#include <cmath>
#include <complex>

namespace hpdmk {

    template <typename Real>
    inline Real vec_doudot(int M,
        const std::complex<Real>* __restrict A,
        const Real* __restrict B) {

        Real res = 0;
        #pragma omp simd reduction(+:res)
        for (int i = 0; i < M; ++i) {
            Real a_re = A[i].real();
            Real a_im = A[i].imag();
            Real re_part = a_re * a_re + a_im * a_im;   
            res += re_part * B[i];
        }
        return res;
    }

    template <typename Real, bool ConjugateA, bool ConjugateB>
    inline Real vec_tridot(int M,
                    const std::complex<Real>* __restrict A,
                    const std::complex<Real>* __restrict B,
                    const Real* __restrict C) {
        Real res = 0;
        #pragma omp simd reduction(+:res)
        for (int i = 0; i < M; ++i) {
            Real a_re = A[i].real(), a_im = A[i].imag();
            Real b_re = B[i].real(), b_im = B[i].imag();
            Real re_part;

            if constexpr (ConjugateA xor ConjugateB) {
                re_part = a_re * b_re + a_im * b_im;
            } else {
                re_part = a_re * b_re - a_im * b_im;
            }

            res += re_part * C[i];
        }
        return res;
    }

    template <typename Real, bool Add, bool ConjugateB>
    inline void vec_addsub(int M,
                        std::complex<Real>* __restrict A,
                        const std::complex<Real>* __restrict B) {
        
        #pragma ivdep
        #pragma omp simd
        for (int i = 0; i < M; ++i) {
            Real a_re = A[i].real();
            Real a_im = A[i].imag();
            Real b_re = B[i].real();
            Real b_im = B[i].imag();

            if constexpr (Add) {
                A[i].real(a_re + b_re);
                if constexpr (ConjugateB)
                    A[i].imag(a_im - b_im);
                else
                    A[i].imag(a_im + b_im);
            } else {
                A[i].real(a_re - b_re);
                if constexpr (ConjugateB)
                    A[i].imag(a_im + b_im);
                else
                    A[i].imag(a_im - b_im);
            }
        }
    }

    template <typename Real, bool Add, bool ConjugateB>
    inline void vec_muladdsub(int M,
                            std::complex<Real>* __restrict A,
                            const std::complex<Real>* __restrict B,
                            const std::complex<Real>* __restrict C) {
        
        #pragma ivdep
        #pragma omp simd
        for (int i = 0; i < M; ++i) {
            const Real a_re = A[i].real();
            const Real a_im = A[i].imag();
            const Real b_re = B[i].real();
            const Real b_im = B[i].imag();
            const Real c_re = C[i].real();
            const Real c_im = C[i].imag();

            Real prod_re, prod_im;

            if constexpr (ConjugateB) {
                // conj(B * C)
                prod_re =  b_re * c_re - b_im * c_im;
                prod_im = -b_re * c_im - b_im * c_re;
            } else {
                // (B * C)
                prod_re =  b_re * c_re - b_im * c_im;
                prod_im =  b_re * c_im + b_im * c_re;
            }

            if constexpr (Add) {
                A[i].real(a_re + prod_re);
                A[i].imag(a_im + prod_im);
            } else {
                A[i].real(a_re - prod_re);
                A[i].imag(a_im - prod_im);
            }
        }
    }

    // operations below are highly specialized
    template <typename Real>
    inline Real vec_shift_window(int M,
        const std::complex<Real>* __restrict target_root,
        const std::complex<Real>* __restrict origin_root,
        const std::complex<Real>* __restrict outgoing_root,
        const Real* __restrict W) {

        // res = real( (t-o) * conj(g-o) ) * W

        Real res = 0;
        #pragma omp simd reduction(+:res)
        for (int i = 0; i < M; ++i) {
            const Real t_re = target_root[i].real();
            const Real t_im = target_root[i].imag();
            const Real o_re = origin_root[i].real();
            const Real o_im = origin_root[i].imag();
            const Real g_re = outgoing_root[i].real();
            const Real g_im = outgoing_root[i].imag();

            const Real a_re = t_re - o_re;
            const Real a_im = t_im - o_im;
            const Real b_re = g_re - o_re;
            const Real b_im = g_im - o_im;

            const Real re_part = a_re * b_re + a_im * b_im;
            res += re_part * W[i];
        }
        return res;
    }

    template <typename Real>
    inline Real vec_shift_diff_origin(int M,
        const std::complex<Real>* __restrict origin,
        const std::complex<Real>* __restrict incoming,
        const Real* __restrict D) {

        Real res = 0;
        #pragma omp simd reduction(+:res)
        for (int i = 0; i < M; ++i) {
            const Real o_re = origin[i].real();
            const Real o_im = origin[i].imag();
            const Real i_re = incoming[i].real();
            const Real i_im = incoming[i].imag();

            // re(o * (i - conj(o))) * D = o_re*i_re - o_im*i_im - (o_re*o_re + o_im*o_im)
            const Real re_part = o_re * i_re - o_im * i_im - (o_re*o_re + o_im*o_im);
            res += re_part * D[i];
        }
        return res;
    }

    template <typename Real>
    inline Real vec_shift_diff_target_same(int M,
        const std::complex<Real>* __restrict target,
        const std::complex<Real>* __restrict origin,
        const std::complex<Real>* __restrict incoming,
        const Real* __restrict D) {

        // res = real( t * (i - conj(o)) ) * D

        Real res = 0;
        #pragma omp simd reduction(+:res)
        for (int i = 0; i < M; ++i) {
            const Real t_re = target[i].real();
            const Real t_im = target[i].imag();
            const Real o_re = origin[i].real();
            const Real o_im = origin[i].imag();
            const Real i_re = incoming[i].real();
            const Real i_im = incoming[i].imag();

            // real[t*(i - conj(o))] = t_re*(i_re - o_re) - t_im*(i_im + o_im)
            const Real re_part = t_re * (i_re - o_re) - t_im * (i_im + o_im);
            res += re_part * D[i];
        }
        return res;
    }

    template <typename Real>
    inline Real vec_shift_diff_target_neib(int M,
        const std::complex<Real>* __restrict target,
        const std::complex<Real>* __restrict origin,
        const std::complex<Real>* __restrict incoming,
        const std::complex<Real>* __restrict shift_vec,
        const Real* __restrict D) {

        // res = real( t * (i - conj(o * s)) ) * D

        Real res = 0;
        #pragma omp simd reduction(+:res)
        for (int i = 0; i < M; ++i) {
            const Real t_re = target[i].real();
            const Real t_im = target[i].imag();
            const Real o_re = origin[i].real();
            const Real o_im = origin[i].imag();
            const Real i_re = incoming[i].real();
            const Real i_im = incoming[i].imag();
            const Real s_re = shift_vec[i].real();
            const Real s_im = shift_vec[i].imag();

            // o*s
            const Real os_re = o_re * s_re - o_im * s_im;
            const Real os_im = o_re * s_im + o_im * s_re;

            // real[t*(i - conj(o*s))] = t_re*(i_re - os_re) - t_im*(i_im + os_im)
            const Real re_part = t_re * (i_re - os_re) - t_im * (i_im + os_im);
            res += re_part * D[i];
        }
        return res;
    }
}

#endif