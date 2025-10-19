#ifndef NUDFT_HPP
#define NUDFT_HPP

#include <hpdmk.h>
#include <vector>
#include <complex>

#include <finufft.h>
#include <sctl.hpp>

using namespace std;

namespace hpdmk {
    template<typename T>
    void nudft3d1(const int M, T* x, T* y, T* z, complex<T>* c, const int iflag, const int N1, const int N2, const int N3, complex<T>* f){
        vector<complex<T>> x_cache(N1);
        vector<complex<T>> y_cache(N2);
        vector<complex<T>> z_cache(N3);

        T iflag_sign = iflag > 0 ? 1 : -1;

        for (int i = 0; i < M; i++){
            auto xi = x[i];
            auto yi = y[i];
            auto zi = z[i];
            auto ci = c[i];

            auto exp_x0 = exp(complex<T>(0, iflag_sign * xi));
            auto exp_y0 = exp(complex<T>(0, iflag_sign * yi));
            auto exp_z0 = exp(complex<T>(0, iflag_sign * zi));
            
            double kx_min = - N1 / 2;
            double ky_min = - N2 / 2;
            double kz_min = - N3 / 2;

            x_cache[0] = exp(complex<T>(0, iflag_sign * xi * (kx_min))) * ci;
            y_cache[0] = exp(complex<T>(0, iflag_sign * yi * (ky_min)));
            z_cache[0] = exp(complex<T>(0, iflag_sign * zi * (kz_min)));

            for (int l = 1; l < N1; l++){
                x_cache[l] = x_cache[l - 1] * exp_x0;
            }

            for (int m = 1; m < N2; m++){
                y_cache[m] = y_cache[m - 1] * exp_y0;
            }

            for (int n = 1; n < N3; n++){
                z_cache[n] = z_cache[n - 1] * exp_z0;
            }

            complex<T> temp_z, temp_zy;
            for (int n = 0; n < N3; n++){
                temp_z = z_cache[n];
                for (int m = 0; m < N2; m++){
                    temp_zy = temp_z * y_cache[m];
                    for (int l = 0; l < N1; l++){
                        f[n * N2 * N1 + m * N1 + l] += temp_zy * x_cache[l];
                    }
                }
            }
        }
    }

    // only consider the half plane of kz >= 0
    template <typename T>
    inline void nudft3d1_halfplane(const int M,
                                const T* __restrict x,
                                const T* __restrict y,
                                const T* __restrict z,
                                const std::complex<T>* __restrict c,
                                const int iflag,
                                const int N1, const int N2, const int N3,
                                std::complex<T>* __restrict f) {

        using complex_t = std::complex<T>;
        const T iflag_sign = (iflag > 0) ? 1 : -1;

        const int N3_half = (N3 + 1) / 2 + (N3 % 2 == 0);

        std::vector<complex_t> x_cache(N1);
        std::vector<complex_t> y_cache(N2);
        std::vector<complex_t> z_cache(N3_half);

        const T kx_min = - N1 / 2;
        const T ky_min = - N2 / 2;
        const T kz_min = - N3 / 2;

        // #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; ++i) {
            const T xi = x[i];
            const T yi = y[i];
            const T zi = z[i];
            const complex_t ci = c[i];

            const complex_t exp_x_step = std::exp(complex_t(0, iflag_sign * xi));
            const complex_t exp_y_step = std::exp(complex_t(0, iflag_sign * yi));
            const complex_t exp_z_step = std::exp(complex_t(0, iflag_sign * zi));

            x_cache[0] = std::exp(complex_t(0, iflag_sign * xi * kx_min)) * ci;
            for (int l = 1; l < N1; ++l)
                x_cache[l] = x_cache[l - 1] * exp_x_step;

            y_cache[0] = std::exp(complex_t(0, iflag_sign * yi * ky_min));
            for (int m = 1; m < N2; ++m)
                y_cache[m] = y_cache[m - 1] * exp_y_step;

            z_cache[0] = std::exp(complex_t(0, iflag_sign * zi * kz_min));
            for (int n = 1; n < N3_half; ++n)
                z_cache[n] = z_cache[n - 1] * exp_z_step;

            for (int n = 0; n < N3_half; ++n) {
                const complex_t z_val = z_cache[n];
                for (int m = 0; m < N2; ++m) {
                    const complex_t zy_val = z_val * y_cache[m];
                    complex_t* __restrict fptr = f + (n * N2 + m) * N1;
                    #pragma omp simd
                    for (int l = 0; l < N1; ++l)
                        fptr[l] += zy_val * x_cache[l];
                }
            }
        }
    }

    template<typename T>
    inline void nudft3d1_single_halfplane(const T x, const T y, const T z,
                                        const std::complex<T> c, const int iflag,
                                        const int N1, const int N2, const int N3,
                                        std::complex<T>* __restrict x_cache,
                                        std::complex<T>* __restrict y_cache,
                                        std::complex<T>* __restrict z_cache,
                                        std::complex<T>* __restrict f) {
        using complex_t = std::complex<T>;
        const T iflag_sign = (iflag > 0) ? 1 : -1;
        const int N3_half = (N3 % 2 == 0) ? (N3 / 2 + 1) : ((N3 + 1) / 2);

        const T kx_min = - N1 / 2;
        const T ky_min = - N2 / 2;
        const T kz_min = - N3 / 2;

        const complex_t exp_x_step = std::exp(complex_t(0, iflag_sign * x));
        const complex_t exp_y_step = std::exp(complex_t(0, iflag_sign * y));
        const complex_t exp_z_step = std::exp(complex_t(0, iflag_sign * z));

        x_cache[0] = std::exp(complex_t(0, iflag_sign * x * kx_min)) * c;
        for (int l = 1; l < N1; ++l)
            x_cache[l] = x_cache[l - 1] * exp_x_step;

        y_cache[0] = std::exp(complex_t(0, iflag_sign * y * ky_min));
        for (int m = 1; m < N2; ++m)
            y_cache[m] = y_cache[m - 1] * exp_y_step;

        z_cache[0] = std::exp(complex_t(0, iflag_sign * z * kz_min));
        for (int n = 1; n < N3_half; ++n)
            z_cache[n] = z_cache[n - 1] * exp_z_step;

        for (int n = 0; n < N3_half; ++n) {
            const complex_t z_val = z_cache[n];
            for (int m = 0; m < N2; ++m) {
                const complex_t zy_val = z_val * y_cache[m];
                complex_t* __restrict fptr = f + (n * N2 + m) * N1;
                #pragma omp simd
                for (int l = 0; l < N1; ++l)
                    fptr[l] = zy_val * x_cache[l];
            }
        }
    }

    template<typename T>
    void nufft3d1(const int M, T* x, T* y, T* z, complex<T>* c, const int iflag, T eps, const int N1, const int N2, const int N3, complex<T>* f){
        if constexpr (is_same_v<T, double>) {
            finufft3d1(M, x, y, z, c, iflag, eps, N1, N2, N3, f, NULL);
        } else if constexpr (is_same_v<T, float>) {
            finufftf3d1(M, x, y, z, c, iflag, eps, N1, N2, N3, f, NULL);
        } else {
            throw std::invalid_argument("Type T must be double or float");
        }
    }
}

#endif