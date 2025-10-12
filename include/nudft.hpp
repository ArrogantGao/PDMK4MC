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
    template<typename T>
    void nudft3d1_halfplane(const int M, T* x, T* y, T* z, complex<T>* c, const int iflag, const int N1, const int N2, const int N3, complex<T>* f){
        vector<complex<T>> x_cache(N1);
        vector<complex<T>> y_cache(N2);
        vector<complex<T>> z_cache(N3);

        int N3_half;
        if (N3 % 2 == 0){
            N3_half = N3 / 2 + 1;
        } else {
            N3_half = (N3 + 1) / 2;
        }

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

            for (int n = 1; n < N3_half; n++){
                z_cache[n] = z_cache[n - 1] * exp_z0;
            }

            complex<T> temp_z, temp_zy;
            for (int n = 0; n < N3_half; n++){
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