#include <cmath>
#include <complex>
#include <iostream>
#include <vector>
#include <array>

#include <hpdmk.h>
#include <ewald.hpp>
#include <vesin.h>
#include <sctl.hpp>
#include <utils.hpp>

#include <omp.h>
#include <chrono>

namespace hpdmk {
    Ewald::Ewald(const double L, const double s, const double alpha, const double eps, const std::vector<double> &q, const std::vector<double> &r, const int n_particles)
        : L(L), s(s), alpha(alpha), eps(eps), V(L * L * L), r_c(s / alpha), k_c(2 * s * alpha), n_particles(n_particles), q(q), r(r) {
        
        int n = std::ceil(k_c / (2 * M_PI / L));
        std::vector<double> k(2 * n + 1);

        double dk = 2 * M_PI / L;

        for (int i = 0; i < 2 * n + 1; i++) {
            k[i] = ( - n + i ) * dk;
        }

        this->k = k;

        const double cutoff = this->r_c;
        auto options = VesinOptions();
        options.cutoff = cutoff;
        options.full = true;
        options.return_shifts = true;
        options.return_distances = true;
        options.return_vectors = false;

        double box[3][3] = {
            {L, 0.0, 0.0},
            {0.0, L, 0.0},
            {0.0, 0.0, L},
        };

        // auto start = std::chrono::high_resolution_clock::now();

        // VesinNeighborList neighbors;
        // const char* error_message = nullptr;
        // auto status = vesin_neighbors(
        //     r,
        //     n_particles,
        //     box,
        //     true,
        //     VesinDevice::VesinCPU,
        //     options,
        //     &neighbors,
        //     &error_message
        // );

        // auto end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> duration = end - start;
        // std::cout << "time for vesin: " << duration.count() << " seconds" << std::endl;

        this->neighbors = neighbors;

        auto start = std::chrono::high_resolution_clock::now();

        // initialize planewave coefficients
        this->init_interaction_matrix();
        this->init_planewave_coeffs();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "time for init planewave coeffs: " << duration.count() << " seconds" << std::endl;
    }

    void Ewald::init_planewave_coeffs() {

        int d = k.size();
        int n = std::ceil(k_c / (2 * M_PI / L));
        planewave_coeffs = std::vector<std::complex<double>>(std::pow(d, 3));
        std::fill(planewave_coeffs.begin(), planewave_coeffs.end(), 0);

        std::vector<std::complex<double>> exp_ikx(d);
        std::vector<std::complex<double>> exp_iky(d);
        std::vector<std::complex<double>> exp_ikz(d);        

        std::complex<double> exp_ikx0, exp_iky0, exp_ikz0;

        for (int i = 0; i < n_particles; i++) {
            double x = r[i * 3 + 0];
            double y = r[i * 3 + 1];
            double z = r[i * 3 + 2];

            auto k0 = 2 * M_PI / L;
            exp_ikx0 = std::exp( - std::complex<double>(0.0, 1.0) * k0 * x);
            exp_iky0 = std::exp( - std::complex<double>(0.0, 1.0) * k0 * y);
            exp_ikz0 = std::exp( - std::complex<double>(0.0, 1.0) * k0 * z);

            for (int a = -n; a <= n; a++) {
                exp_ikx[a + n] = std::pow(exp_ikx0, a);
                exp_iky[a + n] = std::pow(exp_iky0, a);
                exp_ikz[a + n] = std::pow(exp_ikz0, a);
            }

            #pragma omp parallel for
            for (int ix = 0; ix < d; ix++) {
                auto t1 = exp_ikx[ix];
                for (int iy = 0; iy < d; iy++) {
                    auto t2 = exp_iky[iy] * t1;
                    for (int iz = 0; iz < d; iz++) {
                        auto t3 = exp_ikz[iz] * t2;
                        planewave_coeffs[ix * d * d + iy * d + iz] += q[i] * t3;
                    }
                }
            }
        }
    }

    void Ewald::init_interaction_matrix() {
        int d = k.size();

        interaction_matrix = std::vector<double>(std::pow(d, 3));

        for (int ix = 0; ix < d; ix++) {
            for (int iy = 0; iy < d; iy++) {
                for (int iz = 0; iz < d; iz++) {
                    double kx = k[ix];
                    double ky = k[iy];
                    double kz = k[iz];
                    double k2 = kx * kx + ky * ky + kz * kz;
                    if (k2 == 0.0) {
                        continue;
                    }
                    interaction_matrix[ix * d * d + iy * d + iz] = std::exp(-k2 / (4 * alpha * alpha)) / k2;
                }
            }
        }
    }

    double Ewald::compute_energy() {
        
        double E_short = 0.0;
        // short range part
        // #pragma omp parallel for reduction(+:E_short)
        for (int i = 0; i < neighbors.length; i++) {
            int i1 = neighbors.pairs[i][0];
            int i2 = neighbors.pairs[i][1];
            double r12 = neighbors.distances[i];
            E_short += q[i1] * q[i2] * std::erfc(alpha * r12) / r12;
        }

        E_short /= 2.0;

        double E_long = 0.0;
        
        // #pragma omp parallel for reduction(+:E_long)
        for (int i = 0; i < interaction_matrix.size(); i++) {
            E_long += std::real(interaction_matrix[i] * planewave_coeffs[i] * std::conj(planewave_coeffs[i]));
        }
        E_long *= 2 * M_PI / V;

        double E_self = 0.0;
        for (int i = 0; i < n_particles; i++) {
            E_self += - q[i] * q[i] * alpha / std::sqrt(M_PI);
        }

        return (E_short + E_long + E_self) / (eps);
    }

    void Ewald::collect_target_neighbors(const double trg_x, const double trg_y, const double trg_z) {
        target_neighbors.clear();
        target_distances.clear();
        for (int i = 0; i < n_particles; i++) {
            for (int mx = -1; mx <= 1; mx++) {
                for (int my = -1; my <= 1; my++) {
                    for (int mz = -1; mz <= 1; mz++) {
                        double r_ij = std::sqrt(std::pow(r[i * 3 + 0] + mx * L - trg_x, 2) + std::pow(r[i * 3 + 1] + my * L - trg_y, 2) + std::pow(r[i * 3 + 2] + mz * L - trg_z, 2));
                        if (r_ij <= r_c) {
                            target_neighbors.push_back(i);
                            target_distances.push_back(r_ij);
                        }
                    }
                }
            }
        }
    }

    double Ewald::compute_potential(const double trg_x, const double trg_y, const double trg_z) {
        double potential_short = 0.0;
        double potential_long = 0.0;

        // auto start = std::chrono::high_resolution_clock::now();

        // #pragma omp parallel for reduction(+:potential_short)
        for (int i = 0; i < target_neighbors.size(); i++) {
            int i1 = target_neighbors[i];
            double r12 = target_distances[i];
            double dE = q[i1] * std::erfc(alpha * r12) / r12;
            potential_short += dE;
        }

        // auto end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> duration = end - start;
        // std::cout << "time for short range part: " << duration.count() << " seconds" << std::endl;

        // start = std::chrono::high_resolution_clock::now();

        int d = k.size();
        target_planewave_coeffs = std::vector<std::complex<double>>(std::pow(d, 3));
        std::fill(target_planewave_coeffs.begin(), target_planewave_coeffs.end(), 0);

        // #pragma omp parallel for
        for (int ix = 0; ix < d; ix++) {
            for (int iy = 0; iy < d; iy++) {
                for (int iz = 0; iz < d; iz++) {
                    double kx = k[ix];
                    double ky = k[iy];
                    double kz = k[iz];
                    target_planewave_coeffs[ix * d * d + iy * d + iz] = std::exp( - std::complex<double>(0.0, 1.0) * (kx * trg_x + ky * trg_y + kz * trg_z));
                }
            }
        }

        // #pragma omp parallel for reduction(+:potential_long)
        for (int i = 0; i < target_planewave_coeffs.size(); i++) {
            potential_long += std::real(interaction_matrix[i] * planewave_coeffs[i] * std::conj(target_planewave_coeffs[i]));
        }
        potential_long *= 4 * M_PI / V;

        // end = std::chrono::high_resolution_clock::now();
        // duration = end - start;
        // std::cout << "time for long range part: " << duration.count() << " seconds" << std::endl;

        return (potential_short + potential_long) / (eps);
    }
} // namespace hpdmk