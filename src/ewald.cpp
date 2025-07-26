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
    Ewald::Ewald(const double L, const double s, const double alpha, const double eps, const double *q, const double (*r)[3], const int n_particles)
        : L(L), s(s), alpha(alpha), eps(eps), V(L * L * L), r_c(s / alpha), k_c(2 * s * alpha), n_particles(n_particles), q(const_cast<double *>(q)), r(const_cast<double (*)[3]>(r)) {
        
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

        VesinNeighborList neighbors;
        const char* error_message = nullptr;
        auto status = vesin_neighbors(
            r,
            n_particles,
            box,
            true,
            VesinDevice::VesinCPU,
            options,
            &neighbors,
            &error_message
        );

        this->neighbors = neighbors;

        // initialize planewave coefficients
        this->init_interaction_matrix();
        this->init_planewave_coeffs();
    }

    void Ewald::init_planewave_coeffs() {

        int d = k.size();
        planewave_coeffs = CubicTensor<std::complex<double>>(d, sctl::Vector<std::complex<double>>(std::pow(d, 3)));
        planewave_coeffs.tensor *= 0;

        for (int i = 0; i < n_particles; i++) {
            double x, y, z;
            x = r[i][0];
            y = r[i][1];
            z = r[i][2];
            // #pragma omp parallel for
            for (int ix = 0; ix < d; ix++) {
                for (int iy = 0; iy < d; iy++) {
                    for (int iz = 0; iz < d; iz++) {
                        double kx = k[ix];
                        double ky = k[iy];
                        double kz = k[iz];
                        double k2 = kx * kx + ky * ky + kz * kz;
                        if (k2 > k_c * k_c) {
                            continue;
                        }
                        planewave_coeffs.value(ix, iy, iz) += q[i] * std::exp(-std::complex<double>(0.0, 1.0) * (kx * x + ky * y + kz * z));
                    }
                }
            }
        }
    }

    void Ewald::init_interaction_matrix() {
        int d = k.size();

        interaction_matrix = CubicTensor<double>(d, sctl::Vector<double>(std::pow(d, 3)));
        interaction_matrix.tensor *= 0;

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
                    interaction_matrix.value(ix, iy, iz) = std::exp(-k2 / (4 * alpha * alpha)) / k2;
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
        for (int i = 0; i < interaction_matrix.tensor.Dim(); i++) {
            E_long += std::real(interaction_matrix.tensor[i] * planewave_coeffs.tensor[i] * std::conj(planewave_coeffs.tensor[i]));
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
                        double r_ij = std::sqrt(std::pow(r[i][0] + mx * L - trg_x, 2) + std::pow(r[i][1] + my * L - trg_y, 2) + std::pow(r[i][2] + mz * L - trg_z, 2));
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
        target_planewave_coeffs = CubicTensor<std::complex<double>>(d, sctl::Vector<std::complex<double>>(std::pow(d, 3)));
        target_planewave_coeffs.tensor *= 0;

        // #pragma omp parallel for
        for (int ix = 0; ix < d; ix++) {
            for (int iy = 0; iy < d; iy++) {
                for (int iz = 0; iz < d; iz++) {
                    double kx = k[ix];
                    double ky = k[iy];
                    double kz = k[iz];
                    target_planewave_coeffs.value(ix, iy, iz) = std::exp( - std::complex<double>(0.0, 1.0) * (kx * trg_x + ky * trg_y + kz * trg_z));
                }
            }
        }

        #pragma omp parallel for reduction(+:potential_long)
        for (int i = 0; i < target_planewave_coeffs.tensor.Dim(); i++) {
            potential_long += std::real(interaction_matrix.tensor[i] * planewave_coeffs.tensor[i] * std::conj(target_planewave_coeffs.tensor[i]));
        }
        potential_long *= 4 * M_PI / V;

        // end = std::chrono::high_resolution_clock::now();
        // duration = end - start;
        // std::cout << "time for long range part: " << duration.count() << " seconds" << std::endl;

        return (potential_short + potential_long) / (eps);
    }
} // namespace hpdmk