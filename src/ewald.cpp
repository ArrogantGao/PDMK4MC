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

        // if (r_c > 0.5 * L) {
        //     throw std::invalid_argument("r_c is too large");
        // }
        
        int n = std::ceil(k_c / (2 * M_PI / L));
        std::vector<double> k(2 * n + 1);

        double dk = 2 * M_PI / L;

        for (int i = 0; i < 2 * n + 1; i++) {
            k[i] = ( - n + i ) * dk;
        }

        this->k = k;

        const double cutoff = this->r_c;
        options = VesinOptions();
        options.cutoff = cutoff;
        options.full = true;
        options.return_shifts = true;
        options.return_distances = true;
        options.return_vectors = false;

        // initialize planewave coefficients
        this->init_interaction_matrix();
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

    void Ewald::init_neighbors() {
        double box[3][3] = {
            {L, 0.0, 0.0},
            {0.0, L, 0.0},
            {0.0, 0.0, L},
        };

        auto r_array = new double[n_particles][3];
        for (int i = 0; i < n_particles; i++) {
            r_array[i][0] = r[i * 3 + 0];
            r_array[i][1] = r[i * 3 + 1];
            r_array[i][2] = r[i * 3 + 2];
        }

        VesinNeighborList neighbors;
        const char* error_message = nullptr;
        auto status = vesin_neighbors(
            r_array,
            n_particles,
            box,
            true,
            VesinDevice::VesinCPU,
            options,
            &neighbors,
            &error_message
        );

        this->neighbors = neighbors;
    }

    void Ewald::init_planewave_coeffs() {

        int d = k.size();
        int n = std::ceil(k_c / (2 * M_PI / L));
        planewave_coeffs = std::vector<std::complex<double>>(std::pow(d, 3));
        std::fill(planewave_coeffs.begin(), planewave_coeffs.end(), 0.0);

        int num_threads = omp_get_max_threads();

        int particles_per_thread = n_particles / num_threads;
        int remainder = n_particles % num_threads;
        
        std::vector<std::vector<std::complex<double>>> thread_results(num_threads, 
            std::vector<std::complex<double>>(std::pow(d, 3), 0.0));
        
        #pragma omp parallel
        {
            std::vector<std::complex<double>> local_exp_ikx(d);
            std::vector<std::complex<double>> local_exp_iky(d);
            std::vector<std::complex<double>> local_exp_ikz(d);
            
            int thread_id = omp_get_thread_num();
            auto& local_result = thread_results[thread_id];

            std::fill(local_result.begin(), local_result.end(), 0.0);
            
            int start_particle = thread_id * particles_per_thread;
            int end_particle = start_particle + particles_per_thread;
            
            if (thread_id < remainder) {
                start_particle += thread_id;
                end_particle += thread_id + 1;
            } else {
                start_particle += remainder;
                end_particle += remainder;
            }

            std::complex<double> exp_ikx0, exp_iky0, exp_ikz0;
            
            for (int i = start_particle; i < end_particle; i++) {
                double x = r[i * 3 + 0];
                double y = r[i * 3 + 1];
                double z = r[i * 3 + 2];

                auto k0 = 2 * M_PI / L;
                exp_ikx0 = std::exp( - std::complex<double>(0.0, 1.0) * k0 * x);
                exp_iky0 = std::exp( - std::complex<double>(0.0, 1.0) * k0 * y);
                exp_ikz0 = std::exp( - std::complex<double>(0.0, 1.0) * k0 * z);

                for (int a = -n; a <= n; a++) {
                    local_exp_ikx[a + n] = std::pow(exp_ikx0, a);
                    local_exp_iky[a + n] = std::pow(exp_iky0, a);
                    local_exp_ikz[a + n] = std::pow(exp_ikz0, a);
                }
                
                for (int ix = 0; ix < d; ix++) {
                    auto t1 = local_exp_ikx[ix];
                    for (int iy = 0; iy < d; iy++) {
                        auto t2 = local_exp_iky[iy] * t1;
                        for (int iz = 0; iz < d; iz++) {
                            auto t3 = local_exp_ikz[iz] * t2;
                            local_result[ix * d * d + iy * d + iz] += q[i] * t3;
                        }
                    }
                }
            }
        }
        
        for (int thread = 0; thread < num_threads; thread++) {
            for (int idx = 0; idx < std::pow(d, 3); idx++) {
                planewave_coeffs[idx] += thread_results[thread][idx];
            }
        }
    }

    double Ewald::compute_energy() {
        // initialize neighbors and planewave coefficients
        this->init_neighbors();
        this->init_planewave_coeffs();

        double E_short = this->compute_energy_short();
        double E_long = this->compute_energy_long();
        double E_self = this->compute_energy_self();

        double E = (E_short + E_long + E_self) / (eps);
        
        return E;
    }

    double Ewald::compute_energy_short() {
        double E_short = 0.0;
        #pragma omp parallel for reduction(+:E_short)
        for (int i = 0; i < neighbors.length; i++) {
            int i1 = neighbors.pairs[i][0];
            int i2 = neighbors.pairs[i][1];
            double r12 = neighbors.distances[i];
            E_short += q[i1] * q[i2] * std::erfc(alpha * r12) / r12;
        }
        E_short /= 2.0;
        return E_short;
    }

    double Ewald::compute_energy_long() {
        double E_long = 0.0;
        // #pragma omp parallel for reduction(+:E_long) // speed up is not obvious, disabled
        for (int i = 0; i < interaction_matrix.size(); i++) {
            E_long += std::real(interaction_matrix[i] * planewave_coeffs[i] * std::conj(planewave_coeffs[i]));
        }
        
        E_long *= 2 * M_PI / V;
        return E_long;
    }

    double Ewald::compute_energy_self() {
        double E_self = 0.0;
        // #pragma omp parallel for reduction(+:E_self) // slow down, disabled
        for (int i = 0; i < n_particles; i++) {
            E_self += - q[i] * q[i] * alpha / std::sqrt(M_PI);
        }
        return E_self;
    }

    void Ewald::init_target_neighbors(const double trg_x, const double trg_y, const double trg_z) {
        target_neighbors.clear();
        target_distances.clear();

        for (int i = 0; i < n_particles; i++) {
            auto xi = r[i * 3 + 0];
            auto yi = r[i * 3 + 1];
            auto zi = r[i * 3 + 2];

            for (int mx = -1; mx <= 1; mx++) {
                for (int my = -1; my <= 1; my++) {
                    for (int mz = -1; mz <= 1; mz++) {
                        double r_ij = std::sqrt(std::pow(xi + mx * L - trg_x, 2) + std::pow(yi + my * L - trg_y, 2) + std::pow(zi + mz * L - trg_z, 2));
                        if (r_ij <= r_c) {
                            target_neighbors.push_back(i);
                            target_distances.push_back(r_ij);
                        }
                    }
                }
            }
        }
    }

    void Ewald::init_target_planewave_coeffs(const double trg_x, const double trg_y, const double trg_z) {
        int d = k.size();
        int n = std::ceil(k_c / (2 * M_PI / L));

        target_planewave_coeffs = std::vector<std::complex<double>>(std::pow(d, 3));
        std::fill(target_planewave_coeffs.begin(), target_planewave_coeffs.end(), 0);

        double k0 = 2 * M_PI / L;
        auto exp_ikx0 = std::exp( - std::complex<double>(0.0, 1.0) * k0 * trg_x);
        auto exp_iky0 = std::exp( - std::complex<double>(0.0, 1.0) * k0 * trg_y);
        auto exp_ikz0 = std::exp( - std::complex<double>(0.0, 1.0) * k0 * trg_z);

        std::vector<std::complex<double>> local_exp_ikx(d);
        std::vector<std::complex<double>> local_exp_iky(d);
        std::vector<std::complex<double>> local_exp_ikz(d);

        for (int i = 0; i < d; i++) {
            local_exp_ikx[i] = std::pow(exp_ikx0, i - n);
            local_exp_iky[i] = std::pow(exp_iky0, i - n);
            local_exp_ikz[i] = std::pow(exp_ikz0, i - n);
        }

        for (int ix = 0; ix < d; ix++) {
            auto t1 = local_exp_ikx[ix];
            for (int iy = 0; iy < d; iy++) {
                auto t2 = local_exp_iky[iy] * t1;
                for (int iz = 0; iz < d; iz++) {
                    target_planewave_coeffs[ix * d * d + iy * d + iz] = local_exp_ikz[iz] * t2;
                }
            }
        }
    }

    double Ewald::compute_potential(const double trg_x, const double trg_y, const double trg_z) {

        this->init_target_neighbors(trg_x, trg_y, trg_z);
        this->init_target_planewave_coeffs(trg_x, trg_y, trg_z);

        return this->pure_compute_potential(trg_x, trg_y, trg_z);
    }

    double Ewald::pure_compute_potential(const double trg_x, const double trg_y, const double trg_z) {
        double potential_short = 0.0;
        double potential_long = 0.0;

        // #pragma omp parallel for reduction(+:potential_short)
        for (int i = 0; i < target_neighbors.size(); i++) {
            int i1 = target_neighbors[i];
            double r12 = target_distances[i];
            double dE = q[i1] * std::erfc(alpha * r12) / r12;
            potential_short += dE;
        }

        // #pragma omp parallel for reduction(+:potential_long)
        for (int i = 0; i < target_planewave_coeffs.size(); i++) {
            potential_long += std::real(interaction_matrix[i] * planewave_coeffs[i] * std::conj(target_planewave_coeffs[i]));
        }
        potential_long *= 4 * M_PI / V;

        return (potential_short + potential_long) / (eps);
    }
} // namespace hpdmk