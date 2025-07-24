#include <cmath>
#include <complex>
#include <iostream>
#include <vector>
#include <array>

#include <hpdmk.h>
#include <ewald.hpp>
#include <vesin.h>


namespace hpdmk {
    Ewald::Ewald(const double L, const double s, const double alpha, const double eps, const double *q, const double (*r)[3], const int n_particles)
        : L(L), s(s), alpha(alpha), eps(eps), V(L * L * L), r_c(s / alpha), k_c(2 * s * alpha), n_particles(n_particles) {
        
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

        // std::cout << "Ewald initialized" << std::endl;
    }

    double Ewald::compute_energy(const double *q, const double (*r)[3]) {
        
        double E_short = 0.0;
        // short range part
        for (int i = 0; i < neighbors.length; i++) {
            int i1 = neighbors.pairs[i][0];
            int i2 = neighbors.pairs[i][1];
            double r12 = neighbors.distances[i];
            double dE = q[i1] * q[i2] * std::erfc(alpha * r12) / r12;
            E_short += dE;
        }

        E_short /= 2.0;

        #ifdef DEBUG
            std::cout << "ewald short range part: " << E_short << std::endl;
        #endif

        double E_long = 0.0;
        
        double k_c2 = k_c * k_c;
        // long range part        
        for (double kx : k) {
            for (double ky : k) {
                for (double kz : k) {
                    if (kx == 0.0 && ky == 0.0 && kz == 0.0) {
                        continue;
                    }
                    double k2 = kx * kx + ky * ky + kz * kz;
                    if (k2 > k_c2) {
                        continue;
                    }
                    std::complex<double> rhok(0.0, 0.0);
                    for (int i = 0; i < n_particles; i++) {
                        rhok += q[i] * std::exp(std::complex<double>(0.0, 1.0) * (kx * r[i][0] + ky * r[i][1] + kz * r[i][2]));
                    }
                    double Ek = std::exp(-k2 / (4 * alpha * alpha)) * std::real(std::conj(rhok) * rhok) / k2;
                    E_long += Ek * 2 * M_PI / V;
                }
            }
        }

        #ifdef DEBUG
            std::cout << "ewald long range part: " << E_long << std::endl;
        #endif

        double E_self = 0.0;
        for (int i = 0; i < n_particles; i++) {
            E_self += - q[i] * q[i] * alpha / std::sqrt(M_PI);
        }

        #ifdef DEBUG
            std::cout << "ewald self energy: " << E_self << std::endl;
        #endif

        return (E_short + E_long + E_self) / (4 * M_PI * eps);
    }
} // namespace hpdmk