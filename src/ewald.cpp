#include "ewald.hpp"

#include <cmath>
#include <iostream>

namespace hpdmk {

    Ewald::Ewald(const double Lx, const double Ly, const double Lz, const double s, const double alpha, const double eps)
    : Lx(Lx), Ly(Ly), Lz(Lz), alpha(alpha), eps(eps), V(Lx * Ly * Lz), r_c(s / alpha), k_c(2 * alpha * s) {
        std::cout << "Initializing EwaldConfig class" << std::endl;

        const double dkx = 2 * M_PI / Lx;
        const double dky = 2 * M_PI / Ly;
        const double dkz = 2 * M_PI / Lz;

        const int nkx = std::ceil(k_c / dkx);
        const int nky = std::ceil(k_c / dky);
        const int nkz = std::ceil(k_c / dkz);

        kx = sctl::Vector<double>(2 * nkx + 1);
        ky = sctl::Vector<double>(2 * nky + 1);
        kz = sctl::Vector<double>(2 * nkz + 1);

        for (int i = 0; i < 2 * nkx + 1; i++) {
            kx[i] = (-nkx + i) * dkx;
        }
        for (int i = 0; i < 2 * nky + 1; i++) {
            ky[i] = (-nky + i) * dky;
        }
        for (int i = 0; i < 2 * nkz + 1; i++) {
            kz[i] = (-nkz + i) * dkz;
        }

        std::cout << "Ewald class initialized" << std::endl;
    }

    Ewald::~Ewald() {
        std::cout << "Destroying EwaldConfig class" << std::endl;
    }

    double Ewald::compute_energy(const sctl::Vector<double> &q, const sctl::Vector<sctl::Vector<double>> &r) {
        std::cout << "Computing energy" << std::endl;
        // fake compute energy
        return 0.0;
    }

}