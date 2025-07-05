#ifndef EWALD_HPP
#define EWALD_HPP

#include "ewald_export.h"

#include "sctl.hpp"

namespace ewald {

class EWALD_EXPORT EwaldConfig {
    public:
        EwaldConfig(const double Lx, const double Ly, const double Lz, const double s, const double alpha, const double eps);
        ~EwaldConfig();

        double Lx;
        double Ly;
        double Lz;
        double alpha;
        double eps;

        double V;
        double r_c;
        double k_c;
        sctl::Vector<double> kx;
        sctl::Vector<double> ky;
        sctl::Vector<double> kz;

        double compute_energy(const sctl::Vector<double> &q, const sctl::Vector<sctl::Vector<double>> &r);
};

}

#endif