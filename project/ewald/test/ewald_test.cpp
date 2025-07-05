#include <sctl.hpp>
#include <iostream>

#include "ewald.h"

int main() {
    ewald::EwaldConfig ewald_config(10.0, 10.0, 10.0, 1.0, 1.0, 1.0);
    sctl::Vector<double> q(10);
    sctl::Vector<sctl::Vector<double>> r(10);
    for (int i = 0; i < 10; i++) {
        q[i] = 1.0;
        r[i] = sctl::Vector<double>(3);
        r[i][0] = i;
        r[i][1] = i;
        r[i][2] = i;
    }
    double energy = ewald_config.compute_energy(q, r);
    std::cout << "Energy: " << energy << std::endl;

    return 0;
}