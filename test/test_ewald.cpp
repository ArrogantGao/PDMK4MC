#include <ewald.hpp>
#include <sctl.hpp>

#include <iostream>

int main() {
    sctl::Vector<double> q(10);
    for (int i = 0; i < 10; i++) {
        q[i] = 1.0;
    }
    std::cout << "q: " << q << std::endl;

    sctl::Vector<sctl::Vector<double>> r(10);
    for (int i = 0; i < 10; i++) {
        r[i] = sctl::Vector<double>(3);
        r[i][0] = i;
        r[i][1] = i;
        r[i][2] = i;
    }
    std::cout << "r: " << r << std::endl;

    hpdmk::Ewald ewald_config(10.0, 10.0, 10.0, 1.0, 1.0, 1.0);
    double energy = ewald_config.compute_energy(q, r);
    std::cout << "energy: " << energy << std::endl;

    return 0;
}