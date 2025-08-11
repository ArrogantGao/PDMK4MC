#include <omp.h>
#include <hpdmk.h>
#include <ewald.hpp>

#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <fstream>

int main() {
    int n_src = 10000;
    double rho = 1.0;
    // double L = 10.0;
    double L = std::pow(n_src / rho, 1.0 / 3.0);
    double s = 3.0;
    double alpha = 1.0;

    std::vector<double> r(n_src * 3);
    std::vector<double> q(n_src);

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution(0, L);

    for (int i = 0; i < n_src; i++) {
        r[i * 3] = distribution(generator);
        r[i * 3 + 1] = distribution(generator);
        r[i * 3 + 2] = distribution(generator);
        q[i] = std::pow(-1, i) * 1.0;
    }

    hpdmk::Ewald ewald(L, s, alpha, 1.0, q, r, n_src);

    std::vector<int> n_threads = {1, 2, 4, 8, 16, 32, 64};
    std::vector<double> times_planewave(n_threads.size());
    std::vector<double> times_short(n_threads.size());
    std::vector<double> times_long(n_threads.size());
    std::vector<double> times_self(n_threads.size());
    int n_round = 10;

    int count = 0;
    ewald.init_neighbors();
    for (int n_thread : n_threads) {
        omp_set_num_threads(n_thread);
        double time_sum_planewave = 0.0;
        double time_sum_short = 0.0;
        double time_sum_long = 0.0;
        double time_sum_self = 0.0;
        for (int i = 0; i < n_round; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            ewald.init_planewave_coeffs();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time = end - start;
            time_sum_planewave += time.count();
        }
        for (int i = 0; i < n_round; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            ewald.compute_energy_short();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time = end - start;
            time_sum_short += time.count();
        }
        for (int i = 0; i < n_round; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            ewald.compute_energy_long();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time = end - start;
            time_sum_long += time.count();
        }
        for (int i = 0; i < n_round; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            ewald.compute_energy_self();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time = end - start;
            time_sum_self += time.count();
        }
        times_planewave[count] = time_sum_planewave / n_round;
        times_short[count] = time_sum_short / n_round;
        times_long[count] = time_sum_long / n_round;
        times_self[count] = time_sum_self / n_round;
        std::cout << "n_thread: " << n_thread << ", time_planewave: " << times_planewave[count] << " seconds, time_short: " << times_short[count] << " seconds, time_long: " << times_long[count] << " seconds, time_self: " << times_self[count] << " seconds, speed up_planewave: " << times_planewave[0] / times_planewave[count] << ", speed up_short: " << times_short[0] / times_short[count] << ", speed up_long: " << times_long[0] / times_long[count] << ", speed up_self: " << times_self[0] / times_self[count] << std::endl;
        count++;
    }

    return 0;
}