#include <sctl.hpp>
#include <hpdmk.h>
#include <utils.hpp>
#include <chrono>
#include <random>
#include <iostream>
#include <vector>

inline int offset(int i, int j, int k, int d) {
    return i * d * d + j * d + k;
}

void test_vector(){
    int d = 25;
    int n = d * d * d;

    auto generator = std::default_random_engine(0);
    auto distribution = std::uniform_int_distribution<int>(0, d - 1);

    //overhead of random generator
    auto start = std::chrono::high_resolution_clock::now();
    for (int round = 0; round < 100; ++round) {
        for (int l = 0; l < n; ++l) {
            int i = distribution(generator);
            int j = distribution(generator);
            int k = distribution(generator);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_overhead = end - start;
    std::cout << "overhead of random generator: " << time_overhead.count() << " seconds" << std::endl;

    // sctl vector
    sctl::Vector<double> tensor_sctl(n);
    // ordered access
    start = std::chrono::high_resolution_clock::now();
    for (int round = 0; round < 100; ++round) {
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                for (int k = 0; k < d; ++k) {
                    int idx = offset(i, j, k, d);
                    tensor_sctl[idx] = round + idx;
                }
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ordered = end - start;
    std::cout << "sctl vector ordered access time: " << time_ordered.count() << " seconds" << std::endl;

    // random access
    start = std::chrono::high_resolution_clock::now();
    for (int round = 0; round < 100; ++round) {
        for (int l = 0; l < n; ++l) {
            int i = distribution(generator);
            int j = distribution(generator);
            int k = distribution(generator);
            int idx = offset(i, j, k, d);
            tensor_sctl[idx] = round + idx;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_random = end - start;
    std::cout << "sctl vector random access time: " << time_random.count() - time_overhead.count() << " seconds" << std::endl;


    // std::vector
    std::vector<double> tensor_std(n);

    // ordered access

    start = std::chrono::high_resolution_clock::now();
    for (int round = 0; round < 100; ++round) {
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                for (int k = 0; k < d; ++k) {
                    int idx = offset(i, j, k, d);
                    tensor_std[idx] = round + idx;
                }
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ordered_std = end - start;
    std::cout << "std::vector ordered access time: " << time_ordered_std.count() << " seconds" << std::endl;

    // random access
    start = std::chrono::high_resolution_clock::now();
    for (int round = 0; round < 100; ++round) {
        for (int l = 0; l < n; ++l) {
            int i = distribution(generator);
            int j = distribution(generator);
            int k = distribution(generator);
            int idx = offset(i, j, k, d);
            tensor_std[idx] = round + idx;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_random_std = end - start;
    std::cout << "std::vector random access time: " << time_random_std.count() - time_overhead.count() << " seconds" << std::endl;

    // rank3tensor access time
    // ordered access
    hpdmk::Rank3Tensor<double> tensor_rank3(d, d, d);
    start = std::chrono::high_resolution_clock::now();
    for (int round = 0; round < 100; ++round) {
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                for (int k = 0; k < d; ++k) {
                    tensor_rank3(i, j, k) = round + i;
                }
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_rank3 = end - start;
    std::cout << "rank3tensor ordered access time with operator(): " << time_rank3.count() << " seconds" << std::endl;

    // ordered access with operator[]
    start = std::chrono::high_resolution_clock::now();
    for (int round = 0; round < 100; ++round) {
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                for (int k = 0; k < d; ++k) {
                    int idx = offset(i, j, k, d);
                    tensor_rank3[idx] = round + idx;
                }
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_rank3_ordered = end - start;
    std::cout << "rank3tensor ordered access time with operator[]: " << time_rank3_ordered.count() << " seconds" << std::endl;

    // random access
    start = std::chrono::high_resolution_clock::now();
    for (int round = 0; round < 100; ++round) {
        for (int l = 0; l < n; ++l) {
            int i = distribution(generator);
            int j = distribution(generator);
            int k = distribution(generator);
            tensor_rank3(i, j, k) = round + i;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_rank3_random = end - start;
    std::cout << "rank3tensor random access time: " << time_rank3_random.count() - time_overhead.count() << " seconds" << std::endl;
    
}

int main() {

    test_vector();

    return 0;
}