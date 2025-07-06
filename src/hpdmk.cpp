#include <hpdmk.h>
#include <iostream>

// namespace hpdmk {
//     void hpdmk_init(hpdmk_params params) {
//         std::cout << "params.n_dim: " << params.n_dim << std::endl;
//         std::cout << "HPDMK initialized" << std::endl;
//     }

//     void hpdmk_finalize() {
//         std::cout << "HPDMK finalized" << std::endl;
//     }
// } // namespace hpdmk

void hpdmk_init(hpdmk_params params) {
    std::cout << "params.n_dim: " << params.n_dim << std::endl;
    std::cout << "HPDMK initialized" << std::endl;
}

void hpdmk_finalize() {
    std::cout << "HPDMK finalized" << std::endl;
}