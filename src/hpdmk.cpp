#include <hpdmk.h>

#include <iostream>

namespace hpdmk {

    void foo() {
        std::cout << "HPDMK foo" << std::endl;
    }

} // namespace hpdmk

extern "C" {

void hpdmk_init(hpdmk_params params) {
    std::cout << "HPDMK initialized" << std::endl;
    std::cout << "params.n_dim: " << params.n_dim << std::endl;
    std::cout << "params.log_level: " << params.log_level << std::endl;
    hpdmk::foo();
}

void hpdmk_finalize() {
    std::cout << "HPDMK finalized" << std::endl;
}

} // extern "C"