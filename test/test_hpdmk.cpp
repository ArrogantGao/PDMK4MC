#include <hpdmk.h>

#include <doctest.h>

TEST_CASE("hpdmk_init") {
    hpdmk_params params;
    hpdmk_init(params);
}

TEST_CASE("hpdmk_finalize") {
    hpdmk_finalize();
}