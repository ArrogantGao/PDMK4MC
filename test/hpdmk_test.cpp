#include <gtest/gtest.h>
#include <hpdmk.h>

TEST(HPDMKTest, BasicAssertions) {
    hpdmk_params params;
    EXPECT_EQ(params.n_dim, 3);
    EXPECT_NO_THROW(hpdmk_init(params));
    EXPECT_NO_THROW(hpdmk_finalize());
}