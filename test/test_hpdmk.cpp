#include <hpdmk.h>

int main() {
    hpdmk_params params;
    hpdmk_init(params);
    hpdmk_finalize();
    return 0;
}