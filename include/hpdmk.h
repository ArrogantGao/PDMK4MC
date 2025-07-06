#ifndef HPDMK_H
#define HPDMK_H

typedef struct hpdmk_params {
    int n_dim = 3;
} hpdmk_params;

#ifdef __cplusplus
extern "C" {
#endif

void hpdmk_init(hpdmk_params params);
void hpdmk_finalize();

#ifdef __cplusplus
}
#endif

#endif