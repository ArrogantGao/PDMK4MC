#ifndef HPDMK_H
#define HPDMK_H

#include <mpi.h>

typedef enum : int {
    DIRECT = 1,
    PROXY = 2,
} hpdmk_init;

typedef struct HPDMKParams {
    int n_per_leaf = 200; // maximum number of particles per leaf
    int digits = 3; // number of digits of accuracy
    double L; // length of the box
    double prolate_order = 16; // order of the prolate polynomial
    hpdmk_init init = PROXY; // method to initialize the outgoing planewave, DIRECT means direct calculation on all nodes, PROXY for proxy charge
} HPDMKParams;

typedef void *hpdmk_tree;

#ifdef __cplusplus
extern "C" {
#endif

void hpdmk_tree_create(MPI_Comm comm, HPDMKParams params, int n_src, const double *r_src, const double *charge);

#ifdef __cplusplus
}
#endif

#endif