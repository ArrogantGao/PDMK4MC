#ifndef HPDMK_H
#define HPDMK_H

#include <mpi.h>

typedef struct HPDMKParams {
    int n_per_leaf = 50; // maximum number of particles per leaf
    double eps = 1e-4; // tolerance for the result
    double L; // length of the box
    double prolate_order = 15; // order of the prolate polynomial
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