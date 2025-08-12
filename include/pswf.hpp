#ifndef PSWF_HPP
#define PSWF_HPP

#include <iostream>
#include <unordered_map>
#include <vector>
#define MAX_MONO_ORDER 20


#ifdef __cplusplus
extern "C" {
#endif
    // blas and lapack functions
    
    void dgesdd_(char* jobz, int* m, int* n, double* a, int* lda, double* s, double* u,
        int* ldu, double* vt, int* ldvt, double* work, int* lwork, int* iwork, int* info);
    
    // these functions have been declared in sctl.hpp
    // void dgemm_(char* TransA, char* TransB, int* M, int* N, int* K, double* alpha, double* A, int* lda, double* B, int* ldb, double* beta, double* C, int* ldc);
    // void dgesvd_(char* jobu, char* jobvt, int* m, int* n, double* a, int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt, double* work, int* lwork, int* info);
#ifdef __cplusplus
}
#endif

namespace hpdmk{
    // prolate functions
    double prolc180(double eps);
    double prolc180_der3(double eps);

    double prolate0_lambda(double c);

    // prolate0 functor
    struct Prolate0Fun;

    double prolate0_eval_derivative(double c, double x);
    /*
    evaluate prolate0c at x, i.e., \psi_0^c(x)
    */
    double prolate0_eval(double c, double x);

    /*
    evaluate prolate0c function integral of \int_0^r \psi_0^c(x) dx
    */
    double prolate0_int_eval(double c, double r);

    template <typename Real>
    struct PolyFun {
        PolyFun() = default;
    
        inline PolyFun(std::vector<double> coeffs_) : coeffs(coeffs_) {
            order = coeffs.size();
        }
    
        inline Real eval(Real x) const {
            Real val = 0;
            if (x >= 1.0) return 0.0;
            for (int i = 0; i < order; i++) {
                val = val * x + coeffs[i];
            }
            return val;
        }
    
        int order;
        std::vector<double> coeffs;
    };
    

    // approximation functions
    template <typename Real>
    PolyFun<Real> approximate_real_poly(double tol, int order);
    template <typename Real>
    PolyFun<Real> approximate_fourier_poly(double tol, int order);
}

#endif  // PSWF_H
