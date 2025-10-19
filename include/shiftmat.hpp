#ifndef SHIFTMAT_HPP
#define SHIFTMAT_HPP

#include <hpdmk.h>
#include <complex>

#include <sctl.hpp>

namespace hpdmk {

    template <typename Real>
    struct ShiftMatrix {
        sctl::Vector<sctl::Vector<std::complex<Real>>> shift_vec;

        ShiftMatrix() {}
        ShiftMatrix(sctl::Long n_k, Real dk, Real L){
            shift_vec.ReInit(27);
            int d = 2 * n_k + 1;
            for (int flagz = -1; flagz <= 1; ++flagz) {
                for (int flagy = -1; flagy <= 1; ++flagy) {
                    for (int flagx = -1; flagx <= 1; ++flagx) {
                        sctl::Vector<std::complex<Real>> shift_vec_i;
                        shift_vec_i.ReInit(d * d * d);
                        for (int k = 0; k < d; ++k) {
                            for (int j = 0; j < d; ++j) {
                                for (int i = 0; i< d; ++i) {
                                    Real kx = (i - n_k) * dk * flagx;
                                    Real ky = (j - n_k) * dk * flagy;
                                    Real kz = (k - n_k) * dk * flagz;
                                    shift_vec_i[k * d * d + j * d + i] = std::exp(-std::complex<Real>(0, kx * L + ky * L + kz * L));
                                }
                            }
                        }
                        shift_vec[(flagz + 1) * 9 + (flagy + 1) * 3 + (flagx + 1)] = shift_vec_i;
                    }
                }
            }
        }

        sctl::Vector<std::complex<Real>>& select_shift_vec(int flagx, int flagy, int flagz){
            return shift_vec[(flagz + 1) * 9 + (flagy + 1) * 3 + (flagx + 1)];
        }
    };

}

#endif