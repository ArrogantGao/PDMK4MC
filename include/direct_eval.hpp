#ifndef DIRECT_EVAL_HPP
#define DIRECT_EVAL_HPP

#include <sctl.hpp>
#include <limits>

#define VECDIM 4

#ifdef __AVX512F__
#undef VECDIM
#define VECDIM 8
#endif

template <class Real, class VecType, sctl::Integer DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer SCDIM,
          class uKernel, sctl::Integer digits>
struct uKerHelper {
    template <class CtxType>
    static inline void Eval(VecType *vt, const VecType (&dX)[DIM], const Real *vs, const sctl::Integer nd,
                            const CtxType &ctx) {
        VecType M[KDIM0][KDIM1][SCDIM];
        uKernel::template uKerMatrix<digits>(M, dX, ctx);
        for (sctl::Integer i = 0; i < nd; i++) {
            const Real *vs_ = vs + i * SCDIM;
            for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) {
                VecType *vt_ = vt + (k1 * nd + i) * SCDIM;
                for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) {
                    const VecType vs0(vs_[(k0 * nd) * SCDIM + 0]);
                    vt_[0] = FMA(M[k0][k1][0], vs0, vt_[0]);
                    if (SCDIM == 2) {
                        const VecType vs1(vs_[(k0 * nd) * SCDIM + 1]);
                        vt_[0] = FMA(M[k0][k1][1], -vs1, vt_[0]);
                        vt_[1] = FMA(M[k0][k1][1], vs0, vt_[1]);
                        vt_[1] = FMA(M[k0][k1][0], vs1, vt_[1]);
                    }
                }
            }
        }
    }
    template <sctl::Integer nd, class CtxType>
    static inline void EvalND(VecType *vt, const VecType (&dX)[DIM], const Real *vs, const CtxType &ctx) {
        VecType M[KDIM0][KDIM1][SCDIM];
        uKernel::template uKerMatrix<digits>(M, dX, ctx);
        for (sctl::Integer i = 0; i < nd; i++) {
            const Real *vs_ = vs + i * SCDIM;
            for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) {
                VecType *vt_ = vt + (k1 * nd + i) * SCDIM;
                for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) {
                    const VecType vs0(vs_[(k0 * nd) * SCDIM + 0]);
                    vt_[0] = FMA(M[k0][k1][0], vs0, vt_[0]);
                    if (SCDIM == 2) {
                        const VecType vs1(vs_[(k0 * nd) * SCDIM + 1]);
                        vt_[0] = FMA(M[k0][k1][1], -vs1, vt_[0]);
                        vt_[1] = FMA(M[k0][k1][1], vs0, vt_[1]);
                        vt_[1] = FMA(M[k0][k1][0], vs1, vt_[1]);
                    }
                }
            }
        }
    }
};

template <class uKernel>
class GenericKernel : public uKernel {
    static constexpr sctl::Integer VecLen = uKernel::VecLen;
    using VecType = typename uKernel::VecType;
    using Real = typename uKernel::RealType;

    template <sctl::Integer K0, sctl::Integer K1, sctl::Integer Q, sctl::Integer D, class... T>
    static constexpr sctl::Integer get_DIM(void (*uKer)(VecType (&M)[K0][K1][Q], const VecType (&r)[D], T... args)) {
        return D;
    }
    template <sctl::Integer K0, sctl::Integer K1, sctl::Integer Q, sctl::Integer D, class... T>
    static constexpr sctl::Integer get_SCDIM(void (*uKer)(VecType (&M)[K0][K1][Q], const VecType (&r)[D], T... args)) {
        return Q;
    }
    template <sctl::Integer K0, sctl::Integer K1, sctl::Integer Q, sctl::Integer D, class... T>
    static constexpr sctl::Integer get_KDIM0(void (*uKer)(VecType (&M)[K0][K1][Q], const VecType (&r)[D], T... args)) {
        return K0;
    }
    template <sctl::Integer K0, sctl::Integer K1, sctl::Integer Q, sctl::Integer D, class... T>
    static constexpr sctl::Integer get_KDIM1(void (*uKer)(VecType (&M)[K0][K1][Q], const VecType (&r)[D], T... args)) {
        return K1;
    }

    static constexpr sctl::Integer DIM = get_DIM(uKernel::template uKerMatrix<0, GenericKernel>);
    static constexpr sctl::Integer SCDIM = get_SCDIM(uKernel::template uKerMatrix<0, GenericKernel>);
    static constexpr sctl::Integer KDIM0 = get_KDIM0(uKernel::template uKerMatrix<0, GenericKernel>);
    static constexpr sctl::Integer KDIM1 = get_KDIM1(uKernel::template uKerMatrix<0, GenericKernel>);

  public:
    GenericKernel() : ctx_ptr(this) {}

    static constexpr sctl::Integer CoordDim() { return DIM; }
    static constexpr sctl::Integer SrcDim() { return KDIM0 * SCDIM; }
    static constexpr sctl::Integer TrgDim() { return KDIM1 * SCDIM; }

    template <bool enable_openmp = false, sctl::Integer digits = -1>
    void Eval(sctl::Vector<sctl::Vector<Real>> &v_trg_, const sctl::Vector<Real> &r_trg,
              const sctl::Vector<Real> &r_src, const sctl::Vector<sctl::Vector<Real>> &v_src_,
              const sctl::Integer nd) const {
        if (nd == 1)
            EvalHelper<enable_openmp, digits, 1>(v_trg_, r_trg, r_src, v_src_, nd);
        else if (nd == 2)
            EvalHelper<enable_openmp, digits, 2>(v_trg_, r_trg, r_src, v_src_, nd);
        else if (nd == 3)
            EvalHelper<enable_openmp, digits, 3>(v_trg_, r_trg, r_src, v_src_, nd);
        else if (nd == 4)
            EvalHelper<enable_openmp, digits, 4>(v_trg_, r_trg, r_src, v_src_, nd);
        else if (nd == 5)
            EvalHelper<enable_openmp, digits, 5>(v_trg_, r_trg, r_src, v_src_, nd);
        else if (nd == 6)
            EvalHelper<enable_openmp, digits, 6>(v_trg_, r_trg, r_src, v_src_, nd);
        else if (nd == 7)
            EvalHelper<enable_openmp, digits, 7>(v_trg_, r_trg, r_src, v_src_, nd);
        else if (nd == 8)
            EvalHelper<enable_openmp, digits, 8>(v_trg_, r_trg, r_src, v_src_, nd);
        else
            EvalHelper<enable_openmp, digits, 0>(v_trg_, r_trg, r_src, v_src_, nd);
    }

  private:
    template <bool enable_openmp = false, sctl::Integer digits = -1, sctl::Integer ND = 0>
    void EvalHelper(sctl::Vector<sctl::Vector<Real>> &v_trg_, const sctl::Vector<Real> &r_trg,
                    const sctl::Vector<Real> &r_src, const sctl::Vector<sctl::Vector<Real>> &v_src_,
                    const sctl::Integer nd) const {
        static constexpr sctl::Integer digits_ =
            (digits == -1 ? (sctl::Integer)(sctl::TypeTraits<Real>::SigBits * 0.3010299957) : digits);
        auto uKerEval = [this](VecType *vt, const VecType(&dX)[DIM], const Real *vs, const sctl::Integer nd) {
            if (ND > 0)
                uKerHelper<Real, VecType, DIM, KDIM0, KDIM1, SCDIM, uKernel, digits_>::template EvalND<ND>(vt, dX, vs,
                                                                                                           *this);
            else
                uKerHelper<Real, VecType, DIM, KDIM0, KDIM1, SCDIM, uKernel, digits_>::Eval(vt, dX, vs, nd, *this);
        };

        const sctl::Long Ns = r_src.Dim() / DIM;
        const sctl::Long Nt = r_trg.Dim() / DIM;
        SCTL_ASSERT(r_trg.Dim() == Nt * DIM);
        SCTL_ASSERT(r_src.Dim() == Ns * DIM);

        sctl::Vector<sctl::Long> src_cnt(v_src_.Dim()), src_dsp(v_src_.Dim());
        src_dsp = 0;
        sctl::Vector<sctl::Long> trg_cnt(v_trg_.Dim()), trg_dsp(v_trg_.Dim());
        trg_dsp = 0;
        for (sctl::Integer i = 0; i < trg_cnt.Dim(); i++) {
            trg_cnt[i] = v_trg_[i].Dim() / Nt;
            trg_dsp[i] = (i ? trg_dsp[i - 1] + trg_cnt[i - 1] : 0);
        }
        for (sctl::Integer i = 0; i < src_cnt.Dim(); i++) {
            src_cnt[i] = v_src_[i].Dim() / Ns;
            src_dsp[i] = (i ? src_dsp[i - 1] + src_cnt[i - 1] : 0);
        }
        SCTL_ASSERT(src_cnt[src_cnt.Dim() - 1] + src_dsp[src_dsp.Dim() - 1] == SrcDim() * nd);
        SCTL_ASSERT(trg_cnt[trg_cnt.Dim() - 1] + trg_dsp[trg_dsp.Dim() - 1] == TrgDim() * nd);

        sctl::Vector<Real> v_src(Ns * SrcDim() * nd);
        for (sctl::Integer j = 0; j < src_cnt.Dim(); j++) {
            const sctl::Integer src_cnt_ = src_cnt[j];
            const sctl::Integer src_dsp_ = src_dsp[j];
            for (sctl::Integer k = 0; k < src_cnt_; k++) {
                for (sctl::Long i = 0; i < Ns; i++) {
                    v_src[i * SrcDim() * nd + src_dsp_ + k] = v_src_[j][i * src_cnt_ + k];
                }
            }
        }

        const sctl::Long NNt = ((Nt + VecLen - 1) / VecLen) * VecLen;
        {
            const sctl::Matrix<Real> Xs_(Ns, DIM, (sctl::Iterator<Real>)r_src.begin(), false);
            const sctl::Matrix<Real> Vs_(Ns, SrcDim() * nd, (sctl::Iterator<Real>)v_src.begin(), false);

            sctl::Matrix<Real> Xt_(DIM, NNt), Vt_(TrgDim() * nd, NNt);
            for (sctl::Long k = 0; k < DIM; k++) { // Set Xt_
                for (sctl::Long i = 0; i < Nt; i++) {
                    Xt_[k][i] = r_trg[i * DIM + k];
                }
                for (sctl::Long i = Nt; i < NNt; i++) {
                    Xt_[k][i] = 0;
                }
            }
            if (enable_openmp) { // Compute Vt_
#pragma omp parallel for schedule(static)
                for (sctl::Long t = 0; t < NNt; t += VecLen) {
                    VecType xt[DIM], vt[TrgDim() * nd];
                    for (sctl::Integer k = 0; k < TrgDim() * nd; k++)
                        vt[k] = VecType::Zero();
                    for (sctl::Integer k = 0; k < DIM; k++)
                        xt[k] = VecType::LoadAligned(&Xt_[k][t]);

                    for (sctl::Long s = 0; s < Ns; s++) {
                        VecType dX[DIM];
                        for (sctl::Integer k = 0; k < DIM; k++)
                            dX[k] = xt[k] - Xs_[s][k];
                        uKerEval(vt, dX, &Vs_[s][0], nd);
                    }
                    for (sctl::Integer k = 0; k < TrgDim() * nd; k++)
                        vt[k].StoreAligned(&Vt_[k][t]);
                }
            } else {
                for (sctl::Long t = 0; t < NNt; t += VecLen) {
                    VecType xt[DIM], vt[TrgDim() * nd];
                    for (sctl::Integer k = 0; k < TrgDim() * nd; k++)
                        vt[k] = VecType::Zero();
                    for (sctl::Integer k = 0; k < DIM; k++)
                        xt[k] = VecType::LoadAligned(&Xt_[k][t]);

                    for (sctl::Long s = 0; s < Ns; s++) {
                        VecType dX[DIM];
                        for (sctl::Integer k = 0; k < DIM; k++)
                            dX[k] = xt[k] - Xs_[s][k];
                        uKerEval(vt, dX, &Vs_[s][0], nd);
                    }
                    for (sctl::Integer k = 0; k < TrgDim() * nd; k++)
                        vt[k].StoreAligned(&Vt_[k][t]);
                }
            }

            for (sctl::Integer j = 0; j < trg_cnt.Dim(); j++) {
                const sctl::Integer trg_cnt_ = trg_cnt[j];
                const sctl::Integer trg_dsp_ = trg_dsp[j];
                for (sctl::Long i = 0; i < Nt; i++) {
                    for (sctl::Integer k = 0; k < trg_cnt_; k++) {
                        v_trg_[j][i * trg_cnt_ + k] += Vt_[trg_dsp_ + k][i] * uKernel::uKerScaleFactor();
                    }
                }
            }
        }
    }

    void *ctx_ptr;
};

template <class Real, sctl::Integer VecLen_, sctl::Integer chrg, sctl::Integer dipo, sctl::Integer poten,
          sctl::Integer grad>
struct Laplace3D {
    static constexpr sctl::Integer VecLen = VecLen_;
    using VecType = sctl::Vec<Real, VecLen>;
    using RealType = Real;

    VecType thresh2;

    static constexpr Real uKerScaleFactor() { return 1; }
    template <sctl::Integer digits, class CtxType>
    static inline void uKerMatrix(VecType (&M)[chrg + dipo][poten + grad][1], const VecType (&dX)[3],
                                  const CtxType &ctx) {
        using RealType = typename VecType::ScalarType;
        static constexpr sctl::Integer COORD_DIM = 3;

        const VecType &thresh2 = ctx.thresh2;

        const VecType R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const VecType Rinv = sctl::approx_rsqrt<digits>(R2, (R2 > thresh2));
        const VecType Rinv2 = Rinv * Rinv;
        const VecType Rinv3 = Rinv * Rinv2;

        if (chrg && poten) { // charge potential
            M[0][0][0] = Rinv;
        }

        if (chrg && grad) { // charge gradient
            for (sctl::Integer i = 0; i < COORD_DIM; i++) {
                M[0][poten + i][0] = -Rinv3 * dX[i];
            }
        }

        if (dipo && poten) { // dipole potential
            for (sctl::Integer i = 0; i < COORD_DIM; i++) {
                M[chrg + i][0][0] = Rinv3 * dX[i];
            }
        }

        if (dipo && grad) { // dipole gradient
            const VecType J0 = Rinv3 * Rinv2 * (RealType)(-3);
            for (sctl::Integer i = 0; i < COORD_DIM; i++) {
                const VecType J0_dXi = J0 * dX[i];
                for (sctl::Integer j = 0; j < COORD_DIM; j++) {
                    M[chrg + i][poten + j][0] = (i == j ? J0_dXi * dX[j] + Rinv3 : J0_dXi * dX[j]);
                }
            }
        }
    }
};

template <class Real, sctl::Integer VecLen, sctl::Integer chrg, sctl::Integer dipo, sctl::Integer poten,
          sctl::Integer grad>
static void EvalLaplace(sctl::Vector<sctl::Vector<Real>> &v_trg, const sctl::Vector<Real> &r_trg,
                        const sctl::Vector<Real> &r_src, const sctl::Vector<sctl::Vector<Real>> &v_src,
                        const sctl::Integer nd, const Real thresh, const sctl::Integer digits) {
    GenericKernel<Laplace3D<Real, VecLen, chrg, dipo, poten, grad>> ker;
    ker.thresh2 = thresh * thresh;

    if (digits < 0)
        ker.template Eval<true, -1>(v_trg, r_trg, r_src, v_src, nd);
    else if (digits <= 3)
        ker.template Eval<true, 3>(v_trg, r_trg, r_src, v_src, nd);
    else if (digits <= 6)
        ker.template Eval<true, 6>(v_trg, r_trg, r_src, v_src, nd);
    else if (digits <= 9)
        ker.template Eval<true, 9>(v_trg, r_trg, r_src, v_src, nd);
    else if (digits <= 12)
        ker.template Eval<true, 12>(v_trg, r_trg, r_src, v_src, nd);
    else if (digits <= 15)
        ker.template Eval<true, 15>(v_trg, r_trg, r_src, v_src, nd);
    else
        ker.template Eval<true, -1>(v_trg, r_trg, r_src, v_src, nd);
}

/* local kernel for the 1/r kernel */
template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>(), sctl::Integer digits = -1,
          sctl::Integer ndim>
void l3d_local_kernel_directcp_vec_cpp_helper(const int nd, const Real cutoff, const Real center, const Real d2max,
                                              const Real *r_src, const int nsrc, const Real *q_src,
                                              const Real *r_trg, const int ntrg, const Real *q_trg, Real& u) {
    static constexpr sctl::Integer VecLen = MaxVecLen;
    using Vec = sctl::Vec<Real, VecLen>;

    static constexpr sctl::Integer COORD_DIM = ndim; // ndim[0];
    constexpr sctl::Long nd_ = 1;                    // nd[0];
    sctl::Long Nsrc = nsrc;
    sctl::Long Ntrg = ntrg;
    sctl::Long Ntrg_ = ((Ntrg + MaxVecLen - 1) / MaxVecLen) * MaxVecLen;

    alignas(sizeof(Vec)) sctl::StaticArray<Real, 400 * COORD_DIM> buff0;
    alignas(sizeof(Vec)) sctl::StaticArray<Real, 400 * 1> buff1;
    sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
    sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
    if (Ntrg_ > 400) {
        Xt.ReInit(COORD_DIM, Ntrg_);
        Vt.ReInit(nd_, Ntrg_);
    }
    { // Set Xs, Vs, Xt, Vt
        for (sctl::Long t = 0; t < Ntrg; t++) {
            Xt[0][t] = r_trg[t * 3];
            Xt[1][t] = r_trg[t * 3 + 1];
            Xt[2][t] = r_trg[t * 3 + 2];
        }
        Vt = 0.0;
    }

    Vec thresh2 = std::numeric_limits<Real>::epsilon();
    Vec d2max_vec = d2max;
    Vec cen_vec = Real(- 0.5) * cutoff;
    Vec rsc_vec = Real(2.0) / cutoff;

    // load charge
    sctl::Matrix<Real> Vs_(Nsrc, nd_, sctl::Ptr2Itr<Real>((Real *)q_src, nd_ * Nsrc), false);
    // load source
    sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real *)r_src, COORD_DIM * Nsrc), false);
    for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
        Vec Xtrg[COORD_DIM];
        for (sctl::Integer k = 0; k < COORD_DIM; k++)
            Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);

        // load potential
        Vec Vtrg[nd_];
        for (long i = 0; i < nd_; i++)
            Vtrg[i] = Vec::LoadAligned(&Vt[i][t]);

        for (sctl::Long s = 0; s < Nsrc; s++) {
            Vec dX[COORD_DIM], R2 = Vec::Zero();
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
                dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
                R2 += dX[k] * dX[k];
            }

            const auto mask = (R2 > thresh2) & (R2 < d2max_vec);
            if (sctl::mask_popcnt_intrin(mask) == 0)
                continue;

            const Vec Rinv = sctl::approx_rsqrt<digits>(R2, mask);

            // evaluate the PSWF kernel
            const Vec xtmp = FMA(R2, Rinv, cen_vec) * rsc_vec;
            Vec ptmp;
            if constexpr (digits <= 3) {
                constexpr Real coefs[7] = {1.627823522210361e-01,  -4.553645597616490e-01, 4.171687104204163e-01, -7.073638602709915e-02, -8.957845614474928e-02, 2.617986644718201e-02, 9.633427876507601e-03};
                ptmp = EvalPolynomial(xtmp.get(), coefs[0], coefs[1], coefs[2], coefs[3], coefs[4], coefs[5], coefs[6]);
            } else if constexpr (digits <= 6) {
                constexpr Real coefs[13] = {5.482525801351582e-02,  -2.616592110444692e-01, 4.862652666337138e-01, -3.894296348642919e-01, 1.638587821812791e-02,  1.870328434198821e-01,-8.714171086568978e-02, -3.927020727017803e-02, 3.728187607052319e-02, 3.153734425831139e-03,  -8.651313377285847e-03, 1.725110090795567e-04, 1.034762385284044e-03};
                ptmp = EvalPolynomial(xtmp.get(), coefs[0], coefs[1], coefs[2], coefs[3], coefs[4], coefs[5], coefs[6], coefs[7], coefs[8], coefs[9], coefs[10], coefs[11], coefs[12]);
            } else if constexpr (digits <= 9) {
                constexpr Real coefs[19] = {1.835718730962269e-02,  -1.258015846164503e-01, 3.609487248584408e-01,  -5.314579651112283e-01, 3.447559412892380e-01,  9.664692318551721e-02,  -3.124274531849053e-01, 1.322460720579388e-01, 9.773007866584822e-02,  -1.021958831082768e-01, -3.812847450976566e-03, 3.858117355875043e-02, -8.728545924521301e-03, -9.401196355382909e-03, 4.024549377076924e-03,  1.512806105865091e-03, -9.576734877247042e-04, -1.303457547418901e-04, 1.100385844683190e-04};
                ptmp = EvalPolynomial(xtmp.get(), coefs[0], coefs[1], coefs[2], coefs[3], coefs[4], coefs[5], coefs[6], coefs[7], coefs[8], coefs[9], coefs[10], coefs[11], coefs[12], coefs[13], coefs[14], coefs[15], coefs[16], coefs[17], coefs[18]);
            } else if constexpr (digits <= 12) {
                constexpr Real coefs[25] = {6.262472576363448e-03,  -5.605742936112479e-02, 2.185890864792949e-01,  -4.717350304955679e-01, 5.669680214206270e-01,  -2.511606878849214e-01, -2.744523658778361e-01, 4.582527599363415e-01, -1.397724810121539e-01, -2.131762135835757e-01, 1.995489373508990e-01,  1.793390341864239e-02, -1.035055132403432e-01, 3.035606831075176e-02,  3.153931762550532e-02,  -2.033178627450288e-02, -5.406682731236552e-03, 7.543645573618463e-03,  1.437788047407851e-05,  -1.928370882351732e-03, 2.891658777328665e-04,  3.332996162099811e-04,  -8.397699195938912e-05, -3.015837377517983e-05, 9.640642701924662e-06};
                ptmp = EvalPolynomial(xtmp.get(), coefs[0], coefs[1], coefs[2], coefs[3], coefs[4], coefs[5], coefs[6], coefs[7], coefs[8], coefs[9], coefs[10], coefs[11], coefs[12], coefs[13], coefs[14], coefs[15], coefs[16], coefs[17], coefs[18], coefs[19], coefs[20], coefs[21], coefs[22], coefs[23], coefs[24]);
            }

            ptmp = ptmp * Rinv;

            for (long i = 0; i < nd_; i++)
                Vtrg[i] += Vec::Load1(&Vs_[s][i]) * ptmp;
        }

        for (long i = 0; i < nd_; i++)
            Vtrg[i].StoreAligned(&Vt[i][t]);
    }

    for (long j = 0; j < nd_; j++)
        #pragma omp simd reduction(+:u)
        for (long i = 0; i < Ntrg; i++)
            u += Vt[j][i] * q_trg[i];
}

template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>()>
void l3d_local_kernel_directcp_vec_cpp(const int nd, const int ndim, const int n_digits, const Real cutoff, const Real center, const Real d2max, const Real* r_src, const int nsrc, const Real* q_src, const Real* r_trg, const int ntrg, const Real* q_trg, Real& u) {
    if (n_digits <= 3)
        l3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 3, 3>(nd, cutoff, center, d2max, r_src, nsrc, q_src, r_trg, ntrg, q_trg, u);
    else if (n_digits <= 6)
        l3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 6, 3>(nd, cutoff, center, d2max, r_src, nsrc, q_src, r_trg, ntrg, q_trg, u);
    else if (n_digits <= 9)
        l3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 9, 3>(nd, cutoff, center, d2max, r_src, nsrc, q_src, r_trg, ntrg, q_trg, u);
    else if (n_digits <= 12)
        l3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 12, 3>(nd, cutoff, center, d2max, r_src, nsrc, q_src, r_trg, ntrg, q_trg, u);
    else
        throw std::runtime_error("n_digits is not supported");
}

template <typename Real>
Real direct_eval(const Real* r_src, const Real* q_src, const int n_trg, const Real* r_trg, const Real* q_trg, Real cutoff, int n_digits) {

    constexpr int VECWIDTH = std::is_same_v<Real, float> ? 2 * VECDIM : VECDIM;
    const int nd = 1;
    const int ndim = 3;
    const int nsrc = 1;
    const int ntrg = n_trg;
    const Real center = 0.0;
    const Real d2max = cutoff * cutoff;

    Real u = 0;
    l3d_local_kernel_directcp_vec_cpp<Real, VECWIDTH>(nd, ndim, n_digits, cutoff, center, d2max, r_src, nsrc, q_src, r_trg, ntrg, q_trg, u);

    return u;
}

#endif
