#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ARCH22_REFERENCE_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ARCH22_REFERENCE_H

#include "kernel_operator.h"
#include "../chunk_gated_delta_rule_bwd_finalize_struct.h"

namespace GDN {

// A2 的设备正确性参考主体。独立 GEMM 的原始结果由 Cube 提供，
// 其余公式仍以标量循环表达；不作为性能完成版本。
template <typename DT, bool USE_NORM>
class ChunkGatedDeltaRuleBwdFinalizeReference {
    static constexpr int64_t D = 128;
    static constexpr int64_t BT = 64;
    static constexpr int64_t PLANE = BT * D;
    static constexpr int64_t RAW_W = BT * BT;
    static constexpr int64_t RAW_Q = RAW_W + BT * D;
    static constexpr int64_t RAW_K = RAW_Q + BT * D;
    static constexpr int64_t RAW_V = RAW_K + BT * D;
    static constexpr int64_t RAW_ELEMS = RAW_V + BT * BT;
    static constexpr int64_t UB_ELEMS = 32768;
    static constexpr int64_t INPUT_A = 0;
    static constexpr int64_t INPUT_K = 4096;
    static constexpr int64_t INPUT_Q = 12288;
    static constexpr int64_t INPUT_P = 20480;
    static constexpr int64_t INPUT_ELEMS = 28672;
    enum Region : int64_t {
        ACC_Q, ACC_K, F, DW, DA, TMP, MM, DJ, PSI, MK, DG, REGION_COUNT
    };

public:
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR x, GM_ADDR dO, GM_ADDR p,
        GM_ADDR g, GM_ADDR beta, GM_ADDR h, GM_ADDR dh, GM_ADDR a,
        GM_ADDR qRstd, GM_ADDR kRstd, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dBeta, GM_ADDR dG,
        GM_ADDR workspace, GM_ADDR rawWorkspace,
        const ChunkGatedDeltaRuleBwdFinalizeTilingData *tiling)
    {
        t_ = tiling;
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(k));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(v));
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(x));
        do_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dO));
        p_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(p));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(h));
        dh_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dh));
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(a));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(g));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(beta));
        dq_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dq));
        dk_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dk));
        dv_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dv));
        dbeta_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dBeta));
        dg_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dG));
        if constexpr (USE_NORM) {
            qRstd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qRstd));
            kRstd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(kRstd));
        }
        if (t_->isVariable != 0) {
            cu_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
            indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
        }
        w_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        raw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rawWorkspace));
        pipe_.InitBuffer(ubBuf_, UB_ELEMS * sizeof(float));
        ub_ = ubBuf_.Get<float>();
        pipe_.InitBuffer(inputBuf_, INPUT_ELEMS * sizeof(DT));
        input_ = inputBuf_.Get<DT>();
        pipe_.InitBuffer(scratchBuf_, 6 * D * sizeof(float));
        scratch_ = scratchBuf_.Get<float>();
        const int64_t pair = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        physicalAiv_ = pair * 2 + AscendC::GetSubBlockIdx();
        wsBase_ = physicalAiv_ * REGION_COUNT * PLANE;
    }

    __aicore__ inline void Process()
    {
        const int64_t core = physicalAiv_;
        const int64_t cores = 2 * static_cast<int64_t>(AscendC::GetBlockNum());
        for (int64_t task = core; task < t_->taskNum; task += cores) {
            int64_t b = 0, start = 0, m = 0, ci = 0;
            const int64_t chunkTask = task / t_->HK;
            const int64_t hk = task % t_->HK;
            if (t_->isVariable == 0) {
                b = chunkTask / t_->chunkNumForT;
                ci = chunkTask % t_->chunkNumForT;
                start = ci * BT;
                m = Min(BT, t_->T - start);
            } else {
                const int64_t seq = indices_.GetValue(chunkTask * 2);
                const int64_t localChunk = indices_.GetValue(chunkTask * 2 + 1);
                if (seq < 0 || seq >= t_->seqNum || localChunk < 0) {
                    continue;
                }
                start = cu_.GetValue(seq) + localChunk * BT;
                m = Min(BT, cu_.GetValue(seq + 1) - start);
                ci = chunkTask;
            }
            if (m <= 0) {
                continue;
            }
            start_ = start;
            LoadQK(b, hk, start, m);
            for (int64_t i = 0; i < m; ++i) {
                for (int64_t d = 0; d < D; ++d) {
                    Put(ACC_Q, i * D + d, 0.0f);
                    Put(ACC_K, i * D + d, 0.0f);
                }
            }
            for (int64_t r = 0; r < t_->headRatio; ++r) {
                ProcessValueHead(b, hk, hk * t_->headRatio + r, start, ci, m);
            }
            for (int64_t i = 0; i < m; ++i) {
                float qDot = 0.0f, kDot = 0.0f;
                if constexpr (USE_NORM) {
                    for (int64_t d = 0; d < D; ++d) {
                        qDot += Get(ACC_Q, i * D + d) * Q(b, hk, start + i, d);
                        kDot += Get(ACC_K, i * D + d) * K(b, hk, start + i, d);
                    }
                }
                const int64_t scalar = (b * t_->HK + hk) * t_->T + start + i;
                const float qr = USE_NORM ? qRstd_.GetValue(scalar) : 1.0f;
                const float kr = USE_NORM ? kRstd_.GetValue(scalar) : 1.0f;
                for (int64_t d = 0; d < D; ++d) {
                    float dqVal = Get(ACC_Q, i * D + d);
                    float dkVal = Get(ACC_K, i * D + d);
                    if constexpr (USE_NORM) {
                        dqVal = (dqVal - Q(b, hk, start + i, d) * qDot) * qr;
                        dkVal = (dkVal - K(b, hk, start + i, d) * kDot) * kr;
                    }
                    const int64_t off = VectorOffset(b, hk, start + i, d, t_->HK);
                    dq_.SetValue(off, FloatToBf16(dqVal));
                    dk_.SetValue(off, FloatToBf16(dkVal));
                }
            }
        }
    }

private:
    __aicore__ inline static float Bf16ToFloat(DT value)
    {
        union HalfBits {
            __aicore__ HalfBits() {}
            DT bf;
            uint16_t bits;
        } half;
        union FloatBits {
            __aicore__ FloatBits() {}
            uint32_t bits;
            float value;
        } full;
        half.bf = value;
        full.bits = static_cast<uint32_t>(half.bits) << 16;
        return full.value;
    }
    __aicore__ inline static DT FloatToBf16(float value)
    {
        union FloatBits {
            __aicore__ FloatBits() {}
            uint32_t bits;
            float value;
        } full;
        union HalfBits {
            __aicore__ HalfBits() {}
            DT bf;
            uint16_t bits;
        } half;
        full.value = value;
        const uint32_t bias = 0x7FFFu + ((full.bits >> 16) & 1u);
        half.bits = static_cast<uint16_t>((full.bits + bias) >> 16);
        return half.bf;
    }
    __aicore__ inline static int64_t Min(int64_t lhs, int64_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    // 所有已验收训练 gate 的指数输入均为非正。正输入用倒数扩展；
    // 每 0.5 做一次范围缩减，[-0.5,0] 上采用 9 阶 Taylor。
    __aicore__ inline static float Exp(float x)
    {
        const bool positive = x > 0.0f;
        float u = positive ? -x : x;
        float factor = 1.0f;
        while (u < -0.5f) {
            u += 0.5f;
            factor *= 0.6065306597126334f;
        }
        const float polynomial = 1.0f + u * (1.0f + u * (0.5f + u *
            (0.16666666666666666f + u * (0.041666666666666664f + u *
            (0.008333333333333333f + u * (0.001388888888888889f + u *
            (0.0001984126984126984f + u * (0.0000248015873015873f +
            u * 0.0000027557319223985893f))))))));
        const float result = factor * polynomial;
        return positive ? 1.0f / result : result;
    }

    __aicore__ inline int64_t VectorOffset(
        int64_t b, int64_t head, int64_t token, int64_t dim, int64_t heads) const
    {
        return ((b * heads + head) * t_->T + token) * D + dim;
    }
    __aicore__ inline int64_t ScalarOffset(int64_t b, int64_t hv, int64_t token) const
    {
        return (b * t_->HV + hv) * t_->T + token;
    }
    __aicore__ inline int64_t StateOffset(
        int64_t b, int64_t hv, int64_t ci, int64_t kd, int64_t vd) const
    {
        const int64_t nt = t_->isVariable != 0 ? t_->totalChunkNum : t_->chunkNumForT;
        return ((((b * t_->HV + hv) * nt + ci) * D + kd) * D + vd);
    }
    __aicore__ inline float Q(int64_t b, int64_t hk, int64_t tok, int64_t d) const
    {
        return Bf16ToFloat(input_.GetValue(INPUT_Q + (tok - start_) * D + d));
    }
    __aicore__ inline float K(int64_t b, int64_t hk, int64_t tok, int64_t d) const
    {
        return Bf16ToFloat(input_.GetValue(INPUT_K + (tok - start_) * D + d));
    }
    __aicore__ inline float V(int64_t b, int64_t hv, int64_t tok, int64_t d) const
    {
        return Bf16ToFloat(v_.GetValue(VectorOffset(b, hv, tok, d, t_->HV)));
    }
    __aicore__ inline float X(int64_t b, int64_t hv, int64_t tok, int64_t d) const
    {
        return Bf16ToFloat(x_.GetValue(VectorOffset(b, hv, tok, d, t_->HV)));
    }
    __aicore__ inline float DO(int64_t b, int64_t hv, int64_t tok, int64_t d) const
    {
        return Bf16ToFloat(do_.GetValue(VectorOffset(b, hv, tok, d, t_->HV)));
    }
    __aicore__ inline float P(int64_t b, int64_t hv, int64_t tok, int64_t d) const
    {
        return Bf16ToFloat(input_.GetValue(INPUT_P + (tok - start_) * D + d));
    }
    __aicore__ inline float A(int64_t b, int64_t hv, int64_t tok, int64_t col) const
    {
        return Bf16ToFloat(input_.GetValue(INPUT_A + (tok - start_) * BT + col));
    }
    __aicore__ inline float G(int64_t b, int64_t hv, int64_t tok) const
    {
        return g_.GetValue(ScalarOffset(b, hv, tok));
    }
    __aicore__ inline float Beta(int64_t b, int64_t hv, int64_t tok) const
    {
        return beta_.GetValue(ScalarOffset(b, hv, tok));
    }
    __aicore__ inline float Get(Region region, int64_t index) const
    {
        if (region == ACC_Q || region == ACC_K || region == DG) {
            return w_.GetValue(wsBase_ + static_cast<int64_t>(region) * PLANE + index);
        }
        return ub_.GetValue(UbOffset(region) + index);
    }
    __aicore__ inline void Put(Region region, int64_t index, float value)
    {
        if (region == ACC_Q || region == ACC_K || region == DG) {
            w_.SetValue(wsBase_ + static_cast<int64_t>(region) * PLANE + index, value);
        } else {
            ub_.SetValue(UbOffset(region) + index, value);
        }
    }
    __aicore__ inline static int64_t UbOffset(Region region)
    {
        switch (region) {
            case F: return 0;
            case DW: return 4096;
            case DA: return 12288;
            case TMP: return 16384;
            case MM: return 20480;
            case DJ: return 24576;
            case PSI: return 4096;  // dW 最后一次被 dJ 消费后复用
            case MK: return 12288;  // dA/TMP 最后一次被 M 消费后复用
            default: return 0;
        }
    }
    __aicore__ inline void WaitInputCopy()
    {
        auto ev = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(ev);
    }
    __aicore__ inline void LoadQK(int64_t b, int64_t hk, int64_t start, int64_t m)
    {
        const int64_t offset = VectorOffset(b, hk, start, 0, t_->HK);
        AscendC::DataCopy(input_[INPUT_K], k_[offset], m * D);
        AscendC::DataCopy(input_[INPUT_Q], q_[offset], m * D);
        WaitInputCopy();
    }
    __aicore__ inline void LoadAP(int64_t b, int64_t hv, int64_t start, int64_t m)
    {
        const int64_t aOffset = ((b * t_->HV + hv) * t_->T + start) * BT;
        const int64_t pOffset = VectorOffset(b, hv, start, 0, t_->HV);
        AscendC::DataCopy(input_[INPUT_A], a_[aOffset], m * BT);
        AscendC::DataCopy(input_[INPUT_P], p_[pOffset], m * D);
        WaitInputCopy();
    }

    __aicore__ inline void ProcessValueHead(
        int64_t b, int64_t hk, int64_t hv, int64_t start, int64_t ci, int64_t m)
    {
        const float scale = t_->scale;
        LoadAP(b, hv, start, m);
        const float gLast = G(b, hv, start + m - 1);
        const int64_t chunkTask =
            t_->isVariable != 0 ? ci : b * t_->chunkNumForT + ci;
        const int64_t rawBase = (chunkTask * t_->HV + hv) * RAW_ELEMS;
        float stateDot = 0.0f;
        for (int64_t kd = 0; kd < D; ++kd) {
            for (int64_t vd = 0; vd < D; ++vd) {
                const int64_t off = StateOffset(b, hv, ci, kd, vd);
                stateDot += Bf16ToFloat(h_.GetValue(off)) *
                            Bf16ToFloat(dh_.GetValue(off));
            }
        }
        // F = scale * tril((dO X^T) * exp(g_i-g_j)).
        for (int64_t i = 0; i < m; ++i) {
            const float gi = G(b, hv, start + i);
            for (int64_t j = 0; j <= i; ++j) {
                const float dot = raw_.GetValue(rawBase + i * BT + j);
                Put(F, i * BT + j, scale * dot * Exp(gi - G(b, hv, start + j)));
            }
        }
        // dW = -P H^T.
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t kd = 0; kd < D; ++kd) {
                Put(DW, i * D + kd, -raw_.GetValue(rawBase + RAW_W + i * D + kd));
            }
        }
        // dA_raw 的严格下三角，再计算 A^T dA_raw A^T。
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int64_t i = 0; i < m; ++i) {
            auto k32 = scratch_;
            auto scaledBeta = scratch_[D];
            auto scaledGate = scratch_[2 * D];
            auto product = scratch_[3 * D];
            auto sum = scratch_[4 * D];
            auto reduceWork = scratch_[5 * D];
            for (int64_t j = 0; j < m; ++j) {
                float value = 0.0f;
                if (j < i) {
                    const float bj = Beta(b, hv, start + j);
                    const float ej = Exp(G(b, hv, start + j));
                    AscendC::Cast(k32, input_[INPUT_K + j * D], AscendC::RoundMode::CAST_NONE, D);
                    AscendC::Muls(scaledBeta, ub_[UbOffset(DW) + i * D], bj, D);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(scaledGate, scaledBeta, ej, D);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(product, scaledGate, k32, D);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::ReduceSum(sum, product, reduceWork, static_cast<int32_t>(D));
                    AscendC::PipeBarrier<PIPE_ALL>();
                    value = sum.GetValue(0);
                    value += bj * raw_.GetValue(rawBase + RAW_V + i * BT + j);
                }
                Put(DA, i * BT + j, value);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int64_t i = 0; i < m; ++i) {
            auto tmpRow = ub_[UbOffset(TMP) + i * BT];
            auto scaled = scratch_;
            AscendC::Duplicate(tmpRow, 0.0f, BT);
            AscendC::PipeBarrier<PIPE_V>();
            for (int64_t z = 0; z < m; ++z) {
                const float az = A(b, hv, start + z, i);
                AscendC::Muls(scaled, ub_[UbOffset(DA) + z * BT], az, BT);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(tmpRow, tmpRow, scaled, BT);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int64_t i = 0; i < m; ++i) {
            const float gi = G(b, hv, start + i);
            auto a32 = scratch_;
            auto product = scratch_[D];
            auto sum = scratch_[2 * D];
            auto reduceWork = scratch_[3 * D];
            for (int64_t j = 0; j < m; ++j) {
                float value = 0.0f;
                if (j < i) {
                    AscendC::Cast(a32, input_[INPUT_A + j * BT], AscendC::RoundMode::CAST_NONE, BT);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(product, ub_[UbOffset(TMP) + i * BT], a32, BT);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::ReduceSum(sum, product, reduceWork, static_cast<int32_t>(BT));
                    AscendC::PipeBarrier<PIPE_ALL>();
                    value = -sum.GetValue(0);
                    value *= Exp(gi - G(b, hv, start + j));
                }
                Put(MM, i * BT + j, value);
            }
        }
        // dJ=A^T dW；完整消费 dW 后才可复用其 UB 地址。
        for (int64_t i = 0; i < m; ++i) {
            auto dj = ub_[UbOffset(DJ) + i * D];
            auto tmp = scratch_;
            AscendC::Duplicate(dj, 0.0f, D);
            AscendC::PipeBarrier<PIPE_V>();
            for (int64_t z = 0; z < m; ++z) {
                const float az = A(b, hv, start + z, i);
                AscendC::Muls(tmp, ub_[UbOffset(DW) + z * D], az, D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(dj, dj, tmp, D);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        // Psi=A^T P，MK=M K；分别复用已结束生命周期的 dW、dA/TMP。
        for (int64_t i = 0; i < m; ++i) {
            auto psi = ub_[UbOffset(PSI) + i * D];
            auto mk = ub_[UbOffset(MK) + i * D];
            auto p32 = scratch_;
            auto k32 = scratch_[D];
            auto pScaled = scratch_[2 * D];
            auto kScaled = scratch_[3 * D];
            AscendC::Duplicate(psi, 0.0f, D);
            AscendC::Duplicate(mk, 0.0f, D);
            AscendC::PipeBarrier<PIPE_V>();
            for (int64_t z = 0; z < m; ++z) {
                const float az = A(b, hv, start + z, i);
                const float mz = Get(MM, i * BT + z);
                AscendC::Cast(p32, input_[INPUT_P + z * D], AscendC::RoundMode::CAST_NONE, D);
                AscendC::Cast(k32, input_[INPUT_K + z * D], AscendC::RoundMode::CAST_NONE, D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(pScaled, p32, az, D);
                AscendC::Muls(kScaled, k32, mz, D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(psi, psi, pScaled, D);
                AscendC::Add(mk, mk, kScaled, D);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        float dkhDotK = 0.0f;
        for (int64_t i = 0; i < m; ++i) {
            const float gi = G(b, hv, start + i);
            const float ei = Exp(gi);
            const float bi = Beta(b, hv, start + i);
            auto dqIntraVec = scratch_;
            auto dkIntraVec = scratch_[D];
            auto dk2LastVec = scratch_[2 * D];
            auto k32 = scratch_[3 * D];
            auto q32 = scratch_[4 * D];
            auto scaled = scratch_[5 * D];
            AscendC::Duplicate(dqIntraVec, 0.0f, D);
            AscendC::Duplicate(dkIntraVec, 0.0f, D);
            AscendC::Duplicate(dk2LastVec, 0.0f, D);
            AscendC::PipeBarrier<PIPE_V>();
            for (int64_t z = 0; z < m; ++z) {
                AscendC::Cast(k32, input_[INPUT_K + z * D], AscendC::RoundMode::CAST_NONE, D);
                if (z >= i) {
                    AscendC::Cast(q32, input_[INPUT_Q + z * D], AscendC::RoundMode::CAST_NONE, D);
                }
                AscendC::PipeBarrier<PIPE_V>();
                if (z <= i) {
                    AscendC::Muls(scaled, k32, Get(F, i * BT + z), D);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(dqIntraVec, dqIntraVec, scaled, D);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                if (z >= i) {
                    AscendC::Muls(scaled, q32, Get(F, z * BT + i), D);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(dkIntraVec, dkIntraVec, scaled, D);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                const float mkBeta = Get(MM, z * BT + i) * Beta(b, hv, start + z);
                AscendC::Muls(scaled, k32, mkBeta, D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(dk2LastVec, dk2LastVec, scaled, D);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            float dqDotQ = 0.0f, dk1DotK = 0.0f, djDotJ = 0.0f;
            float db = 0.0f;
            for (int64_t d = 0; d < D; ++d) {
                float dqState = raw_.GetValue(rawBase + RAW_Q + i * D + d);
                float dkState = raw_.GetValue(rawBase + RAW_K + i * D + d);
                dqState *= scale * ei;
                dkState *= Exp(gLast - gi);
                const float dqIntra = dqIntraVec.GetValue(d);
                const float dkIntra = dkIntraVec.GetValue(d);
                const float dk2Last = dk2LastVec.GetValue(d);
                const float dqVal = dqState + dqIntra;
                const float dk1 = dkState + dkIntra;
                const float dk2 = bi * ei * Get(DJ, i * D + d) +
                                  bi * Get(MK, i * D + d) + dk2Last;
                const float kval = K(b, hk, start + i, d);
                Put(ACC_Q, i * D + d, Get(ACC_Q, i * D + d) + dqVal);
                Put(ACC_K, i * D + d, Get(ACC_K, i * D + d) + dk1 + dk2);
                dqDotQ += dqVal * Q(b, hk, start + i, d);
                dk1DotK += dk1 * kval;
                dkhDotK += dkState * kval;
                djDotJ += Get(DJ, i * D + d) * bi * ei * kval;
                db += Get(DJ, i * D + d) * ei * kval +
                      Get(PSI, i * D + d) * V(b, hv, start + i, d) +
                      Get(MK, i * D + d) * kval;
                dv_.SetValue(VectorOffset(b, hv, start + i, d, t_->HV),
                             FloatToBf16(bi * Get(PSI, i * D + d)));
            }
            dbeta_.SetValue(ScalarOffset(b, hv, start + i), db);
            Put(DG, i, dqDotQ - dk1DotK + djDotJ);
        }
        Put(DG, m - 1, Get(DG, m - 1) + Exp(gLast) * stateDot + dkhDotK);
        // NW_ij = beta_i * M_ij * (K_i dot K_j); dg2 += rowSum(NW)-colSum(NW).
        for (int64_t i = 0; i < m; ++i) {
            auto ki32 = scratch_;
            auto kj32 = scratch_[D];
            auto product = scratch_[2 * D];
            auto sum = scratch_[3 * D];
            auto reduceWork = scratch_[4 * D];
            AscendC::Cast(ki32, input_[INPUT_K + i * D], AscendC::RoundMode::CAST_NONE, D);
            AscendC::PipeBarrier<PIPE_V>();
            for (int64_t j = 0; j < i; ++j) {
                AscendC::Cast(kj32, input_[INPUT_K + j * D], AscendC::RoundMode::CAST_NONE, D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(product, ki32, kj32, D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ReduceSum(sum, product, reduceWork, static_cast<int32_t>(D));
                AscendC::PipeBarrier<PIPE_ALL>();
                const float dot = sum.GetValue(0);
                const float contribution = Beta(b, hv, start + i) *
                                           Get(MM, i * BT + j) * dot;
                Put(DG, i, Get(DG, i) + contribution);
                Put(DG, j, Get(DG, j) - contribution);
            }
        }
        float suffix = 0.0f;
        for (int64_t i = m - 1; i >= 0; --i) {
            suffix += Get(DG, i);
            dg_.SetValue(ScalarOffset(b, hv, start + i), suffix);
        }
    }

    const ChunkGatedDeltaRuleBwdFinalizeTilingData *t_ = nullptr;
    int64_t wsBase_ = 0;
    int64_t physicalAiv_ = 0;
    int64_t start_ = 0;
    AscendC::GlobalTensor<DT> q_, k_, v_, x_, do_, p_, h_, dh_, a_;
    AscendC::GlobalTensor<DT> dq_, dk_, dv_;
    AscendC::GlobalTensor<float> g_, beta_, dbeta_, dg_, qRstd_, kRstd_, w_, raw_;
    AscendC::GlobalTensor<int64_t> cu_, indices_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_;
    AscendC::LocalTensor<float> ub_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::LocalTensor<DT> input_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratchBuf_;
    AscendC::LocalTensor<float> scratch_;
};

} // namespace GDN
#endif
