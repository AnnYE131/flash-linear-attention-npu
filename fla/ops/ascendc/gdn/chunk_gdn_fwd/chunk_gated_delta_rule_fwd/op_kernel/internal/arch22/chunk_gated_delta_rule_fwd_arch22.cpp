/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gated_delta_rule_fwd_arch22_struct.h"
#include "pipeline/fwd_ho.h"
#if __has_include("../../../../chunk_gated_delta_rule_fwd_prepare/op_kernel/arch22/chunk_gated_delta_rule_fwd_prepare.h")
#include "../../../../chunk_gated_delta_rule_fwd_prepare/op_kernel/arch22/chunk_gated_delta_rule_fwd_prepare.h"
#else
#include "../../../chunk_gated_delta_rule_fwd_prepare/arch22/chunk_gated_delta_rule_fwd_prepare.h"
#endif

namespace GDN {
namespace {

constexpr uint32_t PHASE6_TILING_ALIGNMENT = 8;

// 主路径的静态字段只在对应阶段使用；0/-1 保留旧 key 的运行时契约。
template <int kStaticV = 0, int kStaticBT = 0, int kQkvLayout = -1, int kOLayout = -1>
struct Phase6Config {
    __aicore__ static inline uint64_t ChunkSize(const Arch22ChunkGatedDeltaRuleFwdAbcTiling &abc)
    {
        if constexpr (kStaticBT != 0) {
            return kStaticBT;
        } else {
            return abc.BT;
        }
    }

    __aicore__ static inline uint64_t ChunkAlign(const Arch22ChunkGatedDeltaRuleFwdAbcTiling &abc)
    {
        if constexpr (kStaticBT != 0) {
            return kStaticBT;
        } else {
            return abc.btAlign;
        }
    }

    __aicore__ static inline int64_t ValueDim(const GdnMegaArch22RecomputeWUTilingData &tiling)
    {
        if constexpr (kStaticV != 0) {
            return static_cast<int64_t>(kStaticV);
        } else {
            return tiling.V;
        }
    }

    __aicore__ static inline bool InputSequenceMajor(const Arch22ChunkGatedDeltaRuleFwdAbcTiling &abc)
    {
        if constexpr (kQkvLayout >= 0) {
            return kQkvLayout == 1;
        } else {
            return abc.qkvLayout == 1;
        }
    }

    __aicore__ static inline bool OutputSequenceMajor(const Arch22ChunkGatedDeltaRuleFwdAbcTiling &abc)
    {
        if constexpr (kOLayout >= 0) {
            return kOLayout == 1;
        } else {
            return abc.oLayout == 1;
        }
    }
};

__aicore__ inline uint64_t AlignPhase6(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

__aicore__ inline const __gm__ ChunkRecomputeWUFwdHOTrailer *GetPhase5Trailer(GM_ADDR tiling)
{
    const uint64_t oTilingOffset = AlignPhase6(
        sizeof(GdnMegaArch22FwdHTilingData), PHASE6_TILING_ALIGNMENT);
    return reinterpret_cast<const __gm__ ChunkRecomputeWUFwdHOTrailer *>(
        tiling + oTilingOffset + sizeof(GdnMegaArch22FwdOTilingData));
}

__aicore__ inline const __gm__ Arch22ChunkGatedDeltaRuleFwdTrailer *GetPhase6Trailer(GM_ADDR tiling)
{
    const uint64_t oTilingOffset = AlignPhase6(
        sizeof(GdnMegaArch22FwdHTilingData), PHASE6_TILING_ALIGNMENT);
    const uint64_t phase5End = oTilingOffset + sizeof(GdnMegaArch22FwdOTilingData) +
                               sizeof(ChunkRecomputeWUFwdHOTrailer);
    return reinterpret_cast<const __gm__ Arch22ChunkGatedDeltaRuleFwdTrailer *>(
        tiling + AlignPhase6(phase5End, PHASE6_TILING_ALIGNMENT));
}

__aicore__ inline void CopyAbcTiling(
    const __gm__ Arch22ChunkGatedDeltaRuleFwdAbcTiling *src, Arch22ChunkGatedDeltaRuleFwdAbcTiling &dst)
{
    dst.qkvLayout = src->qkvLayout;
    dst.oLayout = src->oLayout;
    dst.B = src->B;
    dst.Hk = src->Hk;
    dst.Hv = src->Hv;
    dst.hvPerHk = src->hvPerHk;
    dst.T = src->T;
    dst.K = src->K;
    dst.BT = src->BT;
    dst.NT = src->NT;
    dst.taskNum = src->taskNum;
    dst.usedAicNum = src->usedAicNum;
    dst.usedAivNum = src->usedAivNum;
    dst.btAlign = src->btAlign;
    dst.isVarlen = src->isVarlen;
    dst.scoreWorkspaceBytes = src->scoreWorkspaceBytes;
    dst.aWorkspaceBytes = src->aWorkspaceBytes;
    dst.solveWorkspacePerCoreBytes = src->solveWorkspacePerCoreBytes;
    dst.totalTiles = src->totalTiles;
    dst.matrixSize = src->matrixSize;
    dst.numHeads = src->numHeads;
    dst.seqLen = src->seqLen;
    dst.batchSize = src->batchSize;
    dst.isLower = src->isLower;
    dst.hasCuSeqlens = src->hasCuSeqlens;
    dst.tilesPerCore = src->tilesPerCore;
    dst.chunkSize = src->chunkSize;
    dst.numChunks = src->numChunks;
    dst.lastChunkValidSize = src->lastChunkValidSize;
    dst.totalChunks = src->totalChunks;
    dst.layoutMode = src->layoutMode;
    dst.dtypeMode = src->dtypeMode;
    dst.totalTokens = src->totalTokens;

    static_assert(sizeof(TCubeTiling) % sizeof(uint32_t) == 0,
                  "TCubeTiling must be copied as complete 32-bit words");
    const __gm__ uint32_t *cubeSrc =
        reinterpret_cast<const __gm__ uint32_t *>(&src->cubeTilingData);
    uint32_t *cubeDst = reinterpret_cast<uint32_t *>(&dst.cubeTilingData);
    constexpr uint32_t cubeWordCount = sizeof(TCubeTiling) / sizeof(uint32_t);
    for (uint32_t index = 0; index < cubeWordCount; ++index) {
        cubeDst[index] = cubeSrc[index];
    }
}

template <typename InputT, typename TileShapes, bool kPreparedCumsum = false,
          int kStaticV = 0, int kStaticBT = 0, int kQkvLayout = -1, int kOLayout = -1,
          uint32_t kSplitPhase = 1>
__aicore__ inline void RunPhase6(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR rawG, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
    GM_ADDR finalState, GM_ADDR gCumsumBth, GM_ADDR A, GM_ADDR sharedInput, GM_ADDR sharedHo,
    GM_ADDR sharedFront, GM_ADDR workspace, GM_ADDR tiling)
{
    static_assert(kSplitPhase >= 1 && kSplitPhase <= 3, "Invalid private Prepare/H/O phase");
    using Config = Phase6Config<kStaticV, kStaticBT, kQkvLayout, kOLayout>;
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    const __gm__ ChunkRecomputeWUFwdHOTrailer *phase5 = GetPhase5Trailer(tiling);
    const __gm__ Arch22ChunkGatedDeltaRuleFwdTrailer *phase6 = GetPhase6Trailer(tiling);
    Arch22ChunkGatedDeltaRuleFwdAbcTiling abc{};
    CopyAbcTiling(&phase6->abc, abc);
    GdnMegaArch22RecomputeWUTilingData recomputeTiling{};
    CopyRecomputeTiling(&phase5->recompute, recomputeTiling);
    if constexpr (kSplitPhase == 1) {
        // 整个 Prepare 的资源和持久工作循环由独立 arch22 核心拥有。
        GdnUnifiedPrepare::Args args{
            k, v, beta, rawG, cuSeqlens, chunkIndices,
            sharedFront + phase6->sharedGOffset, gCumsumBth,
            kPreparedCumsum ? sharedFront + phase6->sharedDecayOffset : nullptr,
            kPreparedCumsum ? sharedFront + phase6->sharedLastExpOffset : nullptr,
            A, sharedFront + phase6->sharedWOffset, sharedFront + phase6->sharedUOffset,
            userWorkspace + phase6->unifiedPrepareWorkspaceOffset,
            abc.B, abc.Hk, abc.Hv, abc.T, abc.NT, abc.isVarlen, phase6->outputGCumsum};
        const uint64_t bt = Config::ChunkSize(abc);
        const uint64_t valueDim = Config::ValueDim(recomputeTiling);
        const bool sequenceMajor = Config::InputSequenceMajor(abc);
        if (bt == 64) {
            if (valueDim == 128) {
                if (sequenceMajor) GdnUnifiedPrepare::Run<InputT, kPreparedCumsum, 128, 64, 1>(args);
                else GdnUnifiedPrepare::Run<InputT, kPreparedCumsum, 128, 64, 0>(args);
            } else {
                if (sequenceMajor) GdnUnifiedPrepare::Run<InputT, kPreparedCumsum, 256, 64, 1>(args);
                else GdnUnifiedPrepare::Run<InputT, kPreparedCumsum, 256, 64, 0>(args);
            }
        } else {
            if (valueDim == 128) {
                if (sequenceMajor) GdnUnifiedPrepare::Run<InputT, kPreparedCumsum, 128, 128, 1>(args);
                else GdnUnifiedPrepare::Run<InputT, kPreparedCumsum, 128, 128, 0>(args);
            } else {
                if (sequenceMajor) GdnUnifiedPrepare::Run<InputT, kPreparedCumsum, 256, 128, 1>(args);
                else GdnUnifiedPrepare::Run<InputT, kPreparedCumsum, 256, 128, 0>(args);
            }
        }
    } else {
        GdnRawFront::Resource resource;
        GM_ADDR gCumsumBht = sharedInput + phase6->sharedGOffset;
        GM_ADDR gLastExp = kPreparedCumsum ? sharedInput + phase6->sharedLastExpOffset : nullptr;
        GM_ADDR gDecay = kPreparedCumsum ? sharedInput + phase6->sharedDecayOffset : nullptr;
        GM_ADDR w = sharedInput + phase6->sharedWOffset;
        GM_ADDR u = sharedInput + phase6->sharedUOffset;
        GM_ADDR hoBase = kSplitPhase == 2 ? A : sharedHo;
        GM_ADDR h = hoBase;
        GM_ADDR vNew = hoBase + phase6->sharedVNewOffset;
        if constexpr (kSplitPhase == 2) {
            DispatchFwdH<InputT, TileShapes, kStaticV, kStaticBT, kQkvLayout>(
                resource, k, w, u, gCumsumBht, gk, initialState, cuSeqlens,
                chunkIndices, h, vNew, finalState, tiling, userWorkspace,
                nullptr, nullptr, Config::InputSequenceMajor(abc), gLastExp, gDecay);
        } else if constexpr (kSplitPhase == 3) {
            const uint64_t oTilingOffset =
                AlignPhase6(sizeof(GdnMegaArch22FwdHTilingData), PHASE6_TILING_ALIGNMENT);
            const auto *gmOTiling = reinterpret_cast<const __gm__ GdnMegaArch22FwdOTilingData *>(
                tiling + oTilingOffset);
            GdnMegaArch22FwdOTilingData oTiling{};
            CopyOTiling(gmOTiling, oTiling);
            DispatchFwdO<InputT, kStaticV, kStaticBT, kQkvLayout, kOLayout>(
                resource, q, k, vNew, h, gCumsumBht, cuSeqlens, chunkIndices, o,
                userWorkspace, &oTiling, nullptr, nullptr,
                Config::InputSequenceMajor(abc), Config::OutputSequenceMajor(abc));
        }
    }
}

} // namespace
} // namespace GDN

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR a_storage, GM_ADDR raw_g,
    GM_ADDR gk, GM_ADDR initial_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR shared_ho,
    GM_ADDR o, GM_ADDR final_state, GM_ADDR g_cumsum_bth, GM_ADDR A, GM_ADDR shared_front,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GDN::Arch22ChunkGatedDeltaRuleFwdTrailer);
    if (TILING_KEY_IS(5)) {
        KERNEL_TASK_TYPE(5, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       false, 0, 0, -1, -1, 1>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(6)) {
        KERNEL_TASK_TYPE(6, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       false, 0, 0, -1, -1, 1>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(7)) {
        KERNEL_TASK_TYPE(7, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       true, 0, 0, -1, -1, 1>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(8)) {
        KERNEL_TASK_TYPE(8, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       true, 0, 0, -1, -1, 1>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(9)) {
        KERNEL_TASK_TYPE(9, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       false, 0, 0, -1, -1, 2>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(10)) {
        KERNEL_TASK_TYPE(10, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       false, 0, 0, -1, -1, 2>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(11)) {
        KERNEL_TASK_TYPE(11, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       true, 0, 0, -1, -1, 2>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(12)) {
        KERNEL_TASK_TYPE(12, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       true, 0, 0, -1, -1, 2>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(13)) {
        KERNEL_TASK_TYPE(13, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       false, 0, 0, -1, -1, 3>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(14)) {
        KERNEL_TASK_TYPE(14, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       false, 0, 0, -1, -1, 3>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(15)) {
        KERNEL_TASK_TYPE(15, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       true, 0, 0, -1, -1, 3>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(16)) {
        KERNEL_TASK_TYPE(16, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       true, 0, 0, -1, -1, 3>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(108)) {
        KERNEL_TASK_TYPE(108, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       true, 128, 64, 0, 1, 1>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(109)) {
        KERNEL_TASK_TYPE(109, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       true, 128, 64, 1, 1, 1>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(110)) {
        KERNEL_TASK_TYPE(110, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       true, 128, 128, 0, 1, 1>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(111)) {
        KERNEL_TASK_TYPE(111, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       true, 128, 128, 1, 1, 1>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(112)) {
        KERNEL_TASK_TYPE(112, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       true, 256, 64, 0, 1, 1>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(113)) {
        KERNEL_TASK_TYPE(113, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       true, 256, 64, 1, 1, 1>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(114)) {
        KERNEL_TASK_TYPE(114, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       true, 256, 128, 0, 1, 1>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(115)) {
        KERNEL_TASK_TYPE(115, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       true, 256, 128, 1, 1, 1>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(116)) {
        KERNEL_TASK_TYPE(116, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       true, 128, 64, 0, 1, 2>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(117)) {
        KERNEL_TASK_TYPE(117, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       true, 128, 64, 1, 1, 2>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(118)) {
        KERNEL_TASK_TYPE(118, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       true, 128, 128, 0, 1, 2>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(119)) {
        KERNEL_TASK_TYPE(119, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       true, 128, 128, 1, 1, 2>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(120)) {
        KERNEL_TASK_TYPE(120, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       true, 256, 64, 0, 1, 2>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(121)) {
        KERNEL_TASK_TYPE(121, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       true, 256, 64, 1, 1, 2>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(122)) {
        KERNEL_TASK_TYPE(122, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       true, 256, 128, 0, 1, 2>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(123)) {
        KERNEL_TASK_TYPE(123, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       true, 256, 128, 1, 1, 2>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(124)) {
        KERNEL_TASK_TYPE(124, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       true, 128, 64, 0, 1, 3>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(125)) {
        KERNEL_TASK_TYPE(125, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       true, 128, 64, 1, 1, 3>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(126)) {
        KERNEL_TASK_TYPE(126, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       true, 128, 128, 0, 1, 3>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(127)) {
        KERNEL_TASK_TYPE(127, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
                       true, 128, 128, 1, 1, 3>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(128)) {
        KERNEL_TASK_TYPE(128, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       true, 256, 64, 0, 1, 3>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(129)) {
        KERNEL_TASK_TYPE(129, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       true, 256, 64, 1, 1, 3>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(130)) {
        KERNEL_TASK_TYPE(130, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       true, 256, 128, 0, 1, 3>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    } else if (TILING_KEY_IS(131)) {
        KERNEL_TASK_TYPE(131, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes256,
                       true, 256, 128, 1, 1, 3>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, a_storage, shared_ho, shared_front, workspace, tiling);
    }
}
