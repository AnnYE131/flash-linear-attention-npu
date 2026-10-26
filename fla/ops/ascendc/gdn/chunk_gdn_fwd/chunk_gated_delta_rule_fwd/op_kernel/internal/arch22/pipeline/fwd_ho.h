/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include "../stages/fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
#include "../stages/fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
#undef CATLASS_ARCH
#include "../stages/fwd_o/op_kernel/chunk_fwd_o_struct.h"
#include "../stages/fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp"
#include "chunk_gated_delta_rule_ho_pipeline.h"
#include "../common/gdn_ho_resources.h"
#include "wu_ho/chunk_recompute_wu_fwd_ho_struct.h"
#include "kernel_operator.h"


namespace GDN {
namespace {

template <typename InputT, typename GT, typename StateT, typename TileShapes, bool kGated,
          int kStaticV = 0, int kStaticBT = 0, int kQkvLayout = -1, bool kUseExp2 = false>
__aicore__ inline void RunFwdH(GdnRawFront::Resource &resource, GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                               GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                               GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
                               GM_ADDR userWorkspace,
                               const GdnHoPipeline::HoPipelineConfig *idleConfig = nullptr,
                               GM_ADDR hoReadyAddr = nullptr, bool inputSequenceMajor = false,
                               GM_ADDR gLastExp = nullptr, GM_ADDR gDecay = nullptr)
{
    // Keep the same H implementation mode as the established FwdHO kernel.
    // The final boolean enables the H/O fused scheduling path; using the
    // standalone-H mode here changes synchronization and precision behavior.
    using Kernel = Catlass::Gemm::Kernel::GDNFwdHKernel<
        InputT, GT, StateT, float, TileShapes, kGated, true, kUseExp2, true, kStaticV, kStaticBT, kQkvLayout>;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    // The private H implementation retains its original AtlasA2 resource type.
    Catlass::Arch::Resource<Catlass::Arch::AtlasA2> hResource;
    Kernel kernel(hResource);
#else
    Kernel kernel(resource);
#endif
    kernel.Init(k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace);
    if (idleConfig != nullptr) {
        // 统一入口在 Init 后、Process 前配置；关闭态在 kernel 内部直接返回。
        kernel.ConfigureIdlePipeline(*idleConfig, hoReadyAddr);
    }
    kernel.ConfigureInputLayout(inputSequenceMajor);
    if (gLastExp != nullptr) {
        kernel.ConfigureGLastExp(gLastExp);
    }
    kernel.ConfigureGDecay(gDecay);
    kernel.Process();
}

template <typename InputT, typename TileShapes,
          int kStaticV = 0, int kStaticBT = 0, int kQkvLayout = -1, bool kUseExp2 = false>
__aicore__ inline void DispatchFwdHMode(GdnRawFront::Resource &resource, GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                                    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
                                    GM_ADDR userWorkspace,
                                    const GdnHoPipeline::HoPipelineConfig *idleConfig = nullptr,
                                    GM_ADDR hoReadyAddr = nullptr, bool inputSequenceMajor = false,
                                    GM_ADDR gLastExp = nullptr, GM_ADDR gDecay = nullptr)
{
    const __gm__ GdnMegaArch22FwdHTilingData *hTiling =
        reinterpret_cast<const __gm__ GdnMegaArch22FwdHTilingData *>(tiling);
    // Mega's input dtype is fixed by the generated DTYPE_Q variant, and its
    // cumsum/gk contract is FP32. State remains runtime-selected: a disabled
    // final-state output is an FP32 placeholder, not the initial-state dtype.
    if (hTiling->stateDataType == 2) {
        if (hTiling->useGk) {
            RunFwdH<InputT, float, float, TileShapes, true, kStaticV, kStaticBT, kQkvLayout, kUseExp2>(
                resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace, idleConfig, hoReadyAddr, inputSequenceMajor, gLastExp, gDecay);
        } else {
            RunFwdH<InputT, float, float, TileShapes, false, kStaticV, kStaticBT, kQkvLayout, kUseExp2>(
                resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, userWorkspace, idleConfig, hoReadyAddr, inputSequenceMajor, gLastExp, gDecay);
        }
    } else if (hTiling->useGk) {
        RunFwdH<InputT, float, InputT, TileShapes, true, kStaticV, kStaticBT, kQkvLayout, kUseExp2>(
            resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
            tiling, userWorkspace, idleConfig, hoReadyAddr, inputSequenceMajor, gLastExp, gDecay);
    } else {
        RunFwdH<InputT, float, InputT, TileShapes, false, kStaticV, kStaticBT, kQkvLayout, kUseExp2>(
            resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
            tiling, userWorkspace, idleConfig, hoReadyAddr, inputSequenceMajor, gLastExp, gDecay);
    }
}

template <typename InputT, typename TileShapes,
          int kStaticV = 0, int kStaticBT = 0, int kQkvLayout = -1>
__aicore__ inline void DispatchFwdH(GdnRawFront::Resource &resource, GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                                    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
                                    GM_ADDR userWorkspace,
                                    const GdnHoPipeline::HoPipelineConfig *idleConfig = nullptr,
                                    GM_ADDR hoReadyAddr = nullptr, bool inputSequenceMajor = false,
                                    GM_ADDR gLastExp = nullptr, GM_ADDR gDecay = nullptr, bool useExp2 = false)
{
    if (useExp2) {
        DispatchFwdHMode<InputT, TileShapes, kStaticV, kStaticBT, kQkvLayout, true>(
            resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
            tiling, userWorkspace, idleConfig, hoReadyAddr, inputSequenceMajor, gLastExp, gDecay);
    } else {
        DispatchFwdHMode<InputT, TileShapes, kStaticV, kStaticBT, kQkvLayout, false>(
            resource, k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
            tiling, userWorkspace, idleConfig, hoReadyAddr, inputSequenceMajor, gLastExp, gDecay);
    }
}

__aicore__ inline void CopyOTiling(const __gm__ GdnMegaArch22FwdOTilingData *src, GdnMegaArch22FwdOTilingData &dst)
{
    dst.shapeBatch = src->shapeBatch;
    dst.seqlen = src->seqlen;
    dst.kNumHead = src->kNumHead;
    dst.vNumHead = src->vNumHead;
    dst.kHeadDim = src->kHeadDim;
    dst.vHeadDim = src->vHeadDim;
    dst.chunkSize = src->chunkSize;
    dst.isVariedLen = src->isVariedLen;
    dst.tokenBatch = src->tokenBatch;
    dst.dataType = src->dataType;
    dst.gDataType = src->gDataType;
    dst.vWorkspaceOffset = src->vWorkspaceOffset;
    dst.hWorkspaceOffset = src->hWorkspaceOffset;
    dst.attnWorkspaceOffset = src->attnWorkspaceOffset;
    dst.aftermaskWorkspaceOffset = src->aftermaskWorkspaceOffset;
    dst.maskWorkspaceOffset = src->maskWorkspaceOffset;
    dst.scale = src->scale;
}

__aicore__ inline void CopyRecomputeTiling(const __gm__ GdnMegaArch22RecomputeWUTilingData *src,
                                            GdnMegaArch22RecomputeWUTilingData &dst)
{
    // Tiling data is serialized in GM; the recompute process consumes a local-memory copy.
    dst.B = src->B;
    dst.Hk = src->Hk;
    dst.Hv = src->Hv;
    dst.hvPerHk = src->hvPerHk;
    dst.T = src->T;
    dst.K = src->K;
    dst.V = src->V;
    dst.chunkNum = src->chunkNum;
    dst.chunkSize = src->chunkSize;
    dst.vbVecRow = src->vbVecRow;
    dst.kbgExpVecRow = src->kbgExpVecRow;
    dst.interleavedMembaseVecRow = src->interleavedMembaseVecRow;
    dst.isVariable = src->isVariable;
}

template <typename InputT, typename GT, int kStaticV = 0, int kStaticBT = 0,
          int kQkvLayout = -1, int kOLayout = -1, bool kUseExp2 = false>
__aicore__ inline void RunFwdO(GdnRawFront::Resource &resource, GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR h, GM_ADDR g,
                               GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
                               GM_ADDR userWorkspace, const GdnMegaArch22FwdOTilingData *tiling,
                               const GdnHoPipeline::HoPipelineConfig *idleConfig = nullptr,
                               GM_ADDR hoReadyAddr = nullptr, bool inputSequenceMajor = false, bool outputSequenceMajor = false)
{
    using Kernel = Catlass::Gemm::Kernel::GDNFwdOKernel<InputT, GT, float, true, kStaticV, kStaticBT, kQkvLayout, kOLayout, kUseExp2>;
    Kernel kernel(resource);
    kernel.Init(q, k, vNew, h, g, cuSeqlens, chunkIndices, o, tiling, userWorkspace);
    if (idleConfig != nullptr) {
        kernel.ConfigureIdlePipeline(*idleConfig, hoReadyAddr);
    }
    kernel.ConfigureInputLayout(inputSequenceMajor);
    kernel.ConfigureOutputLayout(outputSequenceMajor);
    kernel.Process();
}

template <typename InputT, int kStaticV = 0, int kStaticBT = 0,
          int kQkvLayout = -1, int kOLayout = -1>
__aicore__ inline void DispatchFwdO(GdnRawFront::Resource &resource, GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR h, GM_ADDR g,
                                    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
                                    GM_ADDR userWorkspace, const GdnMegaArch22FwdOTilingData *tiling,
                                    const GdnHoPipeline::HoPipelineConfig *idleConfig = nullptr,
                                    GM_ADDR hoReadyAddr = nullptr, bool inputSequenceMajor = false, bool outputSequenceMajor = false, bool useExp2 = false)
{
    if (useExp2) {
        RunFwdO<InputT, float, kStaticV, kStaticBT, kQkvLayout, kOLayout, true>(
            resource, q, k, vNew, h, g, cuSeqlens, chunkIndices, o, userWorkspace, tiling,
            idleConfig, hoReadyAddr, inputSequenceMajor, outputSequenceMajor);
    } else {
        RunFwdO<InputT, float, kStaticV, kStaticBT, kQkvLayout, kOLayout, false>(
            resource, q, k, vNew, h, g, cuSeqlens, chunkIndices, o, userWorkspace, tiling,
            idleConfig, hoReadyAddr, inputSequenceMajor, outputSequenceMajor);
    }
}

} // namespace
} // namespace GDN
