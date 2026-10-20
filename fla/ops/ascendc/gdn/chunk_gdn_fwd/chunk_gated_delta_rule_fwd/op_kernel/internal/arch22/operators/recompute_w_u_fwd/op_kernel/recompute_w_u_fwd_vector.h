/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_w_u_fwd.h
 * \brief
 */


#ifndef RECOMPUTE_W_U_FWD_VECTOR_H
#define RECOMPUTE_W_U_FWD_VECTOR_H

#include "../../../qkv_input_layout.h"
#include "recompute_w_u_fwd_struct.h"
#include "catlass/arch/cross_core_sync.hpp"
using namespace AscendC;

using GDN::GdnMegaArch22RecomputeWUTilingData;

template <typename kType, typename betaType, bool kFlattenHeadTasks = false,
          bool kAbcTaskOrder = false>
class RecomputeWUFwdVectorProcess {
public:
    bool inputSequenceMajor{false};
    __aicore__ inline void ConfigureInputLayout(bool sequenceMajor) { inputSequenceMajor = sequenceMajor; }

    /** @brief constructor */
    __aicore__ inline RecomputeWUFwdVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_, GM_ADDR cu_seqlens_,
                                                        GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
                                                        GM_ADDR workspace_);

    __aicore__ inline void Process(const GDN::RecomputeTaskRange *range = nullptr);
    __aicore__ inline void ProcessVbAndKbgExpInterleaved();
    __aicore__ inline void Init(const GdnMegaArch22RecomputeWUTilingData &tiling, AscendC::TPipe *pipe_);

private:
    uint64_t B = 0;
    uint64_t T = 0;
    uint64_t Hv = 1;
    uint64_t Hk = 1;
    uint64_t hvPerHk = 1;
    uint64_t K = 0;
    uint64_t V = 0;
    uint64_t chunkSize = 0;
    uint64_t chunkNum = 0;
    uint64_t interleavedMembaseVecRow = 0;

    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR g;
    GM_ADDR cu_seqlens;
    GM_ADDR chunk_indices;
    GM_ADDR w;
    GM_ADDR u;
    GM_ADDR workspace;
    AscendC::TPipe *pipe = nullptr;
    const GDN::RecomputeTaskRange *taskRange = nullptr;

private:
    GlobalTensor<kType> kTensor;
    GlobalTensor<kType> vTensor;
    GlobalTensor<betaType> betaTensor;
    GlobalTensor<betaType> gTensor;
    GlobalTensor<kType> workSpaceTensor;

    // Vb/KbgExp 同循环交错执行，共享按 max(K,V) 尺寸分配的队列。
    TQue<AscendC::TPosition::VECIN, 1> dataInQue;
    TQue<AscendC::TPosition::VECIN, 1> betaInQue;
    TQue<AscendC::TPosition::VECIN, 1> gInQue;
    TQue<AscendC::TPosition::VECOUT, 1> dataOutQue;

    // FP32 中间 buffer（membase 路径需要）
    TBuf<AscendC::TPosition::VECCALC> dataFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32BrcbBuf;
    TBuf<AscendC::TPosition::VECCALC> gFp32Buf;
};

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kAbcTaskOrder>
__aicore__ inline RecomputeWUFwdVectorProcess<kType, betaType,
                                               kFlattenHeadTasks, kAbcTaskOrder>::RecomputeWUFwdVectorProcess(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_,
    GM_ADDR cu_seqlens_, GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
    GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), g(g_), cu_seqlens(cu_seqlens_),
      chunk_indices(chunk_indices_), w(w_), u(u_), workspace(workspace_){};

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kAbcTaskOrder>
__aicore__ void inline RecomputeWUFwdVectorProcess<kType, betaType, kFlattenHeadTasks,
                                                   kAbcTaskOrder>::Init(
    const GdnMegaArch22RecomputeWUTilingData &tiling, AscendC::TPipe *pipe_)
{
    pipe = pipe_;
    workSpaceTensor.SetGlobalBuffer((__gm__ kType *)workspace);
    kTensor.SetGlobalBuffer((__gm__ kType *)k);
    vTensor.SetGlobalBuffer((__gm__ kType *)v);
    betaTensor.SetGlobalBuffer((__gm__ betaType *)beta);
    gTensor.SetGlobalBuffer((__gm__ betaType *)g);

    B = tiling.B;
    T = tiling.T;
    Hv = tiling.Hv;
    Hk = tiling.Hk;
    hvPerHk = tiling.hvPerHk;
    K = tiling.K;
    V = tiling.V;
    chunkSize = tiling.chunkSize;
    chunkNum = tiling.chunkNum;
    interleavedMembaseVecRow = tiling.interleavedMembaseVecRow;
    return;
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kAbcTaskOrder>
__aicore__ void inline RecomputeWUFwdVectorProcess<kType, betaType, kFlattenHeadTasks,
                                                   kAbcTaskOrder>::Process(const GDN::RecomputeTaskRange *range)
{
    // 统一交错协议：taskRange 空与非空都走批内每 chunk+head 的 Vb→KbgExp
    // 交错 + 窗口批量同步；批间正确性由外层 FRONT_ACK 握手兜底。
    taskRange = range;
    ProcessVbAndKbgExpInterleaved();
    return;
}


// =================== ProcessVbAndKbgExpInterleaved ===================
// 交错模式（taskRange 空与非空统一）：每个 chunk+head 内先做 Vb 再做 KbgExp，
// 使 Cube 端能复用 L1 中的 A 矩阵；每核任务序列按 4 个一批同步，
// 末批不满时由 isLast 在最后一个任务上冲刷。
template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kAbcTaskOrder>
__aicore__ void inline RecomputeWUFwdVectorProcess<kType, betaType, kFlattenHeadTasks,
                                                   kAbcTaskOrder>::ProcessVbAndKbgExpInterleaved()
{
    uint32_t coreLoops = kFlattenHeadTasks ? chunkNum * Hv : chunkNum;
    uint32_t subBlockIdx = GetSubBlockIdx();
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = interleavedMembaseVecRow;
    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    uint32_t loopBegin = coreIdx;
    uint32_t loopEnd = coreLoops;
    uint32_t loopStep = coreNumAic;
    if constexpr (kAbcTaskOrder) {
        // Phase 6 ABC assigns a contiguous task range to each AIC. Keep the
        // recompute consumer on the same core so A is locally produced first.
        const uint32_t tasksPerCore = (coreLoops + coreNumAic - 1) / coreNumAic;
        loopBegin = coreIdx * tasksPerCore;
        loopEnd = (loopBegin + tasksPerCore) < coreLoops ? loopBegin + tasksPerCore : coreLoops;
        loopStep = 1;
    }
    if (taskRange != nullptr) {
        loopBegin = static_cast<uint32_t>(taskRange->begin);
        loopEnd = static_cast<uint32_t>(taskRange->end);
        loopStep = 1;
    }
    // Vb and KbgExp execute sequentially within the same loop, so reuse data
    // queues sized for the larger K/V dimension.
    uint32_t maxDataDim = K > V ? K : V;
    pipe->InitBuffer(dataInQue, 2, rowNum * maxDataDim * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(gInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(dataOutQue, 2, rowNum * maxDataDim * sizeof(kType));
    pipe->InitBuffer(dataFp32Buf, rowNum * maxDataDim * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);
    pipe->InitBuffer(gFp32Buf, rowNum * sizeof(float32_t));

    auto tensorDataFp32 = dataFp32Buf.Get<float32_t>();
    auto tensorBetaFP32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbFP32 = betaFp32BrcbBuf.Get<float32_t>();
    auto tensorGFP32 = gFp32Buf.Get<float32_t>();

    for (uint32_t loopIdx = loopBegin; loopIdx < loopEnd; loopIdx += loopStep) {
        uint32_t chunkIdx = 0;
        uint32_t hBegin = 0;
        uint32_t hEnd = 0;
        DecodeRecomputeTask<kFlattenHeadTasks, kAbcTaskOrder>(
            loopIdx, cu_seqlens, Hv, T, chunkSize, chunkNum, chunkIdx, hBegin, hEnd);
        GetChunkOffset(cu_seqlens, chunk_indices, B, Hv, T, chunkSize, chunkIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (uint32_t h = hBegin; h < hEnd; ++h) {
            bool isLast = (loopIdx + loopStep >= loopEnd) && (h + 1 >= hEnd);
            ++vecTaskIdx;
            if ((vecTaskIdx & 0b01) != subBlockIdx) {
                if ((vecTaskIdx & 0b11) == 0 || isLast) {
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
                    if (vecTaskIdx > 4) {
                        CrossCoreWaitFlag<0x2>(SYNC_AIC_AIV_FLAG_5);
                    }
                }
                continue;
            }
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto vOffset = (h * T + bos + rowOffset) * V;
                auto betaOffset = h * T + bos + rowOffset;
                // copyin
                {
                    auto tensorVin = dataInQue.AllocTensor<kType>();
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    if (inputSequenceMajor) {
                        const uint64_t inputOffset = GDN::QkvSequenceMajorOffset(vOffset, T, Hv, V);
                        const uint64_t rowGap = static_cast<uint64_t>(Hv - 1) * V * sizeof(kType);
                        if (rowGap <= 0xffffffffULL) {
                            DataCopyPad(tensorVin, vTensor[inputOffset],
                                {static_cast<uint16_t>(curRowNum), static_cast<uint32_t>(V * sizeof(kType)),
                                 static_cast<uint32_t>(rowGap), 0, 0}, {false, 0, 0, 0});
                        } else {
                            // The DMA gap field is uint32 bytes; large strides remain correct.
                            for (uint32_t row = 0; row < curRowNum; ++row) {
                                DataCopy(tensorVin[row * V],
                                    vTensor[inputOffset + static_cast<uint64_t>(row) * Hv * V], V);
                            }
                        }
                    } else {
                        DataCopy(tensorVin, vTensor[vOffset], V * curRowNum);
                    }
                    DataCopyPad(tensorBetain, betaTensor[betaOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                                {false, 0, 0, 0});
                    dataInQue.EnQue(tensorVin);
                    betaInQue.EnQue(tensorBetain);
                }
                // compute
                {
                    auto tensorVin = dataInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorVbOut = dataOutQue.AllocTensor<kType>();
                    // cast FP32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, curRowNum);
                    } else {
                        DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                    }
                    Cast(tensorDataFp32, tensorVin, RoundMode::CAST_NONE, V * curRowNum);
                    PipeBarrier<PIPE_V>();
                    // brcb
                    Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                    PipeBarrier<PIPE_V>();
                    // mul
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = V * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < V) {
                        Mul(tensorDataFp32[perchannelResOffset], tensorDataFp32[perchannelResOffset],
                            tensorBetaBrcbFP32, FP32_PER_REPEAT_64, curRowNum,
                            {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    // cast
                    Cast(tensorVbOut, tensorDataFp32, RoundMode::CAST_RINT, V * curRowNum);
                    dataInQue.FreeTensor(tensorVin);
                    betaInQue.FreeTensor(tensorBetain);
                    dataOutQue.EnQue(tensorVbOut);
                }
                // copyout
                {
                    auto tensorVbOut = dataOutQue.DeQue<kType>();
                    const uint64_t dstOffset = taskRange == nullptr ? vOffset :
                        ((loopIdx - taskRange->begin) * chunkSize + rowOffset) * V;
                    DataCopy(workSpaceTensor[dstOffset], tensorVbOut, V * curRowNum);
                    dataOutQue.FreeTensor(tensorVbOut);
                }
            }

            uint64_t hk = h / hvPerHk;
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                // 注意：定长场景下 GetChunkOffset 返回的 bos 已含按 Hv 计的批次偏移 bIdx*Hv*T
                // （见 recompute_w_u_fwd_common.h GetChunkOffset 的 bos += bIdx*H*T，此处 H 传入的是 Hv）。
                // k 只有 Hk 个 head，需把批次偏移换算成 bIdx*Hk*T，即 bos - bIdx*(Hv-Hk)*T。
                // 此换算强耦合于 GetChunkOffset 的批次偏移实现，若后者修改需同步更新此处。
                // coreLoopsInB 必须与 GetChunkOffset 内保持一致的算法。
                uint64_t coreLoopsInB = (T + chunkSize - 1) / chunkSize;
                uint64_t bIdx = cu_seqlens ? 0 : (chunkIdx / coreLoopsInB);
                uint64_t bosK = cu_seqlens ? bos : (bos - bIdx * (Hv - Hk) * T);
                auto kSrcOffset = (hk * T + bosK + rowOffset) * K;
                auto kDstOffset = (h * T + bos + rowOffset) * K;
                auto betaOffset = h * T + bos + rowOffset;
                // copyin
                {
                    auto tensorKin = dataInQue.AllocTensor<kType>();
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    auto tensorGin = gInQue.AllocTensor<betaType>();
                    if (inputSequenceMajor) {
                        const uint64_t inputOffset = GDN::QkvSequenceMajorOffset(kSrcOffset, T, Hk, K);
                        const uint64_t rowGap = static_cast<uint64_t>(Hk - 1) * K * sizeof(kType);
                        if (rowGap <= 0xffffffffULL) {
                            DataCopyPad(tensorKin, kTensor[inputOffset],
                                {static_cast<uint16_t>(curRowNum), static_cast<uint32_t>(K * sizeof(kType)),
                                 static_cast<uint32_t>(rowGap), 0, 0}, {false, 0, 0, 0});
                        } else {
                            // The DMA gap field is uint32 bytes; large strides remain correct.
                            for (uint32_t row = 0; row < curRowNum; ++row) {
                                DataCopy(tensorKin[row * K],
                                    kTensor[inputOffset + static_cast<uint64_t>(row) * Hk * K], K);
                            }
                        }
                    } else {
                        DataCopy(tensorKin, kTensor[kSrcOffset], K * curRowNum);
                    }
                    DataCopyPad(tensorBetain, betaTensor[betaOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                                {false, 0, 0, 0});
                    DataCopyPad(tensorGin, gTensor[betaOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                                {false, 0, 0, 0});
                    dataInQue.EnQue(tensorKin);
                    betaInQue.EnQue(tensorBetain);
                    gInQue.EnQue(tensorGin);
                }
                // compute
                {
                    auto tensorKin = dataInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorGin = gInQue.DeQue<betaType>();
                    auto tensorOut = dataOutQue.AllocTensor<kType>();
                    // cast FP32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, curRowNum);
                        Cast(tensorGFP32, tensorGin, RoundMode::CAST_NONE, curRowNum);
                    } else {
                        DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                        DataCopy(tensorGFP32, tensorGin, rowNum);
                    }
                    Cast(tensorDataFp32, tensorKin, RoundMode::CAST_NONE, K * curRowNum);
                    PipeBarrier<PIPE_V>();
                    Exp(tensorGFP32, tensorGFP32, curRowNum);
                    PipeBarrier<PIPE_V>();
                    Mul(tensorBetaFP32, tensorBetaFP32, tensorGFP32, curRowNum);
                    PipeBarrier<PIPE_V>();
                    // brcb
                    Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                    PipeBarrier<PIPE_V>();
                    // mul
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < K) {
                        Mul(tensorDataFp32[perchannelResOffset], tensorDataFp32[perchannelResOffset],
                            tensorBetaBrcbFP32, FP32_PER_REPEAT_64, curRowNum,
                            {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    // cast
                    Cast(tensorOut, tensorDataFp32, RoundMode::CAST_RINT, K * curRowNum);
                    dataInQue.FreeTensor(tensorKin);
                    betaInQue.FreeTensor(tensorBetain);
                    gInQue.FreeTensor(tensorGin);
                    dataOutQue.EnQue(tensorOut);
                }
                // copyout
                {
                    auto tensorOut = dataOutQue.DeQue<kType>();
                    const uint64_t dstOffset = taskRange == nullptr ? kDstOffset :
                        taskRange->capacity * chunkSize * V +
                        ((loopIdx - taskRange->begin) * chunkSize + rowOffset) * K;
                    DataCopy(workSpaceTensor[dstOffset], tensorOut, K * curRowNum);
                    dataOutQue.FreeTensor(tensorOut);
                }
            }
            // KbgExp 完成通知 AIC 可以做 matmul。
            // 统一批量同步：窗口末任务（或 isLast 尾任务）批量通知并等待
            // 上一窗口槽位释放；两个子核在各自分支都执行，保持 mode2 信用一致。
            if ((vecTaskIdx & 0b11) == 0 || isLast) {
                CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
                if (vecTaskIdx > 4) {
                    CrossCoreWaitFlag<0x2>(SYNC_AIC_AIV_FLAG_5);
                }
            }
        }
    }
}

#endif // RECOMPUTE_W_U_FWD_VECTOR_H
