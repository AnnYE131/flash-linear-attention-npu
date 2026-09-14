/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CUMSUM_PREPARE_HPP
#define GDN_CUMSUM_PREPARE_HPP

#include <cstdint>
#include "kernel_operator.h"

namespace GdnCumsumPrepare {

using namespace AscendC;

// Stage P is intentionally limited to the first A2 contract.  The caller keeps
// the original route for all other head counts/chunk sizes/architectures.
constexpr uint64_t kHeads = 8;
constexpr uint64_t kChunk64 = 64;
constexpr uint64_t kChunk128 = 128;
constexpr uint32_t kUbAlignmentBytes = 32;
constexpr uint32_t kFp32BlockElements = kUbAlignmentBytes / sizeof(float);

struct PrepareArgs {
    GM_ADDR rawG;          // [B, T, H], FP32
    GM_ADDR gCumsumBht;    // [B, H, T], FP32 workspace
    GM_ADDR gCumsumBth;    // [B, T, H], FP32 optional public output
    GM_ADDR cuSeqlens;     // varlen only, [sequence_count + 1], int64
    GM_ADDR chunkIndices;  // varlen only, [num_chunks, 2], int64
    uint64_t batch;
    uint64_t heads;
    uint64_t tokens;
    uint64_t chunkSize;
    uint64_t numChunks;
    uint64_t taskNum;
    uint64_t isVarlen;
    uint64_t outputG;
};

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline bool IsSupported(const PrepareArgs &args)
{
    return args.rawG != nullptr && args.gCumsumBht != nullptr && args.batch > 0 &&
           args.heads == kHeads && args.tokens > 0 &&
           (args.chunkSize == kChunk64 || args.chunkSize == kChunk128) &&
           args.numChunks > 0 && args.taskNum > 0 &&
           (args.isVarlen == 0 || (args.cuSeqlens != nullptr && args.chunkIndices != nullptr));
}

class Kernel {
public:
    __aicore__ inline void Init(const PrepareArgs &args)
    {
        args_ = args;
        rawGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.rawG),
                               args_.batch * args_.tokens * args_.heads);
        gCumsumBhtGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.gCumsumBht),
                                      args_.batch * args_.heads * args_.tokens);
        hasPublicOutput_ = args_.outputG != 0 && args_.gCumsumBth != nullptr;
        if (hasPublicOutput_) {
            gCumsumBthGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.gCumsumBth),
                                          args_.batch * args_.tokens * args_.heads);
        }
        if (args_.isVarlen != 0) {
            cuSeqlensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(args_.cuSeqlens));
            chunkIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(args_.chunkIndices),
                                            args_.numChunks * 2);
        }

        const uint32_t tileBytes = static_cast<uint32_t>(args_.chunkSize * args_.heads * sizeof(float));
        pipe_.InitBuffer(inputBthBuf_, tileBytes);
        pipe_.InitBuffer(prefixBthBuf_, tileBytes);
        pipe_.InitBuffer(prefixBhtBuf_, tileBytes);
        pipe_.InitBuffer(offsetBuf_, tileBytes);
        pipe_.InitBuffer(accBuf_, kFp32BlockElements * sizeof(float));

        // Keep each physical dependency role explicit.  Stage P uses one
        // buffer, so reuse waits are required before the next task overwrites
        // any input or output area.
        mte2ToV_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        vToMte2_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        vToMte3_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        mte3ToV_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        mte3ToMte2_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        scalarToV_ = GetTPipePtr()->AllocEventID<HardEvent::S_V>();
    }

    __aicore__ inline void Process()
    {
        if (args_.numChunks == 0 || args_.taskNum == 0) {
            ReleaseEvents();
            return;
        }
        BuildOffsets();

        const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t activeAivCount = static_cast<uint64_t>(GetBlockNum());
        if (activeAivCount == 0 || coreIdx >= activeAivCount || coreIdx >= args_.taskNum) {
            ReleaseEvents();
            return;
        }

        bool hasPreviousTask = false;
        for (uint64_t task = coreIdx; task < args_.taskNum; task += activeAivCount) {
            if (ProcessTask(task, hasPreviousTask)) {
                hasPreviousTask = true;
            }
        }

        // Close all producer edges before returning.  The following kernel in
        // the same stream consumes gCumsumBht only after this boundary.
        if (hasPreviousTask) {
            WaitFlag<HardEvent::V_MTE2>(vToMte2_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToV_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        }
        ReleaseEvents();
    }

private:
    __aicore__ inline void BuildOffsets()
    {
        LocalTensor<uint32_t> offsets = offsetBuf_.Get<uint32_t>();
        // Gather writes the destination in head-major order.  The source is
        // BTH, so dst[head * BT + row] <- src[row * H + head].
        for (uint64_t head = 0; head < args_.heads; ++head) {
            for (uint64_t row = 0; row < args_.chunkSize; ++row) {
                offsets.SetValue(
                    static_cast<uint32_t>(head * args_.chunkSize + row),
                    static_cast<uint32_t>((row * args_.heads + head) * sizeof(float)));
            }
        }
        SetFlag<HardEvent::S_V>(scalarToV_);
        WaitFlag<HardEvent::S_V>(scalarToV_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DecodeTask(uint64_t task, uint64_t &batch, uint64_t &rowStart,
                                      uint64_t &valid) const
    {
        const uint64_t chunk = task % args_.numChunks;
        batch = task / args_.numChunks;
        rowStart = chunk * args_.chunkSize;
        valid = rowStart < args_.tokens ? MinU64(args_.chunkSize, args_.tokens - rowStart) : 0;
        if (args_.isVarlen != 0) {
            // The tiling contract stores canonical (sequence, localChunk)
            // pairs in sequence-major order.  B=1 for this route.
            const int64_t sequence = chunkIndicesGm_.GetValue(chunk * 2);
            const int64_t localChunk = chunkIndicesGm_.GetValue(chunk * 2 + 1);
            const int64_t bos = cuSeqlensGm_.GetValue(sequence);
            const int64_t eos = cuSeqlensGm_.GetValue(sequence + 1);
            const int64_t start = bos + localChunk * static_cast<int64_t>(args_.chunkSize);
            const int64_t remaining = eos - start;
            rowStart = start > 0 ? static_cast<uint64_t>(start) : 0;
            valid = remaining > 0 ? MinU64(args_.chunkSize, static_cast<uint64_t>(remaining)) : 0;
            batch = 0;
        }
    }

    __aicore__ inline bool ProcessTask(uint64_t task, bool hasPreviousTask)
    {
        uint64_t batch = 0;
        uint64_t rowStart = 0;
        uint64_t valid = 0;
        DecodeTask(task, batch, rowStart, valid);
        if (valid == 0 || batch >= args_.batch || rowStart >= args_.tokens) {
            return false;
        }

        // All reuse waits happen before the first instruction that can
        // overwrite an old task's input/prefix area.
        if (hasPreviousTask) {
            WaitFlag<HardEvent::V_MTE2>(vToMte2_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToV_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        }

        LocalTensor<float> input = inputBthBuf_.Get<float>();
        LocalTensor<float> prefixBth = prefixBthBuf_.Get<float>();
        LocalTensor<float> prefixBht = prefixBhtBuf_.Get<float>();
        LocalTensor<float> acc = accBuf_.Get<float>();

        const uint32_t tileElements = static_cast<uint32_t>(args_.chunkSize * args_.heads);
        // Gather reads the entire tile, including the invalid tail of the last
        // chunk.  Initialize that source tail after the previous MTE3 writes
        // have drained; only valid rows are ever exposed to GM.
        Duplicate(prefixBth, 0.0f, tileElements);
        PipeBarrier<PIPE_V>();

        const uint64_t inputOffset = (batch * args_.tokens + rowStart) * args_.heads;
        const DataCopyExtParams inputParams{
            1, static_cast<uint32_t>(valid * args_.heads * sizeof(float)), 0, 0, 0};
        const DataCopyPadExtParams<float> inputPad{false, 0, 0, 0.0f};
        DataCopyPad(input, rawGm_[inputOffset], inputParams, inputPad);
        SetFlag<HardEvent::MTE2_V>(mte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV_);

        // Each Add operates on the eight independent heads.  The outer loop
        // remains strictly increasing in T, with the first element preserving
        // the legacy Adds(+0.0f) operation.
        for (uint64_t row = 0; row < valid; ++row) {
            LocalTensor<float> inputRow = input[row * args_.heads];
            if (row == 0) {
                Adds(acc, inputRow, 0.0f, static_cast<uint32_t>(args_.heads));
            } else {
                Add(acc, acc, inputRow, static_cast<uint32_t>(args_.heads));
            }
            PipeBarrier<PIPE_V>();
            // The first Copy argument is the element mask.  Mask=1 would
            // copy only head 0; mask=8 preserves all independent heads.
            Copy(prefixBth[row * args_.heads], acc,
                 static_cast<uint64_t>(args_.heads), 1, {1, 1, kFp32BlockElements, kFp32BlockElements});
            // Copy reads acc asynchronously; close its WAR edge before the
            // next Add overwrites the accumulator.
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(vToMte2_);

        const uint32_t elementCount = static_cast<uint32_t>(args_.chunkSize * args_.heads);
        Gather(prefixBht, prefixBth, offsetBuf_.Get<uint32_t>(), 0, elementCount);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);

        const DataCopyExtParams headParams{
            1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0};
        for (uint64_t head = 0; head < args_.heads; ++head) {
            const uint64_t bhtOffset = (batch * args_.heads + head) * args_.tokens + rowStart;
            DataCopyPad(gCumsumBhtGm_[bhtOffset], prefixBht[head * args_.chunkSize], headParams);
        }
        if (hasPublicOutput_) {
            const uint64_t bthOffset = (batch * args_.tokens + rowStart) * args_.heads;
            const DataCopyExtParams publicParams{
                1, static_cast<uint32_t>(valid * args_.heads * sizeof(float)), 0, 0, 0};
            DataCopyPad(gCumsumBthGm_[bthOffset], prefixBth, publicParams);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToV_);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        return true;
    }

    __aicore__ inline void ReleaseEvents()
    {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(mte2ToV_);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(vToMte2_);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToV_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_V>(scalarToV_);
    }

    PrepareArgs args_{};
    TPipe pipe_;
    TBuf<TPosition::VECCALC> inputBthBuf_;
    TBuf<TPosition::VECCALC> prefixBthBuf_;
    TBuf<TPosition::VECCALC> prefixBhtBuf_;
    TBuf<TPosition::VECCALC> offsetBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    GlobalTensor<float> rawGm_;
    GlobalTensor<float> gCumsumBhtGm_;
    GlobalTensor<float> gCumsumBthGm_;
    GlobalTensor<int64_t> cuSeqlensGm_;
    GlobalTensor<int64_t> chunkIndicesGm_;
    TEventID mte2ToV_;
    TEventID vToMte2_;
    TEventID vToMte3_;
    TEventID mte3ToV_;
    TEventID mte3ToMte2_;
    TEventID scalarToV_;
    bool hasPublicOutput_ = false;
};

__aicore__ inline void Run(const PrepareArgs &args)
{
    if (!IsSupported(args)) {
        return;
    }
    Kernel kernel;
    kernel.Init(args);
    kernel.Process();
}

} // namespace GdnCumsumPrepare

#endif // GDN_CUMSUM_PREPARE_HPP
