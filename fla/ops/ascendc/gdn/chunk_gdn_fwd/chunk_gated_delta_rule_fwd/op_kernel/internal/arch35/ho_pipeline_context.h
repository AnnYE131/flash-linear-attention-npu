/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */

#ifndef GDN_HO_PIPELINE_CONTEXT_H
#define GDN_HO_PIPELINE_CONTEXT_H

#include <cstdint>

namespace GDN {

constexpr int64_t HO_PIPELINE_DTYPE_BF16 = 1;
constexpr int64_t HO_PIPELINE_K_HEAD_DIM = 128;
constexpr int64_t HO_PIPELINE_V_HEAD_DIM = 128;
constexpr int64_t HO_PIPELINE_CHUNK_SIZE = 64;

// Keep host reservation and device enablement on one scalar predicate.  The
// header has no framework or serialized-tiling dependency, so it is safe to
// include from both sides of the arch35 private implementation.
constexpr bool HoPipelineLayoutEligible(bool isAscend950, int64_t dataType,
                                         int64_t kHeadDim, int64_t vHeadDim,
                                         int64_t chunkSize)
{
    return isAscend950 && dataType == HO_PIPELINE_DTYPE_BF16 &&
           kHeadDim == HO_PIPELINE_K_HEAD_DIM && vHeadDim == HO_PIPELINE_V_HEAD_DIM &&
           chunkSize == HO_PIPELINE_CHUNK_SIZE;
}

// Private Phase-6 context shared by the arch35 fused H/O producer and
// consumer.  It is deliberately independent of serialized tiling data: host
// tiling reserves the eligible layout, while the device fills enabled and
// group counts from the same invocation metadata before entering H/O.
struct HoPipelineContext {
    bool enabled{false};
    uint32_t producerGroups{0};
    uint32_t consumerGroups{0};
    uint64_t physicalShapeBatch{0};
    uint64_t chunksPerPhysicalBatch{0};
    uint64_t readyTaskCount{0};
};

}  // namespace GDN

#endif  // GDN_HO_PIPELINE_CONTEXT_H
