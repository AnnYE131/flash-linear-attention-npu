/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */

#ifndef GDN_HO_PIPELINE_CONTEXT_H
#define GDN_HO_PIPELINE_CONTEXT_H

#include <cstdint>

namespace GDN {

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
