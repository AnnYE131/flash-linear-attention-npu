/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_BWD_DQKWG_HK_PARTITION_H
#define CHUNK_BWD_DQKWG_HK_PARTITION_H

#include <cstdint>
#include "../../op_kernel/chunk_bwd_dqkwg_schedule.h"

// 纯 host 整数评分；不使用相加形式的 ceil，也不先形成可能溢出的乘积。
inline bool DqkwgHkPartitionPeak(uint64_t coreLoops, uint64_t HV, uint64_t coreNum,
                                 uint64_t partitions, uint64_t &peak)
{
    if (coreLoops == 0 || coreNum == 0 || partitions == 0 || HV == 0 || HV % partitions != 0 ||
        coreLoops > static_cast<uint64_t>(INT64_MAX) / partitions) {
        return false;
    }
    const uint64_t tasks = coreLoops * partitions;
    const uint64_t waves = tasks / coreNum + static_cast<uint64_t>(tasks % coreNum != 0);
    const uint64_t heads = HV / partitions;
    if (waves > UINT64_MAX / heads) {
        return false;
    }
    peak = waves * heads;
    return true;
}

// allowBalance 由调用方允许合法varlen或dense整chunk；非950等资格仍由调用方约束。
inline uint64_t DqkwgSelectHkPartitions(uint64_t B, uint64_t numChunks, uint64_t HV, uint64_t HK,
                                      uint64_t coreNum, uint64_t parentPartitions, bool allowBalance)
{
    if (!allowBalance || !DqkwgUseU32Schedule(B, numChunks, HV, HK, coreNum, parentPartitions)) {
        return parentPartitions;
    }
    // 父资格已界定 B*numChunks 在 U32 内；此处才形成 N。
    const uint64_t coreLoops = B * numChunks;
    if (coreLoops < coreNum) {
        return parentPartitions;
    }
    uint64_t parentPeak = 0;
    if (!DqkwgHkPartitionPeak(coreLoops, HV, coreNum, parentPartitions, parentPeak)) {
        return parentPartitions;
    }
    uint64_t bestPeak = parentPeak;
    uint64_t bestPartitions = parentPartitions;
    auto consider = [&](uint64_t partitions) {
        if ((HK / partitions) % 2 != 0) {
            return;
        }
        uint64_t peak = 0;
        if (DqkwgHkPartitionPeak(coreLoops, HV, coreNum, partitions, peak) &&
            (peak < bestPeak || (peak == bestPeak && partitions < bestPartitions))) {
            bestPeak = peak;
            bestPartitions = partitions;
        }
    };
    // 整数约数配对；不用 divisor*divisor 或 O(HK) 全扫。
    for (uint64_t divisor = 1; divisor <= HK / divisor; ++divisor) {
        if (HK % divisor != 0) {
            continue;
        }
        consider(divisor);
        const uint64_t other = HK / divisor;
        if (other != divisor) {
            consider(other);
        }
    }
    return bestPeak < parentPeak ? bestPartitions : parentPartitions;
}

#endif // CHUNK_BWD_DQKWG_HK_PARTITION_H
