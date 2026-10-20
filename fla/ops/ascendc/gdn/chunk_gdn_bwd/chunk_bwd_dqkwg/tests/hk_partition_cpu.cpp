/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include <cstdint>
#include <iostream>
#include "../op_host/op_tiling/chunk_bwd_dqkwg_hk_partition.h"

int main()
{
    struct SelectionCase {
        const char *name;
        uint64_t B, chunks, HV, HK, cores, parent;
        bool allow;
        uint64_t expected;
    };
    const SelectionCase cases[] = {
        {"uniform", 1, 176, 128, 16, 24, 1, true, 8},
        {"batch_product", 2, 88, 128, 16, 24, 1, true, 8},
        {"same_peak_smallest_p", 1, 176, 256, 32, 24, 1, true, 8},
        {"no_strict_improvement", 1, 192, 128, 16, 24, 1, true, 1},
        {"nondividing_gcd", 1, 176, 80, 10, 24, 1, true, 5},
        {"nondividing_coprime", 1, 176, 80, 10, 23, 1, true, 5},
        {"odd_owner_excludes_better_peak", 1, 25, 16, 2, 24, 1, true, 1},
        {"odd_HK", 1, 25, 72, 9, 24, 1, true, 1},
        {"varlen_disabled", 1, 3, 80, 10, 24, 8, false, 8},
        {"tail_disabled", 1, 176, 128, 16, 24, 1, false, 1},
        {"underfilled_parent", 1, 5, 128, 16, 24, 4, true, 4},
        {"parent_u32_rejected", UINT32_MAX, 2, 128, 16, 24, 1, true, 1},
        {"u32_score_boundary", 1, UINT32_MAX, UINT32_MAX - 1ULL, UINT32_MAX - 1ULL, 1, 1, true, 1},
        {"varlen_full_chunks", 1, 256, 128, 16, 20, 1, true, 8},
        {"varlen_tail_chunks", 1, 4, 8, 4, 3, 1, true, 2},
    };
    uint32_t checks = 0;
    for (const auto &test : cases) {
        const uint64_t actual = DqkwgSelectHkPartitions(test.B, test.chunks, test.HV, test.HK,
                                                       test.cores, test.parent, test.allow);
        if (actual != test.expected) {
            std::cerr << test.name << ": actual=" << actual << " expected=" << test.expected << '\n';
            return 1;
        }
        std::cout << test.name << ": P=" << actual << '\n';
        ++checks;
    }
    struct PeakCase {
        const char *name;
        uint64_t loops, HV, cores, partitions;
        bool accepted;
        uint64_t expected;
    };
    const PeakCase peaks[] = {
        {"peak_uniform", 176, 128, 24, 8, true, 944},
        {"int64_product_boundary", static_cast<uint64_t>(INT64_MAX) / 2, 4, 2, 2,
         true, static_cast<uint64_t>(INT64_MAX) - 1},
        {"int64_product_rejected", static_cast<uint64_t>(INT64_MAX) / 2 + 1, 4, 2, 2, false, 0},
        {"u64_score_rejected", static_cast<uint64_t>(INT64_MAX), 3, 1, 1, false, 0},
        {"ceil_without_addition", static_cast<uint64_t>(INT64_MAX), 1, UINT64_MAX, 1, true, 1},
    };
    for (const auto &test : peaks) {
        uint64_t actual = 0;
        const bool accepted = DqkwgHkPartitionPeak(test.loops, test.HV, test.cores, test.partitions, actual);
        if (accepted != test.accepted || (accepted && actual != test.expected)) {
            std::cerr << test.name << ": accepted=" << accepted << " peak=" << actual << '\n';
            return 1;
        }
        std::cout << test.name << ": accepted=" << accepted << " peak=" << actual << '\n';
        ++checks;
    }
    std::cout << "PASS checks=" << checks << '\n';
    return 0;
}
