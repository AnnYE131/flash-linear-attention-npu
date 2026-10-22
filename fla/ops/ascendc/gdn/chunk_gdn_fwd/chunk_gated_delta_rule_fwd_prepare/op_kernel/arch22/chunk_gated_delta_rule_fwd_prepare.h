// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. A2 GDN Prepare：单资源、持久 work 循环、双版本 GM 的统一入口。
#ifndef GDN_ARCH22_UNIFIED_PREPARE_H
#define GDN_ARCH22_UNIFIED_PREPARE_H

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 220
#error "Unified GDN Prepare is a DAV_2201 implementation"
#endif

#include "chunk_gated_delta_rule_fwd_prepare_cube.h"
#include "chunk_gated_delta_rule_fwd_prepare_vector.h"

namespace GdnUnifiedPrepare {

template <typename InputT, bool Prepared, int V, int BT, int QkvLayout>
__aicore__ inline void Run(const Args &args)
{
    static_assert(sizeof(InputT) == 2, "GDN preserves its FP16/BF16 intermediate rounding");
    static_assert(V == 128 || V == 256);
    static_assert(BT == 64 || BT == 128);
    static_assert(QkvLayout == 0 || QkvLayout == 1);
    constexpr uint32_t Window = kHeadsPerWindow;
    const uint64_t g = args.hv / args.hk;
    const auto p = MakeWorkspaceLayout(BT, V, g);
    const uint64_t total = WorkItems(args.batch, args.nt, args.hk, g, BT);
    const uint64_t group = GroupId();
    if (group >= total) return; // 空物理组不建立资源、不 seed、不等待集合。
    Resources owner;            // 整个 Prepare 仅此一个资源拥有者，跨 task/wave 均不重建。
    if ASCEND_IS_AIC {
        SetHF32Mode(false);
        Cube<InputT, V, BT, QkvLayout> cube(owner, args, p);
        // 整个持久 work 循环仅 seed 一次。0/1 未使用，留给既有其它 kernel。
        CrossCoreSetFlag<0x2, PIPE_MTE2>(6);
        CrossCoreSetFlag<0x2, PIPE_MTE2>(7);
        uint64_t wave = 0;
        for (uint64_t work = group; work < total; work += GetBlockNum()) {
            const Chunk c = Decode<BT>(args, work, p.hkPerTask);
            if (c.valid == 0) continue;
            cube.Scores(c); // HK raw KKT 每 task 一次；全部内部窗口共享它。
            for (uint64_t begin = c.headBegin; begin < c.headEnd;) {
                const uint32_t windows = static_cast<uint32_t>(Min(2, CeilDiv(c.headEnd - begin, Window)));
                // 必须先发布本组全部 raw-ready，AIV 才能完成两窗 KKT 并进入 Assemble16。
                for (uint32_t i = 0; i < windows; ++i) {
                    const uint32_t slot = (wave + i) & 1;
                    CrossCoreSetFlag<0x2, PIPE_FIX>(4 + slot);
                }
                for (uint32_t i = 0; i < windows; ++i) {
                    const uint64_t head = begin + i * Window;
                    const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                    const uint32_t slot = (wave + i) & 1;
                    CrossCoreWaitFlag<0x2, PIPE_MTE2>(2 + slot);
                    owner.template Fence<HardEvent::MTE2_S>();
                    cube.template Merge<16>(c, slot, active);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(4 + slot);
                }
                for (uint32_t i = 0; i < windows; ++i) {
                    const uint64_t head = begin + i * Window;
                    const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                    const uint32_t slot = (wave + i) & 1;
                    CrossCoreWaitFlag<0x2, PIPE_MTE2>(2 + slot);
                    owner.template Fence<HardEvent::MTE2_S>();
                    cube.template Merge<32>(c, slot, active);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(4 + slot);
                }
                if constexpr (BT == 128) {
                    for (uint32_t i = 0; i < windows; ++i) {
                        const uint64_t head = begin + i * Window;
                        const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                        const uint32_t slot = (wave + i) & 1;
                        CrossCoreWaitFlag<0x2, PIPE_MTE2>(2 + slot);
                        owner.template Fence<HardEvent::MTE2_S>();
                        cube.template Merge<64>(c, slot, active);
                        CrossCoreSetFlag<0x2, PIPE_FIX>(4 + slot);
                    }
                }
                for (uint32_t i = 0; i < windows; ++i) {
                    const uint64_t head = begin + i * Window;
                    const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                    const uint32_t slot = (wave + i) & 1;
                    CrossCoreWaitFlag<0x2, PIPE_MTE2>(2 + slot);
                    owner.template Fence<HardEvent::MTE2_S>();
                    cube.WU(c, head, active, slot);
                }
                begin += windows * Window;
                wave += windows;
            }
        }
    }
    if ASCEND_IS_AIV {
        Vector<InputT, Prepared, V, BT, QkvLayout> vector(owner, args, p);
        uint64_t wave = 0;
        for (uint64_t work = group; work < total; work += GetBlockNum()) {
            const Chunk c = Decode<BT>(args, work, p.hkPerTask);
            if (c.valid == 0) continue;
            for (uint64_t begin = c.headBegin; begin < c.headEnd;) {
                const uint32_t windows = static_cast<uint32_t>(Min(2, CeilDiv(c.headEnd - begin, Window)));
                for (uint32_t i = 0; i < windows; ++i) {
                    const uint64_t head = begin + i * Window;
                    const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                    const uint32_t slot = (wave + i) & 1;
                    // 空 AIV 仍参加全部集合；wave 只对本组实际窗口递增。
                    CrossCoreWaitFlag<0x2, PIPE_MTE2>(6 + slot);
                    vector.Prefix(c, head, active, slot);
                    CrossCoreWaitFlag<0x2, PIPE_MTE2>(4 + slot);
                    owner.template Fence<HardEvent::MTE2_S>();
                    vector.KktAndLeaves(c, head, active, slot);
                    vector.Publish(slot);
                }
                for (uint32_t i = 0; i < windows; ++i) {
                    const uint64_t head = begin + i * Window;
                    const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                    const uint32_t slot = (wave + i) & 1;
                    CrossCoreWaitFlag<0x2, PIPE_MTE2>(4 + slot);
                    owner.template Fence<HardEvent::MTE2_S>();
                    vector.template Assemble<16>(c, head, active, slot);
                    vector.Publish(slot);
                }
                if constexpr (BT == 128) {
                    for (uint32_t i = 0; i < windows; ++i) {
                        const uint64_t head = begin + i * Window;
                        const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                        const uint32_t slot = (wave + i) & 1;
                        CrossCoreWaitFlag<0x2, PIPE_MTE2>(4 + slot);
                        owner.template Fence<HardEvent::MTE2_S>();
                        vector.template Assemble<32>(c, head, active, slot);
                        vector.Publish(slot);
                    }
                }
                // 本 AIV 的两槽 Assemble16 都已完成，才可用整个 leaf 区做 Final/RHS。
                for (uint32_t i = 0; i < windows; ++i) {
                    const uint64_t head = begin + i * Window;
                    const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                    const uint32_t slot = (wave + i) & 1;
                    CrossCoreWaitFlag<0x2, PIPE_MTE2>(4 + slot);
                    owner.template Fence<HardEvent::MTE2_S>();
                    vector.template Assemble<BT / 2>(c, head, active, slot);
                    vector.Rhs(c, head, active, slot);
                    vector.Publish(slot);
                }
                begin += windows * Window;
                wave += windows;
            }
        }
        // 奇数 wave/未使用版本也消费最后的 free，seed 与每个发布均闭合。
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(6);
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(7);
    }
}
} // namespace GdnUnifiedPrepare
#endif
