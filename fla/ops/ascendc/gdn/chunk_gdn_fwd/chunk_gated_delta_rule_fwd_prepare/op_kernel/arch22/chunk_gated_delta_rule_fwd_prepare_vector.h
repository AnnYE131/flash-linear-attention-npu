// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. prefix/beta 常驻、条带 KKT 到 UB 叶块、固定 16 叶运算与 RHS。
#ifndef GDN_ARCH22_UNIFIED_PREPARE_VECTOR_H
#define GDN_ARCH22_UNIFIED_PREPARE_VECTOR_H

#include "chunk_gated_delta_rule_fwd_prepare_resources.h"

namespace GdnUnifiedPrepare {
template <typename InputT, bool Prepared, int V, int BT, int QkvLayout>
class Vector {
    static constexpr uint32_t Half = kHeadsPerWindow / 2;
    static constexpr uint32_t Leaves = BT / 16;
    static constexpr uint32_t L = Half * Leaves;
    static constexpr uint32_t E = L * 256;
    static constexpr uint32_t InputBth = 4096;
    static constexpr uint32_t PrefixBth = 8192;
    static constexpr uint32_t HeadIndices = 12288;
    static constexpr uint32_t TokenIndices = 14336;
    static constexpr uint32_t Acc = 18432;
    static constexpr uint32_t Factor = 20480;
    Resources &r_;
    const Args &a_;
    WorkspaceLayout p_;
    GM_ADDR core_;
    uint32_t sub_;

    __aicore__ inline GM_ADDR Head(uint32_t slot, uint32_t localHead) const
    { return core_ + p_.slotsOffset + slot * p_.slotBytes + localHead * p_.headBytes; }

    __aicore__ inline uint32_t Heads(uint32_t active) const
    { return active > sub_ * Half ? static_cast<uint32_t>(Min(Half, active - sub_ * Half)) : 0; }

    __aicore__ inline void LoadHeadMajor(LocalTensor<float> dst, GM_ADDR src,
                                        const Chunk &c, uint64_t begin, uint32_t count)
    {
        auto gm = Gm<float>(src, a_.batch * a_.hv * a_.tokens);
        const uint32_t rounded = (static_cast<uint32_t>(c.valid) + 7) / 8 * 8;
        DataCopyExtParams cp{static_cast<uint16_t>(count), static_cast<uint32_t>(c.valid * 4),
            static_cast<uint32_t>((a_.tokens - c.valid) * 4), (BT - rounded) / 8, 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(rounded - c.valid), 0};
        DataCopyPad(dst, gm[HeadOffset(a_, c, begin)], cp, pad);
    }

    __aicore__ inline void StoreHeadMajor(GM_ADDR dst, LocalTensor<float> src,
                                         const Chunk &c, uint64_t begin, uint32_t count)
    {
        auto gm = Gm<float>(dst, a_.batch * a_.hv * a_.tokens);
        const uint32_t rounded = (static_cast<uint32_t>(c.valid) + 7) / 8 * 8;
        DataCopyExtParams cp{static_cast<uint16_t>(count), static_cast<uint32_t>(c.valid * 4),
            (BT - rounded) / 8, static_cast<uint32_t>((a_.tokens - c.valid) * 4), 0};
        DataCopyPad(gm[HeadOffset(a_, c, begin)], src, cp);
    }

    __aicore__ inline void LoadRect(LocalTensor<float> dst, GlobalTensor<float> src,
                                    uint32_t rows, uint32_t cols, uint32_t srcStride, uint32_t dstStride)
    {
        if (rows == 0 || cols == 0) return;
        const uint32_t rounded = (cols + 7) / 8 * 8;
        DataCopyExtParams cp{static_cast<uint16_t>(rows), cols * 4, (srcStride - cols) * 4,
            (dstStride - rounded) / 8, 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(rounded - cols), 0};
        DataCopyPad(dst, src, cp, pad);
    }

public:
    __aicore__ inline Vector(Resources &r, const Args &a, WorkspaceLayout p)
        : r_(r), a_(a), p_(p), core_(a.scratch + GroupId() * p.coreBytes), sub_(GetSubBlockIdx())
    {
        auto heads = r_.template UB<uint32_t>(HeadIndices);
        auto tokens = r_.template UB<uint32_t>(TokenIndices);
        for (uint32_t h = 0; h < Half; ++h) {
            for (uint32_t t = 0; t < BT; ++t) heads.SetValue(h * BT + t, (t * 8 + h) * 4);
        }
        if constexpr (!Prepared) {
            for (uint32_t t = 0; t < BT; ++t) {
                for (uint32_t h = 0; h < 8; ++h) tokens.SetValue(t * 8 + h, ((h % Half) * BT + t) * 4);
            }
        }
        r_.template Fence<HardEvent::S_V>();
    }

    __aicore__ inline void Prefix(const Chunk &c, uint64_t headBegin, uint32_t activeHeads)
    {
        const uint32_t count = Heads(activeHeads);
        if (count == 0) return;
        const uint64_t begin = headBegin + sub_ * Half;
        auto g = r_.template UB<float>(Local::kG);
        auto beta = r_.template UB<float>(Local::kBeta);
        auto input = r_.template UB<float>(InputBth);
        auto prefix = r_.template UB<float>(PrefixBth);
        auto acc = r_.template UB<float>(Acc);
        // 前一 pack 的 UB 换义同时约束下一位 Vector/MTE2 写者，不能只等 MTE3→V。
        r_.template Fence<HardEvent::MTE3_V>();
        r_.template Fence<HardEvent::MTE3_MTE2>();
        r_.template Fence<HardEvent::V_MTE2>();
        Duplicate(g, 0.0f, Half * BT);
        Duplicate(beta, 0.0f, Half * BT);
        Duplicate(prefix, 0.0f, BT * 8);
        Duplicate(acc, 0.0f, 8);
        PipeBarrier<PIPE_V>();
        r_.template Fence<HardEvent::V_MTE2>();
        LoadHeadMajor(beta, a_.beta, c, begin, count);
        if constexpr (Prepared) {
            auto raw = Gm<float>(a_.rawG, a_.batch * a_.tokens * a_.hv);
            DataCopyExtParams cp{static_cast<uint16_t>(c.valid), count * 4,
                static_cast<uint32_t>((a_.hv - count) * 4), 0, 0};
            DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(8 - count), 0};
            DataCopyPad(input, raw[(c.batch * a_.tokens + c.row) * a_.hv + begin], cp, pad);
            r_.template Fence<HardEvent::MTE2_V>();
        } else {
            LoadHeadMajor(g, a_.rawG, c, begin, count);
            r_.template Fence<HardEvent::MTE2_V>();
            Gather(input, g, r_.template UB<uint32_t>(TokenIndices), 0, BT * 8);
            PipeBarrier<PIPE_V>();
        }
        for (uint32_t row = 0; row < c.valid; ++row) {
            if (row == 0) Adds(acc, input[row * 8], 0.0f, count);
            else Add(acc, acc, input[row * 8], count);
            PipeBarrier<PIPE_V>();
            Copy(prefix[row * 8], acc, count, 1, {1, 1, 8, 8});
            PipeBarrier<PIPE_V>(); // Copy 对 acc 的末次读取先于下一行 Add。
        }
        Gather(g, prefix, r_.template UB<uint32_t>(HeadIndices), 0, Half * BT);
        PipeBarrier<PIPE_V>();
        r_.template Fence<HardEvent::V_MTE3>();
        StoreHeadMajor(a_.gBht, g, c, begin, count);
        if (a_.outputG != 0 && a_.gBth != nullptr) {
            auto publicG = Gm<float>(a_.gBth, a_.batch * a_.tokens * a_.hv);
            DataCopyExtParams cp{static_cast<uint16_t>(c.valid), count * 4, 0,
                static_cast<uint32_t>((a_.hv - count) * 4), 0};
            DataCopyPad(publicG[(c.batch * a_.tokens + c.row) * a_.hv + begin], prefix, cp);
        }
        if constexpr (Prepared) {
            // 保留原有效长度 1 的单位衰减，包括非有限 raw g；其他长度严格 final-prefix。
            Duplicate(input, c.valid == 1 ? 1.0f : 0.0f, BT * 8);
            PipeBarrier<PIPE_V>();
            if (c.valid > 1) {
                for (uint32_t row = 0; row < c.valid; ++row) Sub(input[row * 8], acc, prefix[row * 8], count);
                PipeBarrier<PIPE_V>();
                Exp(input, input, static_cast<uint32_t>(c.valid * 8));
                PipeBarrier<PIPE_V>();
            }
            Exp(acc, acc, count);
            PipeBarrier<PIPE_V>();
            r_.template Fence<HardEvent::V_MTE3>();
            auto last = Gm<float>(a_.gLastExp, a_.batch * a_.nt * ((a_.hv + 7) / 8 * 8));
            DataCopyExtParams cp{1, count * 4, 0, 0, 0};
            DataCopyPad(last[(c.batch * a_.nt + c.index) * ((a_.hv + 7) / 8 * 8) + begin], acc, cp);
            // prefixBth 旧公开输出的读者已退出，才将它换义为 head-major decay。
            r_.template Fence<HardEvent::MTE3_V>();
            Gather(prefix, input, r_.template UB<uint32_t>(HeadIndices), 0, Half * BT);
            PipeBarrier<PIPE_V>();
            r_.template Fence<HardEvent::V_MTE3>();
            StoreHeadMajor(a_.gDecay, prefix, c, begin, count);
        }
    }

    __aicore__ inline void KktAndLeaves(const Chunk &c, uint64_t headBegin, uint32_t activeHeads, uint32_t slot)
    {
        auto leaves = r_.template UB<float>(Local::kLeaf);
        auto score = r_.template UB<float>(Local::kArena);
        auto gate = r_.template UB<float>(Local::kArena + 16 * BT * 4);
        auto out = r_.template UB<float>(Local::kArena + 2 * 16 * BT * 4);
        auto typed = r_.template UB<InputT>(Local::kArena + 3 * 16 * BT * 4);
        auto brcb = r_.template UB<float>(Local::kArena + 3 * 16 * BT * 4 + 16 * BT * sizeof(InputT));
        r_.template Fence<HardEvent::MTE3_V>();
        Duplicate(leaves, 0.0f, E);
        PipeBarrier<PIPE_V>();
        auto scores = Gm<float>(core_, p_.scoreBytes / 4);
        Invalidate(scores);
        const uint32_t count = Heads(activeHeads);
        for (uint32_t h = 0; h < count; ++h) {
            const uint32_t local = sub_ * Half + h;
            const uint64_t hk = (headBegin + local) / (a_.hv / a_.hk) - c.hkBegin;
            auto g = r_.template UB<float>(Local::kG)[h * BT];
            auto beta = r_.template UB<float>(Local::kBeta)[h * BT];
            auto x = Gm<float>(Head(slot, local) + p_.xOffset, BT * BT);
            for (uint32_t rowBase = 0; rowBase < c.valid; rowBase += 16) {
                const uint32_t rows = static_cast<uint32_t>(Min(16, c.valid - rowBase));
                const uint32_t cols = rowBase + rows;
                r_.template Fence<HardEvent::MTE3_V>();
                r_.template Fence<HardEvent::V_MTE2>();
                LoadRect(score, scores[hk * BT * BT + rowBase * BT], rows, c.valid, BT, BT);
                Duplicate(out, 0.0f, 16 * BT);
                PipeBarrier<PIPE_V>();
                const uint8_t rs = BT / 8;
                for (uint32_t col = 0; col < cols; col += 64) {
                    Copy(gate[col], g[col], static_cast<uint16_t>(Min(64, cols - col)),
                        static_cast<uint8_t>(rows), {1, 1, rs, 0});
                }
                PipeBarrier<PIPE_V>();
                Brcb(brcb, g[rowBase], static_cast<uint8_t>((rows + 7) / 8), {1, 8});
                PipeBarrier<PIPE_V>();
                for (uint32_t col = 0; col < cols; col += 64) {
                    Sub(gate[col], brcb, gate[col], Min(64, cols - col), static_cast<uint8_t>(rows),
                        {1, 0, 1, rs, 1, rs});
                }
                PipeBarrier<PIPE_V>();
                for (uint32_t col = 0; col < cols; col += 64)
                    Maxs(gate[col], gate[col], -50.0f, Min(64, cols - col), static_cast<uint8_t>(rows), {1, 1, rs, rs});
                PipeBarrier<PIPE_V>();
                for (uint32_t col = 0; col < cols; col += 64)
                    Mins(gate[col], gate[col], 50.0f, Min(64, cols - col), static_cast<uint8_t>(rows), {1, 1, rs, rs});
                PipeBarrier<PIPE_V>();
                for (uint32_t col = 0; col < cols; col += 64)
                    Exp(gate[col], gate[col], Min(64, cols - col), static_cast<uint8_t>(rows), {1, 1, rs, rs});
                PipeBarrier<PIPE_V>();
                Brcb(brcb, beta[rowBase], static_cast<uint8_t>((rows + 7) / 8), {1, 8});
                PipeBarrier<PIPE_V>();
                for (uint32_t col = 0; col < cols; col += 64) {
                    Mul(gate[col], gate[col], brcb, Min(64, cols - col), static_cast<uint8_t>(rows),
                        {1, 1, 0, rs, rs, 1});
                }
                PipeBarrier<PIPE_V>();
                r_.template Fence<HardEvent::MTE2_V>();
                for (uint32_t row = 0; row < rows; ++row) {
                    if (rowBase + row > 0) Mul(out[row * BT], score[row * BT], gate[row * BT], rowBase + row);
                }
                PipeBarrier<PIPE_V>();
                // GDN 的关键舍入边界：FP32 -> InputT RINT -> FP32，叶块直接消费此 UB 结果。
                Cast(typed, out, RoundMode::CAST_RINT, rows * BT);
                PipeBarrier<PIPE_V>();
                Cast(out, typed, RoundMode::CAST_NONE, rows * BT);
                PipeBarrier<PIPE_V>();
                DataCopy(leaves[(h * Leaves + rowBase / 16) * 256], out[rowBase],
                    {static_cast<uint16_t>(rows), 2, static_cast<uint16_t>((BT - 16) / 8), 0});
                PipeBarrier<PIPE_V>();
                r_.template Fence<HardEvent::V_MTE3>();
                DataCopyExtParams cp{static_cast<uint16_t>(rows), BT * 4, 0, 0, 0};
                DataCopyPad(x[rowBase * BT], out, cp);
            }
        }
        r_.template Fence<HardEvent::MTE3_V>(); // prod 与 epilogue arena 换义。
        auto prod = r_.template UB<float>(Local::kArena);
        auto row = r_.template UB<float>(Local::kLeafRow);
        Muls(leaves, leaves, -1.0f, E);
        PipeBarrier<PIPE_V>();
        for (int i = 2; i < 16; ++i) {
            DataCopy(row, leaves[i * 16], {static_cast<uint16_t>(L), 2, 30, 0});
            PipeBarrier<PIPE_V>();
            Brcb(prod, row, L * 2, {2, 16});
            Brcb(prod[8], row, L * 2, {2, 16});
            PipeBarrier<PIPE_V>();
            Mul(prod, leaves, prod, E);
            PipeBarrier<PIPE_V>();
            // 与 solve_tri_pipeline.h 的 16 叶固定 Add 树逐条一致。
            Add(prod, prod, prod[128], 64, L, {1, 1, 1, 32, 32, 32});
            Add(prod[64], prod[64], prod[192], 64, L, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(prod, prod, prod[64], 64, L, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(prod, prod, prod[32], 32, L, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(prod, prod, prod[16], 16, L, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(leaves[i * 16], leaves[i * 16], prod, 16, L, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
        }
        for (int i = 0; i < 16; ++i) {
            uint64_t mask[2] = {1ULL << i, 0};
            Duplicate(leaves[i * 16], 1.0f, mask, L, 1, 32);
        }
        PipeBarrier<PIPE_V>();
        r_.template Fence<HardEvent::V_MTE3>();
        for (uint32_t h = 0; h < count; ++h) {
            auto d = Gm<float>(Head(slot, sub_ * Half + h) + p_.d0Offset, BT * 16);
            for (uint32_t leaf = 0; leaf < Leaves && leaf * 16 < c.valid; ++leaf) {
                const uint32_t rows = static_cast<uint32_t>(Min(16, c.valid - leaf * 16));
                DataCopyExtParams cp{static_cast<uint16_t>(rows), 16 * 4, 0, 0, 0};
                DataCopyPad(d[leaf * 256], leaves[(h * Leaves + leaf) * 256], cp);
            }
        }
    }

    __aicore__ inline void Publish(uint32_t slot)
    {
        // DCCI/DSB 是 Scalar 指令：先用 MTE3→S 明确关闭所有异步 GM 写，再 clean/DSB。
        // 不把 DDR DSB 的名字当作 MTE3 已完成的证据，也不在每条小搬运后排空。
        r_.template Fence<HardEvent::MTE3_S>();
        Invalidate(Gm<float>(core_ + p_.slotsOffset + slot * p_.slotBytes, p_.slotBytes / 4));
        CrossCoreSetFlag<0x2, PIPE_MTE3>(2 + slot);
    }

    template <int S>
    __aicore__ inline void Assemble(const Chunk &c, uint64_t headBegin, uint32_t activeHeads, uint32_t slot)
    {
        constexpr bool Final = 2 * S == BT;
        auto out = r_.template UB<float>(Local::kArena);
        auto typed = r_.template UB<InputT>(Local::kLeaf);
        auto leaves = r_.template UB<float>(Local::kLeaf);
        const uint32_t count = Heads(activeHeads);
        auto publicA = Gm<InputT>(a_.a, a_.batch * a_.hv * a_.tokens * BT);
        // Cube 的整包 result 已 ready，diag 在本 phase 内不再被生产者修改。
        // ENTIRE_DATA_CACHE 一次覆盖所有 head；空 AIV 不额外清理。
        if (count != 0) Invalidate(Gm<float>(Head(slot, sub_ * Half), p_.headBytes / sizeof(float)));
        for (uint32_t h = 0; h < count; ++h) {
            const uint32_t local = sub_ * Half + h;
            auto result = Gm<float>(Head(slot, local) + p_.resultOffset, BT * BT / 4);
            auto diag = Gm<float>(Head(slot, local) + (S == 32 ? p_.d1Offset : p_.d0Offset), BT * S);
            auto next = Gm<float>(Head(slot, local) + (S == 16 ? p_.d1Offset : p_.d0Offset), BT * 2 * S);
            for (uint32_t row = 0, pair = 0; row < c.valid; row += 2 * S, ++pair) {
                const uint32_t n0 = static_cast<uint32_t>(Min(S, c.valid - row));
                const uint32_t n1 = c.valid > row + S ? static_cast<uint32_t>(Min(S, c.valid - row - S)) : 0;
                r_.template Fence<HardEvent::MTE3_V>();
                Duplicate(out, 0.0f, 4 * S * S);
                PipeBarrier<PIPE_V>();
                r_.template Fence<HardEvent::V_MTE2>();
                if constexpr (S == 16) {
                    // 两个对角叶仍在 UB；绝不从旧 X/leaf GM 重新加载它们。
                    Copy(out, leaves[(h * Leaves + row / 16) * 256], n0, static_cast<uint8_t>(n0), {1, 1, 4, 2});
                    if (n1 > 0) Copy(out[2 * S * S + S], leaves[(h * Leaves + row / 16 + 1) * 256],
                        n1, static_cast<uint8_t>(n1), {1, 1, 4, 2});
                    PipeBarrier<PIPE_V>();
                } else {
                    LoadRect(out, diag[row * S], n0, n0, S, 2 * S);
                    if (n1 > 0) LoadRect(out[2 * S * S + S], diag[(row + S) * S], n1, n1, S, 2 * S);
                }
                if (n1 > 0) LoadRect(out[2 * S * S], result[pair * S * S], n1, n0, S, 2 * S);
                r_.template Fence<HardEvent::MTE2_V>();
                if (n1 > 0) Muls(out[2 * S * S], out[2 * S * S], -1.0f, S,
                    static_cast<uint8_t>(n1), {1, 1, static_cast<uint8_t>(2 * S / 8), static_cast<uint8_t>(2 * S / 8)});
                PipeBarrier<PIPE_V>();
                if constexpr (Final) {
                    Cast(typed, out, RoundMode::CAST_RINT, 4 * S * S);
                    PipeBarrier<PIPE_V>();
                    r_.template Fence<HardEvent::V_MTE3>();
                    DataCopyExtParams cp{static_cast<uint16_t>(n0 + n1), 2 * S * sizeof(InputT), 0, 0, 0};
                    DataCopyPad(publicA[HeadOffset(a_, c, headBegin + local, BT) + row * BT], typed, cp);
                } else {
                    r_.template Fence<HardEvent::V_MTE3>();
                    DataCopyExtParams cp{static_cast<uint16_t>(n0 + n1), 2 * S * 4, 0, 0, 0};
                    DataCopyPad(next[row * 2 * S], out, cp);
                }
            }
        }
    }

    __aicore__ inline void Rhs(const Chunk &c, uint64_t headBegin, uint32_t activeHeads, uint32_t slot)
    {
        constexpr uint32_t Rows = 32;
        constexpr uint32_t MatrixBytes = Rows * V * sizeof(InputT);
        auto in = r_.template UB<InputT>(Local::kLeaf);
        auto out = r_.template UB<InputT>(Local::kLeaf + MatrixBytes);
        auto fp = r_.template UB<float>(Local::kLeaf + 2 * MatrixBytes);
        auto broadcast = r_.template UB<float>(Local::kLeaf + 2 * MatrixBytes + Rows * V * 4);
        auto factor = r_.template UB<float>(Factor);
        auto k = Gm<InputT>(a_.k, a_.batch * a_.hk * a_.tokens * 128);
        auto v = Gm<InputT>(a_.v, a_.batch * a_.hv * a_.tokens * V);
        // 最终 A 搬出后才把 leaf/prod/assembly 物理区间整体换义为 RHS 工作区。
        r_.template Fence<HardEvent::MTE3_V>();
        r_.template Fence<HardEvent::MTE3_MTE2>();
        const uint32_t count = Heads(activeHeads);
        for (uint32_t h = 0; h < count; ++h) {
            const uint32_t local = sub_ * Half + h;
            const uint64_t global = headBegin + local;
            auto beta = r_.template UB<float>(Local::kBeta)[h * BT];
            auto g = r_.template UB<float>(Local::kG)[h * BT];
            Exp(factor, g, static_cast<uint32_t>(c.valid));
            PipeBarrier<PIPE_V>();
            Mul(factor, beta, factor, static_cast<uint32_t>(c.valid));
            PipeBarrier<PIPE_V>();
            auto rhs = Gm<InputT>(Head(slot, local), p_.headBytes / sizeof(InputT));
            // 顺序与原 WU 相同：V*beta->InputT，然后 K*(beta*exp(g))->InputT。
            for (uint32_t kind = 0; kind < 2; ++kind) {
                const uint32_t width = kind == 0 ? V : 128;
                const uint64_t sourceHead = kind == 0 ? global : global / (a_.hv / a_.hk);
                const uint64_t sourceHeads = kind == 0 ? a_.hv : a_.hk;
                const uint64_t sourceStride = QkvLayout == 1 ? sourceHeads * width : width;
                auto source = kind == 0 ? v : k;
                for (uint32_t row = 0; row < c.valid; row += Rows) {
                    const uint32_t rows = static_cast<uint32_t>(Min(Rows, c.valid - row));
                    r_.template Fence<HardEvent::V_MTE2>();
                    DataCopyExtParams load{static_cast<uint16_t>(rows), static_cast<uint32_t>(width * sizeof(InputT)),
                        static_cast<uint32_t>((sourceStride - width) * sizeof(InputT)), 0, 0};
                    DataCopyPadExtParams<InputT> pad{false, 0, 0, 0};
                    DataCopyPad(in, source[InputOffset<QkvLayout>(a_, c, sourceHead, sourceHeads, width, row)], load, pad);
                    r_.template Fence<HardEvent::MTE2_V>();
                    Cast(fp, in, RoundMode::CAST_NONE, rows * width);
                    PipeBarrier<PIPE_V>();
                    Brcb(broadcast, kind == 0 ? beta[row] : factor[row], static_cast<uint8_t>((rows + 7) / 8), {1, 8});
                    PipeBarrier<PIPE_V>();
                    for (uint32_t col = 0; col < width; col += 64) {
                        Mul(fp[col], fp[col], broadcast, 64, static_cast<uint8_t>(rows),
                            {1, 1, 0, static_cast<uint8_t>(width / 8), static_cast<uint8_t>(width / 8), 1});
                    }
                    PipeBarrier<PIPE_V>();
                    r_.template Fence<HardEvent::MTE3_V>();
                    Cast(out, fp, RoundMode::CAST_RINT, rows * width);
                    PipeBarrier<PIPE_V>();
                    r_.template Fence<HardEvent::V_MTE3>();
                    DataCopyExtParams store{static_cast<uint16_t>(rows), static_cast<uint32_t>(width * sizeof(InputT)), 0, 0, 0};
                    const uint64_t base = kind == 0 ? p_.vbOffset : p_.kbgOffset;
                    DataCopyPad(rhs[base / sizeof(InputT) + row * width], out, store);
                }
            }
        }
    }
};
} // namespace GdnUnifiedPrepare
#endif
