// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. 只复用 Catlass 的搬运/MMAD 原语，整段调度属于统一 Prepare。
#ifndef GDN_ARCH22_UNIFIED_PREPARE_CUBE_H
#define GDN_ARCH22_UNIFIED_PREPARE_CUBE_H

#include <type_traits>
#include "gdn_unified_prepare_resources.h"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace GdnUnifiedPrepare {
template <typename InputT, int V, int BT, int QkvLayout>
class Cube {
    using Tag = Catlass::Arch::AtlasA2;
    template <class T, class C, bool TransB>
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<Tag, T, Catlass::layout::RowMajor,
        T, std::conditional_t<TransB, Catlass::layout::ColumnMajor, Catlass::layout::RowMajor>,
        C, Catlass::layout::RowMajor>;
    Resources &r_;
    const Args &a_;
    WorkspaceLayout p_;
    GM_ADDR core_;

    template <class T, class C, bool TransB, class LA, class LB>
    __aicore__ inline void MultiplyL1(const LA &left, const LB &right, GlobalTensor<C> output,
                                    uint32_t m, uint32_t n, uint32_t k, uint64_t stride,
                                    uint32_t kStep, bool releaseGeneral)
    {
        using TC = Copy<T, C, TransB>;
        typename TC::CopyL1ToL0A loadA;
        typename TC::CopyL1ToL0B loadB;
        Catlass::Gemm::Tile::TileMmadTla<Tag, T, typename TC::LayoutTagL1A> mm;
        const uint32_t mmRows = m == 1 ? 16 : m;
        auto c0 = tla::MakeTensor(r_.L0C(), tla::MakeLayoutL0C(mmRows, n), Catlass::Arch::PositionL0C{});
        WaitFlag<HardEvent::FIX_M>(r_.cFree);
        for (uint32_t ko = 0; ko < k; ko += kStep) {
            const uint32_t kk = static_cast<uint32_t>(Min(kStep, k - ko));
            auto al0 = tla::MakeTensor(r_.template L0A<T>(),
                tla::MakeLayout<T, typename TC::LayoutTagL0A>(mmRows, kk), Catlass::Arch::PositionL0A{});
            auto bl0 = tla::MakeTensor(r_.template L0B<T>(),
                tla::MakeLayout<T, typename TC::LayoutTagL0B>(kk, n), Catlass::Arch::PositionL0B{});
            auto al1 = tla::GetTile(left, tla::MakeCoord(0, ko), tla::MakeShape(mmRows, kk));
            auto bl1 = tla::GetTile(right, tla::MakeCoord(ko, 0), tla::MakeShape(kk, n));
            WaitFlag<HardEvent::M_MTE1>(r_.l0Free);
            loadA(al0, al1);
            loadB(bl0, bl1);
            if (releaseGeneral && ko + kk == k) {
                // 这是两个通用 L1 区间真正的末次 MTE1 读，独立于 GM/C 的生命周期。
                SetFlag<HardEvent::MTE1_MTE2>(r_.l1Free[0]);
                SetFlag<HardEvent::MTE1_MTE2>(r_.l1Free[1]);
            }
            SetFlag<HardEvent::MTE1_M>(r_.l0Ready);
            WaitFlag<HardEvent::MTE1_M>(r_.l0Ready);
            mm(c0, al0, bl0, mmRows, n, kk, ko == 0, ko + kk == k ? 0b11 : 0b10);
            SetFlag<HardEvent::M_MTE1>(r_.l0Free);
        }
        SetFlag<HardEvent::M_FIX>(r_.cReady);
        WaitFlag<HardEvent::M_FIX>(r_.cReady);
        auto cg = tla::MakeTensor(output,
            tla::MakeLayout(tla::MakeShape(m, n), tla::MakeStride(stride, tla::Int<1>{})),
            Catlass::Arch::PositionGM{});
        typename TC::template CopyL0CToGm<decltype(cg)> store;
        store(cg, c0, 0b11);
        SetFlag<HardEvent::FIX_M>(r_.cFree);
    }

    template <class T, class C, bool TransB = false>
    __aicore__ inline void GmProduct(GlobalTensor<T> left, GlobalTensor<T> right, GlobalTensor<C> output,
                                   uint32_t m, uint32_t n, uint32_t k,
                                   uint64_t sa, uint64_t sb, uint64_t sc)
    {
        using TC = Copy<T, C, TransB>;
        using LayoutB = std::conditional_t<TransB, Catlass::layout::ColumnMajor, Catlass::layout::RowMajor>;
        auto ag = tla::MakeTensor(left,
            tla::MakeLayoutFromTag(Catlass::layout::RowMajor(m, k, sa)), Catlass::Arch::PositionGM{});
        auto bg = tla::MakeTensor(right,
            tla::MakeLayoutFromTag(LayoutB(k, n, sb)), Catlass::Arch::PositionGM{});
        const uint32_t mmRows = m == 1 ? 16 : m;
        auto al1 = tla::MakeTensor(r_.template L1<T>(Local::kGemmA),
            tla::MakeLayout<T, typename TC::LayoutTagL1A>(mmRows, k), Catlass::Arch::PositionL1{});
        auto bl1 = tla::MakeTensor(r_.template L1<T>(Local::kGemmB),
            tla::MakeLayout<T, typename TC::LayoutTagL1B>(k, n), Catlass::Arch::PositionL1{});
        typename TC::template CopyGmToL1A<decltype(ag)> copyA;
        typename TC::template CopyGmToL1B<decltype(bg)> copyB;
        WaitFlag<HardEvent::MTE1_MTE2>(r_.l1Free[0]);
        WaitFlag<HardEvent::MTE1_MTE2>(r_.l1Free[1]);
        copyA(al1, ag);
        copyB(bl1, bg);
        SetFlag<HardEvent::MTE2_MTE1>(r_.l1Ready);
        WaitFlag<HardEvent::MTE2_MTE1>(r_.l1Ready);
        MultiplyL1<T, C, TransB>(al1, bl1, output, m, n, k, sc, k, true);
    }

    __aicore__ inline GM_ADDR Head(uint32_t slot, uint32_t localHead) const
    { return core_ + p_.slotsOffset + slot * p_.slotBytes + localHead * p_.headBytes; }

    template <class T, bool IsA>
    __aicore__ inline void ResidentCopy(GlobalTensor<T> input, uint32_t offset,
                                       uint32_t m, uint32_t n, uint64_t stride)
    {
        using TC = Copy<T, T, false>;
        auto gm = tla::MakeTensor(input,
            tla::MakeLayoutFromTag(Catlass::layout::RowMajor(m, n, stride)), Catlass::Arch::PositionGM{});
        if constexpr (IsA) {
            auto l1 = tla::MakeTensor(r_.template L1<T>(offset),
                tla::MakeLayout<T, typename TC::LayoutTagL1A>(m == 1 ? 16 : m, n), Catlass::Arch::PositionL1{});
            typename TC::template CopyGmToL1A<decltype(gm)> copy;
            copy(l1, gm);
        } else {
            auto l1 = tla::MakeTensor(r_.template L1<T>(offset),
                tla::MakeLayout<T, typename TC::LayoutTagL1B>(m, n), Catlass::Arch::PositionL1{});
            typename TC::template CopyGmToL1B<decltype(gm)> copy;
            copy(l1, gm);
        }
    }

public:
    __aicore__ inline Cube(Resources &r, const Args &args, WorkspaceLayout p)
        : r_(r), a_(args), p_(p), core_(args.scratch + GroupId() * p.coreBytes) {}

    __aicore__ inline void Scores(const Chunk &c)
    {
        auto kg = Gm<InputT>(a_.k, a_.batch * a_.hk * a_.tokens * 128);
        auto scores = Gm<float>(core_, p_.scoreBytes / sizeof(float));
        for (uint64_t hk = c.hkBegin; hk < c.hkEnd; ++hk) {
            const uint64_t offset = InputOffset<QkvLayout>(a_, c, hk, a_.hk, 128);
            const uint64_t stride = QkvLayout == 1 ? a_.hk * 128 : 128;
            GmProduct<InputT, float, true>(kg[offset], kg[offset], scores[(hk - c.hkBegin) * BT * BT],
                c.valid, c.valid, 128, stride, stride, BT);
        }
        // raw score 仅在下一个 HK task 开始时覆盖；本 task 的所有 G>pack waves 只读它。
    }

    template <int S>
    __aicore__ inline void Merge(const Chunk &c, uint32_t slot, uint32_t activeHeads)
    {
        static_assert(S == 16 || S == 32 || S == 64);
        for (uint32_t h = 0; h < activeHeads; ++h) {
            GM_ADDR head = Head(slot, h);
            auto x = Gm<float>(head + p_.xOffset, BT * BT);
            const uint64_t dOffset = S == 32 ? p_.d1Offset : p_.d0Offset;
            auto diag = Gm<float>(head + dOffset, BT * S);
            Invalidate(x);
            Invalidate(diag);
            for (uint32_t row = 0, pair = 0; row < c.valid; row += 2 * S, ++pair) {
                const uint32_t n0 = static_cast<uint32_t>(Min(S, c.valid - row));
                const uint32_t n1 = c.valid > row + S ? static_cast<uint32_t>(Min(S, c.valid - row - S)) : 0;
                if (n1 == 0) continue;
                // GDN 固定合并顺序的首乘。off 的 GM 读取先完整进入 L1，才可能 FIX 覆盖。
                // quadtree 各层 offdiag 互不相交，此块没有后层读者，可原址换义为 tempX。
                GmProduct<float, float>(diag[(row + S) * S], x[(row + S) * BT + row],
                    x[(row + S) * BT + row], n1, n0, n1, S, BT, BT);
            }
        }
        // 整包首乘已发出后统一交接；不在每个子块之间做 first→second 串行等待。
        r_.template Fence<HardEvent::FIX_MTE2>();
        r_.template Fence<HardEvent::MTE2_S>(); // FIX→MTE2→Scalar，之后的 DCCI 才可能执行。
        for (uint32_t h = 0; h < activeHeads; ++h) {
            GM_ADDR head = Head(slot, h);
            auto tempX = Gm<float>(head + p_.xOffset, BT * BT);
            auto diag = Gm<float>(head + (S == 32 ? p_.d1Offset : p_.d0Offset), BT * S);
            auto result = Gm<float>(head + p_.resultOffset, BT * BT / 4);
            // FIX 新写 tempX 后的读侧可见性加固，位置必须在首乘完成之后。
            Invalidate(tempX);
            for (uint32_t row = 0, pair = 0; row < c.valid; row += 2 * S, ++pair) {
                const uint32_t n0 = static_cast<uint32_t>(Min(S, c.valid - row));
                const uint32_t n1 = c.valid > row + S ? static_cast<uint32_t>(Min(S, c.valid - row - S)) : 0;
                if (n1 == 0) continue;
                GmProduct<float, float>(tempX[(row + S) * BT + row], diag[row * S], result[pair * S * S],
                    n1, n0, n0, BT, S, S);
            }
        }
    }

    __aicore__ inline void WU(const Chunk &c, uint64_t headBegin, uint32_t activeHeads, uint32_t slot)
    {
        using TC = Copy<InputT, InputT, false>;
        auto ag = Gm<InputT>(a_.a, a_.batch * a_.hv * a_.tokens * BT);
        auto wg = Gm<InputT>(a_.w, a_.batch * a_.hv * a_.tokens * 128);
        auto ug = Gm<InputT>(a_.u, a_.batch * a_.hv * a_.tokens * V);
        const uint32_t valid = static_cast<uint32_t>(c.valid);
        for (uint32_t first = 0; first < activeHeads; first += 2) {
            const uint32_t count = static_cast<uint32_t>(Min(2, activeHeads - first));
            // 此许可属于 L1 lane；GM slot 的许可只在下面整包最后 pair 的读取后归还。
            WaitFlag<HardEvent::MTE1_MTE2>(r_.rhsFree);
            for (uint32_t pair = 0; pair < count; ++pair) {
                const uint32_t h = first + pair;
                const uint64_t globalHead = headBegin + h;
                auto rhs = Gm<InputT>(Head(slot, h), p_.headBytes / sizeof(InputT));
                Invalidate(rhs);
                Invalidate(ag[HeadOffset(a_, c, globalHead, BT)]);
                ResidentCopy<InputT, true>(ag[HeadOffset(a_, c, globalHead, BT)],
                    Local::kRhsA + pair * BT * BT * sizeof(InputT), valid, valid, BT);
                ResidentCopy<InputT, false>(rhs[p_.kbgOffset / sizeof(InputT)],
                    Local::kRhsK + pair * BT * 128 * sizeof(InputT), valid, 128, 128);
                ResidentCopy<InputT, false>(rhs[p_.vbOffset / sizeof(InputT)],
                    Local::kRhsV + pair * BT * V * sizeof(InputT), valid, V, V);
            }
            SetFlag<HardEvent::MTE2_MTE1>(r_.l1Ready);
            if (first + count == activeHeads) {
                // 整包的全部 A/Kbg/Vb GM 读取已发出；从 MTE2 发布保证最后一个读已完成。
                // 此后最后 pair 的 MMAD/FIX 仍可执行，下一 pack 可重用另一个/此 GM 版本。
                CrossCoreSetFlag<0x2, PIPE_MTE2>(6 + slot);
            }
            WaitFlag<HardEvent::MTE2_MTE1>(r_.l1Ready);
            for (uint32_t pair = 0; pair < count; ++pair) {
                const uint64_t globalHead = headBegin + first + pair;
                const uint32_t mmRows = valid == 1 ? 16 : valid;
                auto left = tla::MakeTensor(
                    r_.template L1<InputT>(Local::kRhsA + pair * BT * BT * sizeof(InputT)),
                    tla::MakeLayout<InputT, typename TC::LayoutTagL1A>(mmRows, valid), Catlass::Arch::PositionL1{});
                auto vb = tla::MakeTensor(
                    r_.template L1<InputT>(Local::kRhsV + pair * BT * V * sizeof(InputT)),
                    tla::MakeLayout<InputT, typename TC::LayoutTagL1B>(valid, V), Catlass::Arch::PositionL1{});
                auto kbg = tla::MakeTensor(
                    r_.template L1<InputT>(Local::kRhsK + pair * BT * 128 * sizeof(InputT)),
                    tla::MakeLayout<InputT, typename TC::LayoutTagL1B>(valid, 128), Catlass::Arch::PositionL1{});
                // 原 V256 路径的累加边界为 64，原 V128 路径为 128；W 与 U 使用同一边界。
                MultiplyL1<InputT, InputT, false>(left, vb, ug[HeadOffset(a_, c, globalHead, V)],
                    valid, V, valid, V, V == 256 ? 64 : 128, false);
                MultiplyL1<InputT, InputT, false>(left, kbg, wg[HeadOffset(a_, c, globalHead, 128)],
                    valid, 128, valid, 128, V == 256 ? 64 : 128, false);
            }
            SetFlag<HardEvent::MTE1_MTE2>(r_.rhsFree);
        }
    }
};
} // namespace GdnUnifiedPrepare
#endif
