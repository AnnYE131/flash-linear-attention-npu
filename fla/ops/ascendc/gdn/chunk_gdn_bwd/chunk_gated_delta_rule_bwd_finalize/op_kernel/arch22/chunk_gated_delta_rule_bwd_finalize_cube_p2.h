#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ARCH22_CUBE_P2_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ARCH22_CUBE_P2_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "../chunk_gated_delta_rule_bwd_finalize_struct.h"

namespace GDN {

// P2 把五个没有中间依赖的 GEMM 交给 Cube。输出在 workspace 中按
// (chunk, HV) 固定寻址；单次全局同步后由 AIV 消费。
class ChunkGatedDeltaRuleBwdFinalizeCubeP2 {
    using DT = bfloat16_t;
    using ArchTag = Catlass::Arch::AtlasA2;
    using RowMajor = Catlass::layout::RowMajor;
    using ColumnMajor = Catlass::layout::ColumnMajor;
    using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true>;
    using L1TileShape = tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<128>>;
    using L0TileShape = tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<128>>;
    using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, RowMajor, DT, ColumnMajor, float, RowMajor>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, DT, DT, float, void, TileCopy>;

    static constexpr int64_t BT = 64;
    static constexpr int64_t D = 128;
    static constexpr int64_t RAW_W = BT * BT;
    static constexpr int64_t RAW_Q = RAW_W + BT * D;
    static constexpr int64_t RAW_K = RAW_Q + BT * D;
    static constexpr int64_t RAW_V = RAW_K + BT * D;
    static constexpr int64_t RAW_ELEMS = RAW_V + BT * BT;

public:
    __aicore__ inline void Init(
        GM_ADDR dO, GM_ADDR x, GM_ADDR p, GM_ADDR h, GM_ADDR dh, GM_ADDR v,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR rawWorkspace,
        const ChunkGatedDeltaRuleBwdFinalizeTilingData *tiling)
    {
        t_ = tiling;
        do_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dO));
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(x));
        p_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(p));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(h));
        dh_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dh));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(v));
        raw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rawWorkspace));
        if (t_->isVariable != 0) {
            cu_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
            indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
        }
    }

    __aicore__ inline void Process()
    {
        using namespace Catlass;
        using namespace tla;
        AscendC::SetHF32Mode(false);
        Arch::Resource<ArchTag> resource;
        BlockMmad mm(resource);
        const int64_t block = AscendC::GetBlockIdx();
        const int64_t blocks = AscendC::GetBlockNum();
        const auto leftLayout = RowMajor::MakeLayout<DT>(BT, D);
        const auto fRightLayout = ColumnMajor::MakeLayout<DT>(D, BT);
        const auto wRightLayout = ColumnMajor::MakeLayout<DT>(D, D);
        const auto fOutLayout = RowMajor::MakeLayout<float>(BT, BT);
        const auto wOutLayout = RowMajor::MakeLayout<float>(BT, D);
        const int64_t stateNt =
            t_->isVariable != 0 ? t_->totalChunkNum : t_->chunkNumForT;
        for (int64_t work = block;
             work < t_->totalChunkNum * t_->HV; work += blocks) {
            const int64_t chunkTask = work / t_->HV;
            const int64_t hv = work % t_->HV;
            int64_t b = 0, start = 0, ci = 0, m = 0;
            if (t_->isVariable == 0) {
                b = chunkTask / t_->chunkNumForT;
                ci = chunkTask % t_->chunkNumForT;
                start = ci * BT;
                m = t_->T - start;
            } else {
                const int64_t seq = indices_.GetValue(chunkTask * 2);
                const int64_t localChunk = indices_.GetValue(chunkTask * 2 + 1);
                if (seq < 0 || seq >= t_->seqNum || localChunk < 0) {
                    continue;
                }
                start = cu_.GetValue(seq) + localChunk * BT;
                m = cu_.GetValue(seq + 1) - start;
                ci = chunkTask;
            }
            if (m > BT) {
                m = BT;
            }
            if (m <= 0) {
                continue;
            }
            const int64_t vecOffset = ((b * t_->HV + hv) * t_->T + start) * D;
            const int64_t stateOffset =
                ((b * t_->HV + hv) * stateNt + ci) * D * D;
            const int64_t rawOffset = work * RAW_ELEMS;
            auto tensorDo = tla::MakeTensor(do_[vecOffset], MakeLayoutFromTag(leftLayout),
                                       Arch::PositionGM{});
            auto tensorX = tla::MakeTensor(x_[vecOffset], MakeLayoutFromTag(fRightLayout),
                                      Arch::PositionGM{});
            auto tensorF = tla::MakeTensor(raw_[rawOffset], MakeLayoutFromTag(fOutLayout),
                                      Arch::PositionGM{});
            auto blockDoF = tla::GetTile(tensorDo, tla::MakeCoord(0, 0), tla::MakeShape(m, D));
            auto blockX = tla::GetTile(tensorX, tla::MakeCoord(0, 0), tla::MakeShape(D, m));
            auto blockF = tla::GetTile(tensorF, tla::MakeCoord(0, 0), tla::MakeShape(m, m));
            mm(blockDoF, blockX, blockF,
               Catlass::GemmCoord{static_cast<uint32_t>(m),
                                  static_cast<uint32_t>(m), D});

            auto tensorP = tla::MakeTensor(p_[vecOffset], MakeLayoutFromTag(leftLayout),
                                      Arch::PositionGM{});
            auto tensorH = tla::MakeTensor(h_[stateOffset], MakeLayoutFromTag(wRightLayout),
                                      Arch::PositionGM{});
            auto tensorW = tla::MakeTensor(raw_[rawOffset + RAW_W],
                                      MakeLayoutFromTag(wOutLayout),
                                      Arch::PositionGM{});
            auto blockP = tla::GetTile(tensorP, tla::MakeCoord(0, 0), tla::MakeShape(m, D));
            auto blockH = tla::GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(D, D));
            auto blockW = tla::GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(m, D));
            mm(blockP, blockH, blockW,
               Catlass::GemmCoord{static_cast<uint32_t>(m), D, D});

            auto tensorQ = tla::MakeTensor(raw_[rawOffset + RAW_Q],
                                      MakeLayoutFromTag(wOutLayout),
                                      Arch::PositionGM{});
            auto blockQ = tla::GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(m, D));
            mm(blockDoF, blockH, blockQ,
               Catlass::GemmCoord{static_cast<uint32_t>(m), D, D});

            auto tensorXLeft = tla::MakeTensor(x_[vecOffset],
                                      MakeLayoutFromTag(leftLayout),
                                      Arch::PositionGM{});
            auto tensorDh = tla::MakeTensor(dh_[stateOffset],
                                      MakeLayoutFromTag(wRightLayout),
                                      Arch::PositionGM{});
            auto tensorK = tla::MakeTensor(raw_[rawOffset + RAW_K],
                                      MakeLayoutFromTag(wOutLayout),
                                      Arch::PositionGM{});
            auto blockXLeft = tla::GetTile(tensorXLeft, tla::MakeCoord(0, 0), tla::MakeShape(m, D));
            auto blockDh = tla::GetTile(tensorDh, tla::MakeCoord(0, 0), tla::MakeShape(D, D));
            auto blockK = tla::GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(m, D));
            mm(blockXLeft, blockDh, blockK,
               Catlass::GemmCoord{static_cast<uint32_t>(m), D, D});

            const auto vOffset = ((b * t_->HV + hv) * t_->T + start) * D;
            auto tensorVRight = tla::MakeTensor(v_[vOffset],
                                      MakeLayoutFromTag(fRightLayout),
                                      Arch::PositionGM{});
            auto tensorVOut = tla::MakeTensor(raw_[rawOffset + RAW_V],
                                      MakeLayoutFromTag(fOutLayout),
                                      Arch::PositionGM{});
            auto blockVRight = tla::GetTile(tensorVRight, tla::MakeCoord(0, 0), tla::MakeShape(D, m));
            auto blockVOut = tla::GetTile(tensorVOut, tla::MakeCoord(0, 0), tla::MakeShape(m, m));
            mm(blockP, blockVRight, blockVOut,
               Catlass::GemmCoord{static_cast<uint32_t>(m),
                                  static_cast<uint32_t>(m), D});
        }
    }

private:
    const ChunkGatedDeltaRuleBwdFinalizeTilingData *t_ = nullptr;
    AscendC::GlobalTensor<DT> do_, x_, p_, h_, dh_, v_;
    AscendC::GlobalTensor<float> raw_;
    AscendC::GlobalTensor<int64_t> cu_, indices_;
};

} // namespace GDN
#endif
