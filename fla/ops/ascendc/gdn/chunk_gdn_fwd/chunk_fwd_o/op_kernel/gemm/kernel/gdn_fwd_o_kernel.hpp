/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdo_output.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "../block/block_scheduler_gdn_fwd_o.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using _0 = tla::Int<0>;
using _1 = tla::Int<1>;
using _2 = tla::Int<2>;
using _4 = tla::Int<4>;
using _8 = tla::Int<8>;
using _16 = tla::Int<16>;
using _32 = tla::Int<32>;
using _64 = tla::Int<64>;
using _128 = tla::Int<128>;
using _256 = tla::Int<256>;
using _512 = tla::Int<512>;
using _1024 = tla::Int<1024>;
using _2048 = tla::Int<2048>;
using _4096 = tla::Int<4096>;
using _8192 = tla::Int<8192>;
using _16384 = tla::Int<16384>;
using _32768 = tla::Int<32768>;
using _65536 = tla::Int<65536>;

#else
#define CATLASS_ARCH 2201

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdo_output.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_pipelined.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "../block/block_scheduler_gdn_fwd_o.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#endif

#include "kernel_operator.h"
#include "../../chunk_fwd_o_struct.h"
using namespace Catlass;
using namespace tla;

// template <>
namespace Catlass::Gemm::Kernel {

template<
    typename INPUT_TYPE,
    typename G_TYPE,
    typename WORKSPACE_TYPE
>
class GDNFwdOKernel {
public:

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using ArchTag = Arch::Ascend950;
#else
    using ArchTag = Arch::AtlasA2;
#endif
    using GDNFwdOOffsets = Catlass::Gemm::Block::GDNFwdOOffsets;

    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOCube;
    using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOVec;
    // [R12] 分组调度器（A2）：task = (chunk, taskGroupSize 连续 vHead)，跨核握手按组征收
    using GroupCubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOGroupCube;
    using GroupVecScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOGroupVec;

    using DispatchPolicyTla = Gemm::MmadPingpongTlaMulti<ArchTag, true, false>;
    using L1TileShapeQKTla = Shape<_128, _128, _128>;
    using L0TileShapeQKTla = L1TileShapeQKTla;
    using L1TileShapeV128Tla = Shape<_128, _128, _128>;
    using L0TileShapeV128Tla = L1TileShapeV128Tla;
    using L1TileShapeV256Tla = Shape<_128, _256, _128>;
    using L0TileShapeV256Tla = Shape<_128, _256, _64>;
    using QType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using KType = Gemm::GemmType<INPUT_TYPE, layout::ColumnMajor>;
    using AttenType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using AttenMaskedType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using HType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using OinterType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using VNEWType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;

    using GType = Gemm::GemmType<G_TYPE, layout::RowMajor>;
    using OType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using MaskType = Gemm::GemmType<bool, layout::RowMajor>;

    // cube 1
    using TileCopyQK = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::ColumnMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeQKTla, L0TileShapeQKTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQK>;

    // cube 2
    using TileCopyQH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadQH128 = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeV128Tla, L0TileShapeV128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;
    using BlockMmadQH256 = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeV256Tla, L0TileShapeV256Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;

    // cube 3
    using TileCopyAttenVNEW = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadAttenVNEW128 = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeV128Tla, L0TileShapeV128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;
    using BlockMmadAttenVNEW256 = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeV256Tla, L0TileShapeV256Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
    // [R11] chunk64 重叠路径类型（仅 A2/arch22）：L1 空间分区 + L0 时分复用
    // （A5 Pipelined 同构资源策略；R13 起另有 chunk128 同构类型见下）。chunk64 下
    // QK tile M=N=blockTokens<=64：
    // L1A tile 64x128 bf16=16KB、L1B tile 128x128 bf16=32KB（QK 的 N<=64 只是
    // 用不满）、L0C tile 64x128 fp32=32KB << 128KB——R4 时代"128x128 tile 占满
    // L0C"的 507015 冲突容量根源消除；L0A/L0B 同样宽裕。
    // 策略（与 A5 实证过的结构一致）：
    //   - L1 三实例独立区+独立 event id（装载与上一 Cube 计算重叠的前提）：
    //     QK [0,48K)/L1A id0,L1B id1；QH [48K,96K)/id2,3；AV [96K,144K)/id4,5
    //   - L0A/L0B/L0C 不分区，跨实例窗口互斥由相位结构保证（waitL0Drained/
    //     finalWaitFlags + Cube2→Cube3 的 MTE2/FIX 双屏障，与串行路径同机制）
    //   - ENABLE_UNIT_FLAG=true：L0 同步走硬件 unitFlag 流控不占 event id
    //   - L1A/L1B_STAGES=1：K=128 单 k-block 无 k 向预取需求，省 event 预算
    // Tile 形状：M=64 匹配 blockTokens<=64；K=128；N=128 覆盖 QK(N<=64) 与
    // QH/AV(N=vBlockDim<=128)。V256（vBlockDim>128）不进本路径。
    using DispatchPolicyTlaOvl = Gemm::MmadPingpongTlaGdnFwdO<ArchTag, true, false, 1, false, false, 1, 1, 2, 2>;
    using L1TileShapeOvlTla = Shape<_64, _128, _128>;
    using L0TileShapeOvlTla = L1TileShapeOvlTla;
    using BlockMmadQKOvl = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaOvl, L1TileShapeOvlTla, L0TileShapeOvlTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQK>;
    using BlockMmadQHOvl = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaOvl, L1TileShapeOvlTla, L0TileShapeOvlTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;
    using BlockMmadAttenVNEWOvl = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaOvl, L1TileShapeOvlTla, L0TileShapeOvlTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;
    // [R13] chunk128 重叠路径类型：tile 与串行生产路径 V128 同款 Shape<_128,_128,_128>，
    // 仅 DispatchPolicy 换 Ovl（unitFlag + L1A/L1B 单槽化省 event 预算）。L1 分区预算：
    // L1A 32K + L1B 32K = 64KB/实例 ×3（QK/QH/AV）= 192KB <= L1 512KB；L0A/L0B 各双槽
    // 64KB、L0C 单槽 64KB（fp32 128×128）与串行 chunk128 生产路径完全同容量——R4 时代
    // 128 tile 的 507015 根源是多实例 L0 窗口交错，R11 的 L1 分区 + 相位互斥结构已消除。
    using L1TileShapeOvl128Tla = Shape<_128, _128, _128>;
    using L0TileShapeOvl128Tla = L1TileShapeOvl128Tla;
    using BlockMmadQKOvl128 = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaOvl, L1TileShapeOvl128Tla, L0TileShapeOvl128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQK>;
    using BlockMmadQHOvl128 = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaOvl, L1TileShapeOvl128Tla, L0TileShapeOvl128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;
    using BlockMmadAttenVNEWOvl128 = Gemm::Block::BlockMmadTlaPipelined<DispatchPolicyTlaOvl, L1TileShapeOvl128Tla, L0TileShapeOvl128Tla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;
#endif

    // vec 1
    using DispatchPolicyGDNFwdOQkmask = Epilogue::EpilogueAtlasGDNFwdOQkmask;
    using EpilogueGDNFwdOQkmask = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdOQkmask, AttenMaskedType, GType, AttenType, MaskType>;

    // vec 2
    using DispatchPolicyGDNFwdOOutput = Epilogue::EpilogueAtlasGDNFwdOOutput;
    using EpilogueGDNFwdOOutput = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdOOutput, OType, GType, OinterType, OinterType>;

    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = Catlass::layout::RowMajor;

    using ElementK =  typename BlockMmadQK::ElementB;
    using LayoutK = Catlass::layout::ColumnMajor;

    using ElementAtten = typename BlockMmadQK::ElementC;
    using LayoutAtten = Catlass::layout::RowMajor;

    using ElementAttenMasked = typename BlockMmadQH128::ElementA;
    using LayoutAttenMasked = Catlass::layout::RowMajor;

    using ElementH = typename BlockMmadQH128::ElementB;
    using LayoutH = Catlass::layout::RowMajor;

    using ElementOinter = typename BlockMmadQH128::ElementC;
    using LayoutOinter = Catlass::layout::RowMajor;


    using ElementVNEW = typename BlockMmadAttenVNEW128::ElementB;
    using LayoutVNEW = Catlass::layout::RowMajor;


    using ElementG = G_TYPE;
    using ElementMask = bool;

    using L1TileShape = typename BlockMmadQK::L1TileShape;

    uint32_t shapeBatch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    float scale;
    uint32_t numChunks;
    uint32_t isVariedLen;
    uint32_t tokenBatch;
    int64_t vWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t attnWorkspaceOffset;
    int64_t aftermaskWorkspaceOffset;
    int64_t maskWorkspaceOffset;

    AscendC::GlobalTensor<ElementQ> gmQ;
    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementVNEW> gmV;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementVNEW> gmO;
    AscendC::GlobalTensor<ElementOinter> gmVWorkspace;
    AscendC::GlobalTensor<ElementOinter> gmHWorkspace;
    AscendC::GlobalTensor<ElementAtten> gmAttnWorkspace;
    AscendC::GlobalTensor<ElementAttenMasked> gmAftermaskWorkspace;
    AscendC::GlobalTensor<ElementMask> gmMask;

    CubeScheduler cubeBlockScheduler;
    VecScheduler vecBlockScheduler;

    // [R12] 分组路径状态（groupedMode 在 Init 按 tiling 判定，AIC/AIV 两侧同值
    // 同源——分派不一致会造成跨核 flag 失配死锁）
    bool groupedMode{false};
    GroupCubeScheduler groupCubeScheduler;
    GroupVecScheduler groupVecScheduler;

    Arch::Resource<ArchTag> resource;

    __aicore__ inline GDNFwdOKernel() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
        GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR o, const GDN::ChunkFwdOTilingData *tilingData, GM_ADDR user) {

        shapeBatch = tilingData->shapeBatch;
        seqlen = tilingData->seqlen;
        kNumHead = tilingData->kNumHead;
        vNumHead = tilingData->vNumHead;
        kHeadDim = tilingData->kHeadDim;
        vHeadDim = tilingData->vHeadDim;
        scale = tilingData->scale;
        chunkSize = tilingData->chunkSize;
        isVariedLen = tilingData->isVariedLen;
        tokenBatch = tilingData->tokenBatch;
        vWorkspaceOffset = tilingData->vWorkspaceOffset;
        hWorkspaceOffset = tilingData->hWorkspaceOffset;
        attnWorkspaceOffset = tilingData->attnWorkspaceOffset;
        aftermaskWorkspaceOffset = tilingData->aftermaskWorkspaceOffset;
        maskWorkspaceOffset = tilingData->maskWorkspaceOffset;

        gmQ.SetGlobalBuffer((__gm__ ElementQ *)q);
        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmV.SetGlobalBuffer((__gm__ ElementVNEW *)v);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmG.SetGlobalBuffer((__gm__ ElementG *)g);
        gmO.SetGlobalBuffer((__gm__ ElementVNEW *)o);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementOinter *)(user + vWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementOinter *)(user + hWorkspaceOffset));
        gmAttnWorkspace.SetGlobalBuffer((__gm__ ElementAtten *)(user + attnWorkspaceOffset));
        gmAftermaskWorkspace.SetGlobalBuffer((__gm__ ElementAttenMasked *)(user + aftermaskWorkspaceOffset));
        gmMask.SetGlobalBuffer((__gm__ ElementMask *)(user + maskWorkspaceOffset));

        if ASCEND_IS_AIC {
            cubeBlockScheduler.Init(cu_seqlens, chunk_offsets, tilingData);
        }

        if ASCEND_IS_AIV {
            vecBlockScheduler.Init(cu_seqlens, chunk_offsets, tilingData);
        }

        // [R12] 分组模式判定（AIC/AIV 同式同源，跨核 flag 失配即死锁）：
        //   vNumHead 为 kNumHead 整数倍且 headGroup 数>1 才有分组意义（MHA 1:1
        //   分组即单 task 路径，白付扩容无收益）；taskGroupSize ∈ {3,4}（host
        //   公式上界）；不排除 R11/R13 重叠路径——重叠路径在 Process 内优先分派，
        //   本守卫只覆盖未进重叠路径的调用形态。
        groupedMode = (tilingData->vNumHead % tilingData->kNumHead == 0) &&
                      (tilingData->vNumHead / tilingData->kNumHead > 1) &&
                      (tilingData->vNumHead / tilingData->kNumHead <= 4) &&
                      (tilingData->taskGroupSize >= 2) && (tilingData->taskGroupSize <= 4) &&
                      (tilingData->vNumHead % tilingData->taskGroupSize != 0 ||
                       tilingData->vNumHead / tilingData->taskGroupSize > 1);
        if (groupedMode) {
            if ASCEND_IS_AIC {
                groupCubeScheduler.Init(cu_seqlens, chunk_offsets, tilingData);
            }
            if ASCEND_IS_AIV {
                groupVecScheduler.Init(cu_seqlens, chunk_offsets, tilingData);
            }
        }
    }

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
    /// [R11/R13] varlen 专属三段式重叠路径（A5 Pipelined 相位结构的 A2 平移；仅
    /// isVariedLen 时启用——R11 实测变长净收益 -2.7%、定长净回归 +2.0%，定长留在
    /// 串行+R10 路径）。R13 起模板化：chunk64（tile M=64，分区 48KB/实例）与
    /// chunk128（tile M=128，分区 64KB/实例，tile 与串行 V128 生产路径同款）共用
    /// 同一相位编排，仅 L1 分区图与 tile 类型不同。
    /// [资源策略] L1 空间分区 + L0 时分复用（与 A5 同构）：
    ///   - L1 各实例独立区与 event id（GM→L1 装载与上一 Cube 计算重叠的前提，
    ///     消除跨实例 flag credit 串扰）：QK [0,PART)/id{0,1}、QH [PART,2*PART)/id{2,3}、
    ///     AV [2*PART,3*PART)/id{4,5}——各 L1A/L1B 单槽（K<=128 单 k-block
    ///     无需 k 向预取双槽；构造器 event 偏移为绝对 id 基）。
    ///   - L0A/L0B/L0C 不分区：三实例窗口互斥由相位结构保证（waitL0Drained/
    ///     finalWaitFlags + Cube2→Cube3 双屏障），L0C 跨实例交接与生产串行路径
    ///     同机制（unitFlag）。chunk64 下 L0C tile 64x128x4B=32KB、chunk128 下
    ///     128x128x4B=64KB（与串行生产路径同容量），均 << 128KB 且窗口不重叠。
    /// [每迭代相位]（task 滞后一拍与串行路径一致；跨核协议与 AIV 侧零改动）：
    ///   1a Cube1 operator()（当前 task 装载+计算+写出）+ waitL1Drained
    ///   1b Cube2 copyGmToL1(Q,H)——与 Cube1 的 L0 Mmad 重叠（MTE2‖M 管线）
    ///   1c Cube1 waitL0Drained + cube1Done（PIPE_FIX 序，Vec1 开工门）
    ///   2a 等 vec2Done[prev]（HWork/VWork 槽复用保护，写前等 Vec2 释放上一代）
    ///   2b Cube3 copyGmToL1BOnly(V)——与 Cube2 计算重叠
    ///   2c Cube2 preSetL0Flags + executeCompute（写 HWork）+ finalWaitFlags
    ///   2d MTE2/FIX 双屏障（L0C 排空，与串行路径 Cube2→Cube3 同位同序）
    ///   2e 等 vec1Done[prev] + Cube3 copyGmToL1AOnly(AttnMask)——vec1Done 比
    ///      串行路径更晚等：Vec1 慢不再阻塞 Cube2（额外重叠收益）
    ///   2f Cube3 preSetL0Flags + executeCompute（写 VWork）+ finalWaitFlags
    ///   2g cube3Done（PIPE_FIX 序）
    /// 数值路径与串行版本一致（同 tile 形状内相同 Mmad/Fixpipe 序列，精度红线
    /// 合规）；R10 的 Q 复用由 1b 装载隐藏于 Cube1 Mmad 窗口自然替代。
    template <typename BlockQKT, typename BlockQHT, typename BlockAVT, uint32_t PART_BYTES>
    __aicore__ inline void ProcessOverlapVarlen(uint32_t coreIdx, uint32_t coreNum) {
        // 分区预算：每实例 L1A tile + L1B tile 必须装进 PART_BYTES（chunk64 48KB/
        // chunk128 64KB；×3 实例 <= L1 512KB）
        static_assert(BlockQHT::L1A_TILE_SIZE + BlockQHT::L1B_TILE_SIZE <= PART_BYTES,
            "R11/R13: per-instance L1 partition (L1A + L1B tiles) exceeds PART_BYTES");

        BlockQKT blockMmadQKOvl(resource);
        BlockQHT blockMmadQHOvl(resource, PART_BYTES, 2, 3);
        BlockAVT blockMmadAttenVNEWOvl(resource, PART_BYTES * 2, 4, 5);

        auto qLayout = tla::MakeLayout<ElementQ, LayoutQ>(shapeBatch * kNumHead * seqlen, kHeadDim);
        auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * seqlen);
        auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * seqlen * kHeadDim, vHeadDim);
        auto ointerLayout = tla::MakeLayout<ElementOinter, LayoutOinter>(coreNum * chunkSize * PING_PONG_STAGES, cubeBlockScheduler.vBlockSize);
        auto vnewLayout = tla::MakeLayout<ElementVNEW, LayoutVNEW>(shapeBatch * vNumHead * seqlen, vHeadDim);

        bool needRun = false;

        while (cubeBlockScheduler.isRunning) {
            cubeBlockScheduler.InitTask();

            // === Phase 1a: Cube1 当前 task（装载+计算+Fixpipe 写出完整流）===
            if (cubeBlockScheduler.isRunning && coreIdx < coreNum) {
                uint32_t streamId = cubeBlockScheduler.GetCurStageId();
                GDNFwdOOffsets& cube1Offsets = cubeBlockScheduler.GetCube1Offsets();
                int64_t cube1OffsetQ = cube1Offsets.qkOffset;
                int64_t cube1OffsetK = cube1Offsets.qkOffset;
                int64_t cube1OffsetAttn = cube1Offsets.attnWorkOffset;
                auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(coreNum * chunkSize * PING_PONG_STAGES, cube1Offsets.blockTokens);
                auto tensorQ = tla::MakeTensor(gmQ[cube1OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                auto tensorK = tla::MakeTensor(gmK[cube1OffsetK], kLayout, Catlass::Arch::PositionGM{});
                auto tensorAttn = tla::MakeTensor(gmAttnWorkspace[cube1OffsetAttn], attenLayout, Catlass::Arch::PositionGM{});
                GemmCoord cube1Shape{cube1Offsets.blockTokens, cube1Offsets.blockTokens, kHeadDim};
                auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                auto tensorBlockAttn = GetTile(tensorAttn, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));
                blockMmadQKOvl.preSetFlags();
                blockMmadQKOvl(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);
                blockMmadQKOvl.waitL1Drained();
            }

            // === Phase 1b: Cube2 GM→L1（Q/H 纯 GM 输入无跨核依赖），与 Cube1
            //     尚在飞行的 L0 Mmad 重叠 ===
            if (needRun && coreIdx < coreNum) {
                GDNFwdOOffsets& cube2Offsets = cubeBlockScheduler.GetCube23Offsets();
                int64_t cube2OffsetQ = cube2Offsets.qkOffset;
                int64_t cube2OffsetH = cube2Offsets.hOffset;
                auto tensorQ2 = tla::MakeTensor(gmQ[cube2OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                auto tensorH = tla::MakeTensor(gmH[cube2OffsetH], hLayout, Catlass::Arch::PositionGM{});
                GemmCoord cube2Shape{cube2Offsets.blockTokens, cube2Offsets.vBlockDim, kHeadDim};
                auto tensorBlockQ2 = GetTile(tensorQ2, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                blockMmadQHOvl.preSetL1Flags();
                blockMmadQHOvl.copyGmToL1(tensorBlockQ2, tensorBlockH, cube2Shape);
            }

            // === Phase 1c: Cube1 L0 排空 + cube1Done（FIX 管线序）===
            if (cubeBlockScheduler.isRunning && coreIdx < coreNum) {
                uint32_t streamId = cubeBlockScheduler.GetCurStageId();
                blockMmadQKOvl.waitL0Drained();
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done[streamId]);
            }

            // === Phase 2: 上一 task 的 Cube2/Cube3（重叠编排）===
            if (needRun && coreIdx < coreNum) {
                uint32_t streamId = cubeBlockScheduler.GetPrevStageId();
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);

                GDNFwdOOffsets& cube2Offsets = cubeBlockScheduler.GetCube23Offsets();
                GDNFwdOOffsets& cube3Offsets = cubeBlockScheduler.GetCube23Offsets();
                int64_t cube2OffsetHWork = cube2Offsets.hvWorkOffset;
                int64_t cube3OffsetAttnMask = cube3Offsets.attnWorkOffset;
                int64_t cube3OffsetV = cube3Offsets.ovOffset;
                int64_t cube3OffsetVWork = cube3Offsets.hvWorkOffset;
                GemmCoord cube2Shape{cube2Offsets.blockTokens, cube2Offsets.vBlockDim, kHeadDim};
                GemmCoord cube3Shape{cube3Offsets.blockTokens, cube3Offsets.vBlockDim, cube3Offsets.blockTokens};
                auto tensorHWork = tla::MakeTensor(gmHWorkspace[cube2OffsetHWork], ointerLayout, Catlass::Arch::PositionGM{});
                auto tensorBlockHWork = GetTile(tensorHWork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(coreNum * chunkSize * PING_PONG_STAGES, cube3Offsets.blockTokens);
                auto tensorAttnMask = tla::MakeTensor(gmAftermaskWorkspace[cube3OffsetAttnMask], attenLayout, Catlass::Arch::PositionGM{});
                auto tensorBlockAttnMask = GetTile(tensorAttnMask, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.k()));
                auto tensorV = tla::MakeTensor(gmV[cube3OffsetV], vnewLayout, Catlass::Arch::PositionGM{});
                auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.k(), cube3Shape.n()));
                auto tensorVWork = tla::MakeTensor(gmVWorkspace[cube3OffsetVWork], ointerLayout, Catlass::Arch::PositionGM{});
                auto tensorBlockVWork = GetTile(tensorVWork, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.n()));

                // 2b: Cube3 V 预载（纯 GM 输入，与 Cube2 计算重叠）
                blockMmadAttenVNEWOvl.preSetL1Flags();
                blockMmadAttenVNEWOvl.copyGmToL1BOnly(tensorBlockV, cube3Shape);

                // 2c: Cube2 计算（写 HWork；vec2Done 已等）
                blockMmadQHOvl.preSetL0Flags();
                blockMmadQHOvl.executeCompute(tensorBlockHWork, cube2Shape);
                blockMmadQHOvl.finalWaitFlags();

                // 2d: L0C 排空屏障（与串行路径 Cube2→Cube3 同位同序）
                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::PipeBarrier<PIPE_FIX>();

                // 2e: Cube3 AttnMask 装载（依赖 Vec1 的 aftermask 写出）
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);
                blockMmadAttenVNEWOvl.copyGmToL1AOnly(tensorBlockAttnMask, cube3Shape);

                // 2f: Cube3 计算（写 VWork）
                blockMmadAttenVNEWOvl.preSetL0Flags();
                blockMmadAttenVNEWOvl.executeCompute(tensorBlockVWork, cube3Shape);
                blockMmadAttenVNEWOvl.finalWaitFlags();
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube3Done[streamId]);
            }
            needRun = true;
        }
        // 收尾与串行路径一致：AIV 启动时预置 vec2Done 双槽 credit，此处消耗
        Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
        Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);
    }
#endif

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
    /// [R12] 分组主循环（A5 调度粒度的 A2 平移）：task = (chunk, taskGroupSize
    /// 个连续 vHead)。组内每 head 的 Cube1/Cube2/Cube3/Vec1/Vec2 调用序列与
    /// 串行路径单 task 完全同构（复用同 BlockMmad 实例与槽轮转），仅粒度变粗：
    /// 每 head 的跨核 flag 由组内最后一个 head 的调用收尾置位（Cube1/Cube3 的
    /// SetFlag 与 Vec 的 SetFlag 只在组尾发）——每 task 的 8 次跨核握手按组征
    /// 收，G=4 时握手次数降为 1/4（骨架实测 655us 握手税的摊薄路径）。
    /// 相位时序与串行路径逐点一致：Cube1(当前组) 与 Cube2/3(上一组) 滞后一拍；
    /// Vec1 等 cube1Done 后逐 head 跑 epilogue，Vec2 等 cube3Done 后逐 head 跑
    /// output epilogue。
    /// [phase 2] 组内 Q 复用（Cube1/Cube2 各自 j-2 同槽同 kHead skip，交错序
    /// 保证相位不变量；R10 跨实例复用在分组下结构性不可能——组内 n>=2 次调用
    /// 消耗双槽，Cube1 当前组装载在 Cube2 读取前必然被逐出）：Q 装载量与
    /// 串行+R10 持平（每组 4 次），K/H 每组 4+4 次不变，净变化 = 握手 4×削减。
    __aicore__ inline void ProcessGroup(uint32_t coreIdx, uint32_t coreNum) {
        BlockMmadQK blockMmadQK(resource);
        BlockMmadQH128 blockMmadQH128(resource);
        BlockMmadQH256 blockMmadQH256(resource);
        BlockMmadAttenVNEW128 blockMmadAttenVNEW128(resource);
        BlockMmadAttenVNEW256 blockMmadAttenVNEW256(resource);

        auto qLayout = tla::MakeLayout<ElementQ, LayoutQ>(shapeBatch * kNumHead * seqlen, kHeadDim);
        auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * seqlen);
        auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * seqlen * kHeadDim, vHeadDim);
        auto ointerLayout = tla::MakeLayout<ElementOinter, LayoutOinter>(
            coreNum * chunkSize * PING_PONG_STAGES * groupCubeScheduler.taskGroupSize,
            cubeBlockScheduler.vBlockSize);
        auto vnewLayout = tla::MakeLayout<ElementVNEW, LayoutVNEW>(shapeBatch * vNumHead * seqlen, vHeadDim);

        bool needRun = false;

        // [R12 phase 2] Q 复用槽跟踪：slotQk[s] = L1A 物理槽 s 当前持有的 Q 的
        // qkOffset（-1 未知）。Cube1/Cube2 实例共享 L1A 区（同为 l1BufAddrStart=0），
        // 但 Cube2 滞后一组、组内双槽被 Cube1 当前组消耗——跨实例复用结构性不可
        // 行，实际命中来源是各实例自身 j-2 调用的同槽残留（交错序保证同 kHead）。
        // qReuseOk：kHeadDim<=128 单 k-block 相位前提。
        const bool qReuseOk = (kHeadDim <= 128);
        int64_t slotQk[2] = {-1, -1};
        uint32_t cube1QkCalls = 0;
        uint32_t cube2QhCalls = 0;

        while (groupCubeScheduler.isRunning) {
            groupCubeScheduler.InitTask();

            // === Phase 1: Cube1 当前组逐 head（装载+计算+写出完整流）===
            // [R12 phase 2] 组内 Q 复用：head 顺序取"偶先后奇"交错（n=4 时
            // 0,2,1,3——kHead 序列 0,1,0,1 或全同），使 L1A 双槽 ping-pong 下
            // 第 j 与 j-2 次调用恒同槽同 kHead。Cube1 对 GM 源与本槽上次装载
            // 一致的 head 调 SkipNextADataCopy 跳过 GM→L1A（R10 同模式纯增
            // 益，未命中退化正常装载）；kHeadDim<=128 单 k-block 保证每次调用
            // 恰好推进一个 L1A 槽（相位不变量前提）。Cube3 的 AttnMask 装载
            // 会覆写共享 L1A 区，故每迭代开头失效重置槽跟踪。
            if (groupCubeScheduler.isRunning && coreIdx < coreNum) {
                uint32_t streamId = groupCubeScheduler.GetCurStageId();
                const uint32_t headCount = groupCubeScheduler.GetGroupTaskCount(streamId);
                const uint32_t halfCount = (headCount + 1) / 2;
                slotQk[0] = -1;
                slotQk[1] = -1;
                for (uint32_t j = 0; j < headCount; ++j) {
                    const uint32_t headIdx = (j < halfCount) ? 2 * j : 2 * (j - halfCount) + 1;
                    Catlass::Gemm::Block::GDNFwdOOffsets headOffsets;
                    groupCubeScheduler.GetHeadOffsets(streamId, headIdx, headOffsets);
                    int64_t cube1OffsetQ = headOffsets.qkOffset;
                    int64_t cube1OffsetK = headOffsets.qkOffset;
                    int64_t cube1OffsetAttn = headOffsets.attnWorkOffset;
                    auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(
                        coreNum * chunkSize * PING_PONG_STAGES * groupCubeScheduler.taskGroupSize,
                        headOffsets.blockTokens);
                    auto tensorQ = tla::MakeTensor(gmQ[cube1OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                    auto tensorK = tla::MakeTensor(gmK[cube1OffsetK], kLayout, Catlass::Arch::PositionGM{});
                    auto tensorAttn =
                        tla::MakeTensor(gmAttnWorkspace[cube1OffsetAttn], attenLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube1Shape{headOffsets.blockTokens, headOffsets.blockTokens, kHeadDim};
                    auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                    auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                    auto tensorBlockAttn = GetTile(tensorAttn, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));
                    const uint32_t qkSlot = cube1QkCalls & 1;
                    if (qReuseOk && slotQk[qkSlot] == cube1OffsetQ) {
                        blockMmadQK.SkipNextADataCopy();
                    } else {
                        slotQk[qkSlot] = cube1OffsetQ;
                    }
                    ++cube1QkCalls;
                    blockMmadQK.preSetFlags();
                    blockMmadQK(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);
                    blockMmadQK.finalWaitFlags();
                }
                // 组内最后一个 head 的 FIX 序后才置 cube1Done（Vec1 开工门）
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(groupCubeScheduler.cube1Done[streamId]);
            }

            // === Phase 2: 上一组的 Cube2/Cube3 逐 head ===
            if (needRun && coreIdx < coreNum) {
                uint32_t streamId = groupCubeScheduler.GetPrevStageId();
                const uint32_t headCount = groupCubeScheduler.GetGroupTaskCount(streamId);
                // 组级 flag 等待（每组恰好一次，与 AIV 侧组级 set 1:1 配对——跨核
                // flag wait 消耗 credit，head 循环内重复 wait 会挂死）：vec1Done
                // 保护 aftermask 子槽组（Vec1 全组完成后 Cube3 才读），vec2Done
                // 保护 H/V workspace 子槽组（Vec2 释放上一代后 Cube2 才写）
                Arch::CrossCoreWaitFlag(groupCubeScheduler.vec1Done[streamId]);
                Arch::CrossCoreWaitFlag(groupCubeScheduler.vec2Done[streamId]);
                const uint32_t halfCount2 = (headCount + 1) / 2;
                for (uint32_t j = 0; j < headCount; ++j) {
                    // [R12 phase 2] 与 Cube1 同"偶先后奇"交错序（同 j 同物理槽
                    // 相位）。Cube2 滞后一组处理，槽内是 Cube1 当前组的 Q（异组
                    // 异 kHead）——j=0/1 比对不命中正常装载；j>=2 槽内为自身
                    // j-2 装载（交错序下同 kHead），比对命中 skip。纯运行时比对，
                    // 未命中零风险退化。
                    const uint32_t headIdx = (j < halfCount2) ? 2 * j : 2 * (j - halfCount2) + 1;
                    Catlass::Gemm::Block::GDNFwdOOffsets headOffsets;
                    groupCubeScheduler.GetHeadOffsets(streamId, headIdx, headOffsets);
                    int64_t cube2OffsetQ = headOffsets.qkOffset;
                    int64_t cube2OffsetH = headOffsets.hOffset;
                    int64_t cube2OffsetHWork = headOffsets.hvWorkOffset;
                    auto tensorQ = tla::MakeTensor(gmQ[cube2OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                    auto tensorH = tla::MakeTensor(gmH[cube2OffsetH], hLayout, Catlass::Arch::PositionGM{});
                    auto tensorHWork =
                        tla::MakeTensor(gmHWorkspace[cube2OffsetHWork], ointerLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube2Shape{headOffsets.blockTokens, headOffsets.vBlockDim, kHeadDim};
                    auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                    auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                    auto tensorBlockHWork =
                        GetTile(tensorHWork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                    const uint32_t qhSlot = cube2QhCalls & 1;
                    if (qReuseOk && slotQk[qhSlot] == cube2OffsetQ) {
                        if (headOffsets.vBlockDim <= 128) {
                            blockMmadQH128.SkipNextADataCopy();
                        } else {
                            blockMmadQH256.SkipNextADataCopy();
                        }
                    } else {
                        slotQk[qhSlot] = cube2OffsetQ;
                    }
                    ++cube2QhCalls;
                    if (headOffsets.vBlockDim <= 128) {
                        blockMmadQH128.preSetFlags();
                        blockMmadQH128(tensorBlockQ, tensorBlockH, tensorBlockHWork, cube2Shape);
                        blockMmadQH128.finalWaitFlags();
                    } else {
                        blockMmadQH256.preSetFlags();
                        blockMmadQH256(tensorBlockQ, tensorBlockH, tensorBlockHWork, cube2Shape);
                        blockMmadQH256.finalWaitFlags();
                    }
                }

                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::PipeBarrier<PIPE_FIX>();

                for (uint32_t headIdx = 0; headIdx < headCount; ++headIdx) {
                    Catlass::Gemm::Block::GDNFwdOOffsets headOffsets;
                    groupCubeScheduler.GetHeadOffsets(streamId, headIdx, headOffsets);
                    int64_t cube3OffsetAttnMask = headOffsets.attnWorkOffset;
                    int64_t cube3OffsetV = headOffsets.ovOffset;
                    int64_t cube3OffsetVWork = headOffsets.hvWorkOffset;
                    auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(
                        coreNum * chunkSize * PING_PONG_STAGES * groupCubeScheduler.taskGroupSize,
                        headOffsets.blockTokens);
                    auto tensorAttnMask = tla::MakeTensor(
                        gmAftermaskWorkspace[cube3OffsetAttnMask], attenLayout, Catlass::Arch::PositionGM{});
                    auto tensorV = tla::MakeTensor(gmV[cube3OffsetV], vnewLayout, Catlass::Arch::PositionGM{});
                    auto tensorVWork =
                        tla::MakeTensor(gmVWorkspace[cube3OffsetVWork], ointerLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube3Shape{headOffsets.blockTokens, headOffsets.vBlockDim, headOffsets.blockTokens};
                    auto tensorBlockAttnMask =
                        GetTile(tensorAttnMask, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.k()));
                    auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.k(), cube3Shape.n()));
                    auto tensorBlockVWork =
                        GetTile(tensorVWork, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.n()));
                    if (headOffsets.vBlockDim <= 128) {
                        blockMmadAttenVNEW128.preSetFlags();
                        blockMmadAttenVNEW128(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);
                        blockMmadAttenVNEW128.finalWaitFlags();
                    } else {
                        blockMmadAttenVNEW256.preSetFlags();
                        blockMmadAttenVNEW256(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);
                        blockMmadAttenVNEW256.finalWaitFlags();
                    }
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(groupCubeScheduler.cube3Done[streamId]);
            }
            needRun = true;
        }
        // 收尾与串行路径一致：AIV 启动时预置 vec2Done 双槽 credit，此处消耗
        Arch::CrossCoreWaitFlag(groupCubeScheduler.vec2Done[0]);
        Arch::CrossCoreWaitFlag(groupCubeScheduler.vec2Done[1]);
    }

    /// [R12] AIV 分组主循环：Vec1 等 cube1Done 后逐 head 跑 qkmask epilogue，
    /// Vec2 等 cube3Done 后逐 head 跑 output epilogue；跨核 flag 按组置位
    ///（vec1Done/vec2Done 只在组内最后一个 head 完成后发——与 AIC 侧按组
    /// 等待配对，握手按组征收）。
    __aicore__ inline void ProcessGroupVec() {
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(groupVecScheduler.vec2Done[0]);
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(groupVecScheduler.vec2Done[1]);

        AscendC::LocalTensor<float> maskUbTensor = resource.ubBuf.template GetBufferByByte<float>(0);
        AscendC::Duplicate<float>(maskUbTensor, (float)0.0, 64*64);
        AscendC::PipeBarrier<PIPE_V>();
        for(uint32_t i = 0; i < 64; ++ i) AscendC::Duplicate<float>(maskUbTensor[i * 64], (float)1.0, i + 1);
        AscendC::PipeBarrier<PIPE_V>();

        bool needRun = false;
        uint32_t pingpongFlag = 0;

        while (groupVecScheduler.isRunning) {
            groupVecScheduler.InitTask();

            if (groupVecScheduler.isRunning && coreIdx < coreNum * subBlockNum) {
                uint32_t streamId = groupVecScheduler.GetCurStageId();
                Arch::CrossCoreWaitFlag(groupVecScheduler.cube1Done[streamId]);
                const uint32_t headCount = groupVecScheduler.GetGroupTaskCount(streamId);
                for (uint32_t headIdx = 0; headIdx < headCount; ++headIdx) {
                    Catlass::Gemm::Block::GDNFwdOOffsets headOffsets;
                    groupVecScheduler.GetHeadOffsets(streamId, headIdx, headOffsets);
                    int64_t vec1OffsetAttnMask = headOffsets.attnWorkOffset;
                    int64_t vec1OffsetG = headOffsets.gOffset;
                    int64_t vec1OffsetAttn = headOffsets.attnWorkOffset;
                    EpilogueGDNFwdOQkmask epilogueGDNFwdOQkmask(resource);
                    epilogueGDNFwdOQkmask(
                        gmAftermaskWorkspace[vec1OffsetAttnMask],
                        gmG[vec1OffsetG], gmAttnWorkspace[vec1OffsetAttn], gmMask,
                        chunkSize, headOffsets.blockTokens, kHeadDim, vHeadDim, pingpongFlag,
                        headOffsets.batchIdx, headOffsets.headIdx, headOffsets.chunkIdx
                    );
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(groupVecScheduler.vec1Done[streamId]);
            }

            if (needRun && coreIdx < coreNum * subBlockNum) {
                uint32_t streamId = groupVecScheduler.GetPrevStageId();
                Arch::CrossCoreWaitFlag(groupVecScheduler.cube3Done[streamId]);
                const uint32_t headCount = groupVecScheduler.GetGroupTaskCount(streamId);
                for (uint32_t headIdx = 0; headIdx < headCount; ++headIdx) {
                    Catlass::Gemm::Block::GDNFwdOOffsets headOffsets;
                    groupVecScheduler.GetHeadOffsets(streamId, headIdx, headOffsets);
                    int64_t vec2OffsetO = headOffsets.ovOffset;
                    int64_t vec2OffsetG = headOffsets.gOffset;
                    int64_t vec2OffsetVWork = headOffsets.hvWorkOffset;
                    int64_t vec2OffsetHWork = headOffsets.hvWorkOffset;
                    EpilogueGDNFwdOOutput epilogueGDNFwdOOutput(resource);
                    epilogueGDNFwdOOutput(
                        gmO[vec2OffsetO],
                        gmG[vec2OffsetG], gmVWorkspace[vec2OffsetVWork], gmHWorkspace[vec2OffsetHWork],
                        scale, headOffsets.blockTokens, kHeadDim, headOffsets.vBlockDim, vHeadDim, pingpongFlag,
                        headOffsets.batchIdx, headOffsets.headIdx, headOffsets.chunkIdx
                    );
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(groupVecScheduler.vec2Done[streamId]);
            }
            needRun = true;
        }
    }
#endif

    __aicore__ inline void Process() {
        if ASCEND_IS_AIC {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            // [R11/R13] varlen 重叠路径：L1 空间分区后三 Cube 流水重叠（A5
            // 三段式的 A2 平移；L1 分区消除 R4 时代多实例 L0 窗口交错的 507015
            // 根源）。守卫条件全部运行时判定，不满足则走原串行路径（含 R10 Q
            // 复用），V256/定长功能零影响：
            //   chunkSize==64 或 128（R11 chunk64 / R13 chunk128，tile 与分区
            //   图按 chunk 尺寸分派）且 vHeadDim<=128（V256 的 N=256 超出本路径
            //   tile 形状）且 kHeadDim<=128（Pipelined 阶段函数仅单 k-block 合法）
            //   且 isVariedLen（R11 实测裁决：变长 -2.7% 净收益，定长 +2.0% 净回
            //   归——定长下 MTE2 已饱和，重叠掩盖不掉显式回归的 Q 装载，定长
            //   留在串行+R10 路径各取最优）。
            const bool useOverlapPath = (chunkSize == 64 || chunkSize == 128) && (vHeadDim <= 128) &&
                                         (kHeadDim <= 128) && (isVariedLen != 0);
            if (useOverlapPath) {
                if (chunkSize == 64) {
                    // [R11] chunk64：tile M=64，L1 分区 48KB/实例（L1A 16K + L1B 32K）
                    ProcessOverlapVarlen<BlockMmadQKOvl, BlockMmadQHOvl, BlockMmadAttenVNEWOvl, 48 * 1024>(
                        coreIdx, coreNum);
                } else {
                    // [R13] chunk128：tile M=128 与串行 V128 生产路径同款，L1 分区
                    // 64KB/实例（L1A 32K + L1B 32K），相位编排与 chunk64 完全一致
                    ProcessOverlapVarlen<BlockMmadQKOvl128, BlockMmadQHOvl128, BlockMmadAttenVNEWOvl128,
                        64 * 1024>(coreIdx, coreNum);
                }
                return;
            }
            // [R12] 分组路径（未进重叠路径时的分组分派；重叠路径自身不分组——
            // 两结构都是大改，组合留后续评估）：定长 GVA 走分组摊薄握手税（骨架
            // 655us 主战场），未分组化场景（MHA 1:1/V256/不整除）回串行路径。
            if (groupedMode) {
                ProcessGroup(coreIdx, coreNum);
                return;
            }
#endif

            BlockMmadQK blockMmadQK(resource);
            BlockMmadQH128 blockMmadQH128(resource);
            BlockMmadQH256 blockMmadQH256(resource);
            BlockMmadAttenVNEW128 blockMmadAttenVNEW128(resource);
            BlockMmadAttenVNEW256 blockMmadAttenVNEW256(resource);

            auto qLayout = tla::MakeLayout<ElementQ, LayoutQ>(shapeBatch * kNumHead * seqlen, kHeadDim);
            auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * seqlen);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * seqlen * kHeadDim, vHeadDim);
            auto ointerLayout = tla::MakeLayout<ElementOinter, LayoutOinter>(coreNum * chunkSize * PING_PONG_STAGES, cubeBlockScheduler.vBlockSize);
            auto vnewLayout = tla::MakeLayout<ElementVNEW, LayoutVNEW>(shapeBatch * vNumHead * seqlen, vHeadDim);

            bool needRun = false;

            // [R10] Q L1 常驻复用：Cube2 处理的 task 正是 Cube1 上一迭代同 stage 槽
            // 装载的 task（流水线滞后一拍），其 Q tile 仍在 L1A 对应槽内——三 Cube
            // 实例串行轮转保证中间无覆盖（Cube3 在 Cube2 finalWaitFlags 排空后才写
            // 同槽；QK 调用 #(i-1) 与 QH 调用 #(i-1) 槽相位一致）。命中即跳过 Cube2
            // 的 GM→L1A 重复搬运：SkipNextADataCopy 保留槽级 Wait/Set flag 协议，
            // 仅省 DataCopy 本身，数值路径零变化。按 (qkOffset, blockTokens) 运行时
            // 比较判定（同源即同字节，GQA 相邻 head 共享 Q 也是真命中）；未命中退化
            // 为正常装载，纯增益无正确性风险。
            int64_t cube1QOffsetSlot[PING_PONG_STAGES] = {-1, -1};
            uint32_t cube1QRowsSlot[PING_PONG_STAGES] = {0, 0};

            while (cubeBlockScheduler.isRunning) {
                cubeBlockScheduler.InitTask();

                if (cubeBlockScheduler.isRunning && coreIdx < coreNum) {
                    uint32_t streamId = cubeBlockScheduler.GetCurStageId();

                    GDNFwdOOffsets& cube1Offsets = cubeBlockScheduler.GetCube1Offsets();
                    int64_t cube1OffsetQ = cube1Offsets.qkOffset;
                    int64_t cube1OffsetK = cube1Offsets.qkOffset;
                    int64_t cube1OffsetAttn = cube1Offsets.attnWorkOffset;
                    auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(coreNum * chunkSize * PING_PONG_STAGES, cube1Offsets.blockTokens);
                    auto tensorQ = tla::MakeTensor(gmQ[cube1OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                    auto tensorK = tla::MakeTensor(gmK[cube1OffsetK], kLayout, Catlass::Arch::PositionGM{});
                    auto tensorAttn = tla::MakeTensor(gmAttnWorkspace[cube1OffsetAttn], attenLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube1Shape{cube1Offsets.blockTokens, cube1Offsets.blockTokens, kHeadDim};
                    auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                    auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                    auto tensorBlockAttn = GetTile(tensorAttn, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));
                    // [R10] 记录本槽 Q tile 源，供下一迭代 Cube2 复用判定
                    cube1QOffsetSlot[streamId] = cube1OffsetQ;
                    cube1QRowsSlot[streamId] = cube1Offsets.blockTokens;
                    blockMmadQK.preSetFlags();
                    blockMmadQK(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);
                    blockMmadQK.finalWaitFlags();
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done[streamId]);

                }
                // AscendC::PipeBarrier<PIPE_ALL>();

                if (needRun && coreIdx < coreNum) {
                    uint32_t streamId = cubeBlockScheduler.GetPrevStageId();
                    Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);
                    // vec2Done protects the H/V workspace consumed by Cube2/3; Cube1 uses a separate slot.
                    Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);
                    GDNFwdOOffsets& cube2Offsets = cubeBlockScheduler.GetCube23Offsets();
                    int64_t cube2OffsetQ = cube2Offsets.qkOffset;
                    int64_t cube2OffsetH = cube2Offsets.hOffset;
                    int64_t cube2OffsetHWork = cube2Offsets.hvWorkOffset;
                    auto tensorQ = tla::MakeTensor(gmQ[cube2OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                    auto tensorH = tla::MakeTensor(gmH[cube2OffsetH], hLayout, Catlass::Arch::PositionGM{});
                    auto tensorHWork = tla::MakeTensor(gmHWorkspace[cube2OffsetHWork], ointerLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube2Shape{cube2Offsets.blockTokens, cube2Offsets.vBlockDim, kHeadDim};
                    auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                    auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                    auto tensorBlockHWork = GetTile(tensorHWork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                    // [R10] Q 常驻复用：kHeadDim<=128（L1_TILE_K）保证单 k-block，
                    // 槽轮转对齐推导成立；>128 时多 k-block 槽推进不同步，禁用。
                    if (kHeadDim <= 128 && cube2Offsets.qkOffset == cube1QOffsetSlot[streamId] &&
                        cube2Offsets.blockTokens == cube1QRowsSlot[streamId]) {
                        if (cube2Offsets.vBlockDim <= 128) {
                            blockMmadQH128.SkipNextADataCopy();
                        } else {
                            blockMmadQH256.SkipNextADataCopy();
                        }
                    }
                    if (cube2Offsets.vBlockDim <= 128) {
                        blockMmadQH128.preSetFlags();
                        blockMmadQH128(tensorBlockQ, tensorBlockH, tensorBlockHWork, cube2Shape);
                        blockMmadQH128.finalWaitFlags();
                    } else {
                        blockMmadQH256.preSetFlags();
                        blockMmadQH256(tensorBlockQ, tensorBlockH, tensorBlockHWork, cube2Shape);
                        blockMmadQH256.finalWaitFlags();
                    }
                }

                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::PipeBarrier<PIPE_FIX>();

                if (needRun && coreIdx < coreNum) {
                    uint32_t streamId = cubeBlockScheduler.GetPrevStageId();
                    GDNFwdOOffsets& cube3Offsets = cubeBlockScheduler.GetCube23Offsets();
                    int64_t cube3OffsetAttnMask = cube3Offsets.attnWorkOffset;
                    int64_t cube3OffsetV = cube3Offsets.ovOffset;
                    int64_t cube3OffsetVWork = cube3Offsets.hvWorkOffset;
                    auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(coreNum * chunkSize * PING_PONG_STAGES, cube3Offsets.blockTokens);
                    auto tensorAttnMask = tla::MakeTensor(gmAftermaskWorkspace[cube3OffsetAttnMask], attenLayout, Catlass::Arch::PositionGM{});
                    auto tensorV = tla::MakeTensor(gmV[cube3OffsetV], vnewLayout, Catlass::Arch::PositionGM{});
                    auto tensorVWork = tla::MakeTensor(gmVWorkspace[cube3OffsetVWork], ointerLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube3Shape{cube3Offsets.blockTokens, cube3Offsets.vBlockDim, cube3Offsets.blockTokens};
                    auto tensorBlockAttnMask = GetTile(tensorAttnMask, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.k()));
                    auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.k(), cube3Shape.n()));
                    auto tensorBlockVWork = GetTile(tensorVWork, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.n()));
                    if (cube3Offsets.vBlockDim <= 128) {
                        blockMmadAttenVNEW128.preSetFlags();
                        blockMmadAttenVNEW128(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);
                        blockMmadAttenVNEW128.finalWaitFlags();
                    } else {
                        blockMmadAttenVNEW256.preSetFlags();
                        blockMmadAttenVNEW256(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);
                        blockMmadAttenVNEW256.finalWaitFlags();
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube3Done[streamId]);
                }
                needRun = true;
                // AscendC::PipeBarrier<PIPE_ALL>();
            }
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);
        }

        if ASCEND_IS_AIV {

            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            uint32_t subBlockNum = AscendC::GetSubBlockNum();

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            // [R12] AIV 分组分派：与 AIC 侧 groupedMode 同源判定（Init 内同式
            // 计算），未重叠路径的分组调用走分组 Vec 主循环。重叠路径 AIV 侧
            // 与串行路径相同（cube1Done→Vec1→vec1Done / cube3Done→Vec2→
            // vec2Done 协议不变），无需分派。
            if (groupedMode && !((chunkSize == 64 || chunkSize == 128) && (vHeadDim <= 128) &&
                                 (kHeadDim <= 128) && (isVariedLen != 0))) {
                ProcessGroupVec();
                return;
            }
#endif

            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[0]);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[1]);

            AscendC::LocalTensor<float> maskUbTensor = resource.ubBuf.template GetBufferByByte<float>(0);
            AscendC::Duplicate<float>(maskUbTensor, (float)0.0, 64*64);
            AscendC::PipeBarrier<PIPE_V>();
            for(uint32_t i = 0; i < 64; ++ i) AscendC::Duplicate<float>(maskUbTensor[i * 64], (float)1.0, i + 1);
            AscendC::PipeBarrier<PIPE_V>();

            bool needRun = false;
            uint32_t pingpongFlag = 0;

            while (vecBlockScheduler.isRunning) {
                vecBlockScheduler.InitTask();

                if (vecBlockScheduler.isRunning && coreIdx < coreNum * subBlockNum) {
                    uint32_t streamId = vecBlockScheduler.GetCurStageId();
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube1Done[streamId]);
                    GDNFwdOOffsets& vec1Offsets = vecBlockScheduler.GetVec1Offsets();
                    int64_t vec1OffsetAttnMask = vec1Offsets.attnWorkOffset;
                    int64_t vec1OffsetG = vec1Offsets.gOffset;
                    int64_t vec1OffsetAttn = vec1Offsets.attnWorkOffset;
                    EpilogueGDNFwdOQkmask epilogueGDNFwdOQkmask(resource);
                    epilogueGDNFwdOQkmask(
                        gmAftermaskWorkspace[vec1OffsetAttnMask],
                        gmG[vec1OffsetG], gmAttnWorkspace[vec1OffsetAttn], gmMask,
                        chunkSize, vec1Offsets.blockTokens, kHeadDim, vHeadDim, pingpongFlag, vec1Offsets.batchIdx, vec1Offsets.headIdx, vec1Offsets.chunkIdx
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done[streamId]);
                }

                // AscendC::PipeBarrier<PIPE_ALL>();

                if (needRun && coreIdx < coreNum * subBlockNum) {
                    uint32_t streamId = vecBlockScheduler.GetPrevStageId();
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube3Done[streamId]);
                    GDNFwdOOffsets& vec2Offsets = vecBlockScheduler.GetVec2Offsets();
                    int64_t vec2OffsetO = vec2Offsets.ovOffset;
                    int64_t vec2OffsetG = vec2Offsets.gOffset;
                    int64_t vec2OffsetVWork = vec2Offsets.hvWorkOffset;
                    int64_t vec2OffsetHWork = vec2Offsets.hvWorkOffset;
                    EpilogueGDNFwdOOutput epilogueGDNFwdOOutput(resource);
                    epilogueGDNFwdOOutput(
                        gmO[vec2OffsetO],
                        gmG[vec2OffsetG], gmVWorkspace[vec2OffsetVWork], gmHWorkspace[vec2OffsetHWork],
                        scale, vec2Offsets.blockTokens, kHeadDim, vec2Offsets.vBlockDim, vHeadDim, pingpongFlag, vec2Offsets.batchIdx, vec2Offsets.headIdx, vec2Offsets.chunkIdx
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[streamId]);
                }
                needRun = true;
            }
        }
    }

};

}
