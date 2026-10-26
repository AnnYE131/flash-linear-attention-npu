/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "aclnn_chunk_gated_delta_rule_fwd.h"

#include "chunk_gated_delta_rule_fwd.h"
#include "../../op_kernel/internal/arch22/gdn_front_ho_shared_layout.h"
#include "../../../chunk_fwd_h/op_host/op_api/chunk_fwd_h.h"
#include "../../../chunk_fwd_o/op_host/op_api/chunk_fwd_o.h"
#include "../../../chunk_gated_delta_rule_fwd_prepare/op_host/op_api/chunk_gated_delta_rule_fwd_prepare.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include <cstring>
#include <initializer_list>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t HEAD_DIM_128 = 128;
constexpr int64_t CHUNK_GATED_DELTA_RULE_FWD_V256 = 256;
constexpr int64_t CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 = 64;
constexpr int64_t CHUNK_GATED_DELTA_RULE_FWD_CHUNK_128 = 128;
// Stage P encodes the GM row gap in DataCopyExtParams as uint32 bytes.
// This is a transport-field bound, so larger Hv stays on the BHT route.
constexpr int64_t CUMSUM_PREPARE_MAX_HEADS =
    static_cast<int64_t>(0xffffffffULL / sizeof(float) + 1);

struct ChunkGatedDeltaRuleFwdParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *aLogOptional = nullptr;
    const aclTensor *dtBiasOptional = nullptr;
    const aclTensor *initialStateOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    const char *layout = nullptr;
    double scale = 1.0;
    int64_t chunkSize = CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64;
    bool useExp2 = false;
    bool useQkL2norm = false;
    bool allowNegEigval = false;
    bool stateVFirst = false;
    const aclTensor *oOut = nullptr;
    const aclTensor *finalStateOutOptional = nullptr;
    const aclTensor *qHatOutOptional = nullptr;
    const aclTensor *kHatOutOptional = nullptr;
    const aclTensor *qRstdOutOptional = nullptr;
    const aclTensor *kRstdOutOptional = nullptr;
    const aclTensor *betaEffOutOptional = nullptr;
    const aclTensor *gCumsumOutOptional = nullptr;
    const aclTensor *aOutOptional = nullptr;
    const aclTensor *hOutOptional = nullptr;
};

enum class GdnLayout { BNSD, BSND, NTD, TND };

struct GdnShapeInfo {
    GdnLayout layout = GdnLayout::BNSD;
    bool isSequenceMajor = false;
    int64_t batch = 0;
    int64_t hq = 0;
    int64_t hv = 0;
    int64_t seqlen = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
};

static bool IsAscend950()
{
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    using Ops::Transformer::AclnnUtil::IsRegbase;
    return IsRegbase(npuArch);
}

static bool IsDav2201CandidateSoc();

static bool IsA2FeatureRequest(const ChunkGatedDeltaRuleFwdParams &params)
{
    return GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 &&
        (params.useQkL2norm || params.useExp2 || params.aLogOptional != nullptr ||
         params.dtBiasOptional != nullptr || params.betaEffOutOptional != nullptr || params.allowNegEigval ||
         params.stateVFirst || params.hOutOptional != nullptr);
}

static bool IsNativeQkvLayout(const ChunkGatedDeltaRuleFwdParams &params)
{
    return IsDav2201CandidateSoc() && params.layout != nullptr &&
           (std::strcmp(params.layout, "BSND") == 0 || std::strcmp(params.layout, "TND") == 0);
}

static bool UsePreparePath(const ChunkGatedDeltaRuleFwdParams &params)
{
    if (IsA2FeatureRequest(params)) return false; // A2 沿私有 Prepare/H/O 实现模式。
    if (IsAscend950()) {
        // On Ascend950 the supported new path is selected by its shape contract.
        // use_exp2 and layout are independent features of that path.
        if (params.q == nullptr || params.k == nullptr || params.v == nullptr || params.layout == nullptr ||
            params.q->GetViewShape().GetDimNum() != 4 || params.k->GetViewShape().GetDimNum() != 4 ||
            params.v->GetViewShape().GetDimNum() != 4) {
            return false;
        }
        const bool sequenceMajor = std::strcmp(params.layout, "BSND") == 0 ||
                                   std::strcmp(params.layout, "TND") == 0;
        const size_t headDim = sequenceMajor ? 2 : 1;
        const int64_t hq = params.q->GetViewShape().GetDim(headDim);
        const int64_t hv = params.v->GetViewShape().GetDim(headDim);
        return params.q->GetDataType() == DataType::DT_BF16 &&
               params.k->GetDataType() == DataType::DT_BF16 && params.v->GetDataType() == DataType::DT_BF16 &&
               params.q->GetViewShape().GetDim(3) == HEAD_DIM_128 &&
               params.v->GetViewShape().GetDim(3) == HEAD_DIM_128 &&
               params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 && hq > 0 && hv / hq <= 4;
    }
    const bool legacyLayout = std::strcmp(params.layout, "BNSD") == 0 ||
                              std::strcmp(params.layout, "NTD") == 0;
    // A2 reads token-major QKV directly; feature selection is otherwise unchanged.
    return params.useExp2 || params.useQkL2norm ||
           params.aLogOptional != nullptr || params.dtBiasOptional != nullptr ||
           params.betaEffOutOptional != nullptr || params.allowNegEigval ||
           params.hOutOptional != nullptr || params.stateVFirst ||
           (!legacyLayout && !IsNativeQkvLayout(params));
}

static op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

static const aclIntArray *MakePerm(std::initializer_list<int64_t> dims, aclOpExecutor *executor)
{
    return executor->AllocIntArray(dims.begin(), dims.size());
}

static const aclTensor *DenseQkvView(const aclTensor *tensor, aclOpExecutor *executor)
{
    // MakeContiguous has already materialized noncontiguous inputs. This is metadata only.
    // Preserve a contiguous caller view's storage offset without mutating its descriptor.
    auto *view = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetViewOffset());
    if (view != nullptr) {
        view->SetStorageShape(tensor->GetViewShape());
        view->SetOriginalShape(tensor->GetViewShape());
    }
    return view;
}

static bool CheckedShapeElements(const op::Shape &shape, uint64_t &elements)
{
    elements = 1;
    if (shape.GetDimNum() == 0) return false;
    for (size_t axis = 0; axis < shape.GetDimNum(); ++axis) {
        const int64_t dim = shape.GetDim(axis);
        if (dim <= 0 || elements > static_cast<uint64_t>(INT64_MAX) / static_cast<uint64_t>(dim))
            return false;
        elements *= static_cast<uint64_t>(dim);
    }
    return true;
}

static const aclTensor *NormParentView(const aclTensor *parent, const op::Shape &shape,
                                       uint64_t offsetBytes, aclOpExecutor *executor)
{
    if (parent == nullptr || executor == nullptr) return nullptr;
    const auto dtype = parent->GetDataType();
    if (dtype != DataType::DT_FLOAT && dtype != DataType::DT_FLOAT16 && dtype != DataType::DT_BF16)
        return nullptr;
    const uint64_t width = dtype == DataType::DT_FLOAT ? sizeof(float) : 2;
    uint64_t parentCount = 0, storageCount = 0, count = 0;
    const int64_t base = parent->GetViewOffset();
    if (!CheckedShapeElements(parent->GetViewShape(), parentCount) ||
        !CheckedShapeElements(parent->GetStorageShape(), storageCount) ||
        !CheckedShapeElements(shape, count) || base < 0 || offsetBytes % width ||
        static_cast<uint64_t>(base) > storageCount ||
        parentCount > storageCount - static_cast<uint64_t>(base)) return nullptr;
    const uint64_t offset = offsetBytes / width;
    if (offset > parentCount || count > parentCount - offset ||
        offset > static_cast<uint64_t>(INT64_MAX - base)) return nullptr;
    // CreateView继承parent的dtype/存储所有者；只缩view，不缩实际生产输出的容量。
    auto *view = executor->CreateView(parent, shape, base + static_cast<int64_t>(offset));
    if (view != nullptr) {
        view->SetStorageShape(shape);
        view->SetOriginalShape(shape);
    }
    return view;
}

static const aclTensor *TransposeContiguous(const aclTensor *tensor, std::initializer_list<int64_t> dims,
                                            aclOpExecutor *executor)
{
    const aclIntArray *perm = MakePerm(dims, executor);
    if (perm == nullptr) {
        return nullptr;
    }
    const aclTensor *permuted = l0op::Transpose(tensor, perm, executor);
    if (permuted == nullptr) {
        return nullptr;
    }
    const aclTensor *materialized = l0op::Contiguous(permuted, executor);
    if (materialized == nullptr) {
        return nullptr;
    }

    // Contiguous materializes the storage, but the resulting tensor can still
    // carry the transpose view metadata. Re-declare the logical shape so the
    // following custom ops see a dense BHT/BHTC tensor instead of a stale view.
    const aclTensor *reshaped = l0op::Reshape(materialized, permuted->GetViewShape(), executor);
    if (reshaped == nullptr) {
        return nullptr;
    }
    reshaped->SetStorageShape(reshaped->GetViewShape());
    reshaped->SetOriginalShape(reshaped->GetViewShape());
    return reshaped;
}

static int64_t Dim(const aclTensor *tensor, size_t index)
{
    return tensor->GetViewShape().GetDim(index);
}

static size_t Rank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

static bool HasShape(const aclTensor *tensor, std::initializer_list<int64_t> dims)
{
    if (tensor == nullptr || Rank(tensor) != dims.size()) {
        return false;
    }
    size_t index = 0;
    for (int64_t dim : dims) {
        if (Dim(tensor, index++) != dim) {
            return false;
        }
    }
    return true;
}

static bool ShapeEqual(const op::Shape &lhs, const op::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t index = 0; index < lhs.GetDimNum(); ++index) {
        if (lhs.GetDim(index) != rhs.GetDim(index)) {
            return false;
        }
    }
    return true;
}

static int64_t SeqNum(const ChunkGatedDeltaRuleFwdParams &params, int64_t batch)
{
    return params.cuSeqlensOptional == nullptr
               ? batch
               : static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;
}

static int64_t ExpectedChunks(const ChunkGatedDeltaRuleFwdParams &params, int64_t seqlen)
{
    if (params.cuSeqlensOptional == nullptr) {
        return (seqlen + params.chunkSize - 1) / params.chunkSize;
    }

    int64_t total = 0;
    const aclIntArray &cu = *params.cuSeqlensOptional;
    for (size_t idx = 0; idx + 1 < cu.Size(); ++idx) {
        const int64_t length = cu[idx + 1] - cu[idx];
        total += (length + params.chunkSize - 1) / params.chunkSize;
    }
    return total;
}

static aclnnStatus CheckStateShape(const aclTensor *state, const char *name, int64_t seqNum, int64_t hv,
                                   int64_t kDim, int64_t vDim)
{
    if (state == nullptr) {
        return ACLNN_SUCCESS;
    }
    const auto shape = state->GetViewShape();
    CHECK_COND(shape.GetDimNum() == 4 && shape.GetDim(0) == seqNum && shape.GetDim(1) == hv &&
                   shape.GetDim(2) == kDim && shape.GetDim(3) == vDim,
               ACLNN_ERR_PARAM_INVALID, "%s must have shape [seqNum,Hv,K,V].", name);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckMetadata(const ChunkGatedDeltaRuleFwdParams &params, int64_t seqlen)
{
    if (params.cuSeqlensOptional == nullptr) {
        return ACLNN_SUCCESS;
    }

    const aclIntArray &cu = *params.cuSeqlensOptional;
    CHECK_COND(cu.Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlens must contain at least [0,totalTokens].");
    CHECK_COND(cu[0] == 0 && cu[cu.Size() - 1] == seqlen, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlens must start at 0 and end at T.");
    const bool requirePositiveLengths =
        GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201;
    for (size_t idx = 0; idx + 1 < cu.Size(); ++idx) {
        CHECK_COND(cu[idx] <= cu[idx + 1], ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlens must be nondecreasing.");
        CHECK_COND(!requirePositiveLengths || cu[idx] < cu[idx + 1], ACLNN_ERR_PARAM_INVALID,
                   "DAV_2201 requires strictly increasing cuSeqlens.");
    }

    const aclIntArray &indices = *params.chunkIndicesOptional;
    const int64_t expectedChunks = ExpectedChunks(params, seqlen);
    CHECK_COND(indices.Size() % 2 == 0 && static_cast<int64_t>(indices.Size() / 2) == expectedChunks,
               ACLNN_ERR_PARAM_INVALID,
               "chunkIndices must contain one (seqId,localChunkId) pair per chunk.");
    size_t index = 0;
    for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
        const int64_t seqChunks = (cu[seq + 1] - cu[seq] + params.chunkSize - 1) / params.chunkSize;
        for (int64_t localChunk = 0; localChunk < seqChunks; ++localChunk) {
            CHECK_COND(indices[index] == static_cast<int64_t>(seq) && indices[index + 1] == localChunk,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunkIndices must use canonical sequence-major order.");
            index += 2;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static const aclTensor *MakeA2GateParameters(const aclTensor *aLog, const aclTensor *dtBias,
                                            int64_t hv, aclOpExecutor *executor)
{
    if (aLog == nullptr || hv <= 0 || static_cast<uint64_t>(hv) > INT64_MAX / sizeof(float))
        return nullptr;
    const bool hasBias = dtBias != nullptr;
    if (MakeContiguous(aLog, executor) != ACLNN_SUCCESS ||
        MakeContiguous(dtBias, executor) != ACLNN_SUCCESS) return nullptr;
    if (aLog->GetDataType() != DataType::DT_FLOAT)
        aLog = l0op::Cast(aLog, DataType::DT_FLOAT, executor);
    if (dtBias != nullptr && dtBias->GetDataType() != DataType::DT_FLOAT)
        dtBias = l0op::Cast(dtBias, DataType::DT_FLOAT, executor);
    if (aLog == nullptr || (hasBias && dtBias == nullptr)) return nullptr;
    if (dtBias == nullptr) return DenseQkvView(aLog, executor);
    // 两次正规 ViewCopy 生产 executor-owned 参数包，再作为只读 input 提交。
    // 子 view 共享该 parent 的存储所有者，不写调用方的 aLog/dtBias。
    const auto *parent = executor->AllocTensor(MakeShape({2, hv}), DataType::DT_FLOAT, Format::FORMAT_ND);
    const auto *aRow = NormParentView(parent, MakeShape({hv}), 0, executor);
    const auto *biasRow = NormParentView(parent, MakeShape({hv}), static_cast<uint64_t>(hv) * sizeof(float), executor);
    if (aRow == nullptr || biasRow == nullptr || l0op::ViewCopy(aLog, aRow, executor) == nullptr ||
        l0op::ViewCopy(dtBias, biasRow, executor) == nullptr) return nullptr;
    return parent;
}

static aclnnStatus CheckRequiredInputs(const ChunkGatedDeltaRuleFwdParams &params)
{
    CHECK_COND(params.q != nullptr && params.k != nullptr && params.v != nullptr && params.g != nullptr &&
                   params.beta != nullptr && params.oOut != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "q/k/v/g/beta/oOut must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckZeroShape(const ChunkGatedDeltaRuleFwdParams &params, uint64_t *workspaceSize)
{
    if (params.q->IsEmpty() || params.k->IsEmpty() || params.v->IsEmpty() || params.g->IsEmpty() ||
        params.beta->IsEmpty()) {
        *workspaceSize = 0UL;
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ViewCopyIfPresent(const aclTensor *src, const aclTensor *dst, aclOpExecutor *executor)
{
    if (dst == nullptr) {
        return ACLNN_SUCCESS;
    }
    CHECK_RET(src != nullptr && l0op::ViewCopy(src, dst, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

#define GDN_STAGE_CHECK(condition, code) \
    do {                                  \
        if (!(condition)) {               \
            return static_cast<aclnnStatus>(code); \
        }                                 \
    } while (false)

static aclnnStatus CheckRank(const aclTensor *tensor, size_t rank, const char *name)
{
    CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "%s must not be nullptr.", name);
    CHECK_COND(tensor->GetViewShape().GetDimNum() == rank, ACLNN_ERR_PARAM_INVALID,
               "%s must be rank %zu.", name, rank);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOptionalRank(const aclTensor *tensor, size_t rank, const char *name)
{
    return tensor == nullptr ? ACLNN_SUCCESS : CheckRank(tensor, rank, name);
}

static bool IsDav2201CandidateSoc()
{
    const char *socName = aclrtGetSocName();
    // A2 910B variants are the DAV_2201 deployment targets. Unknown or other
    // SoC names stay on the historical BHT route until tiling proves the
    // candidate architecture explicitly.
    return socName != nullptr &&
           (std::strncmp(socName, "Ascend910B", std::strlen("Ascend910B")) == 0 ||
            std::strcmp(socName, "ascend910b") == 0);
}

static bool UsePreparedCumsum(const ChunkGatedDeltaRuleFwdParams &params,
                              const GdnShapeInfo &info)
{
    return IsDav2201CandidateSoc() && !UsePreparePath(params) &&
           info.hv <= CUMSUM_PREPARE_MAX_HEADS &&
           info.kDim == HEAD_DIM_128 &&
           (info.vDim == HEAD_DIM_128 ||
            info.vDim == CHUNK_GATED_DELTA_RULE_FWD_V256) &&
           (params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 ||
            params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_128);
}

static aclnnStatus ResolveShapeInfo(const ChunkGatedDeltaRuleFwdParams &params, GdnShapeInfo &info)
{
    if (std::strcmp(params.layout, "BNSD") == 0) {
        info.layout = GdnLayout::BNSD;
    } else if (std::strcmp(params.layout, "BSND") == 0) {
        info.layout = GdnLayout::BSND;
    } else if (std::strcmp(params.layout, "NTD") == 0) {
        info.layout = GdnLayout::NTD;
    } else if (std::strcmp(params.layout, "TND") == 0) {
        info.layout = GdnLayout::TND;
    } else {
        CHECK_COND(false, ACLNN_ERR_PARAM_INVALID,
                   "layout must be uppercase and one of BNSD, BSND, NTD or TND.");
    }
    info.isSequenceMajor = info.layout == GdnLayout::BSND || info.layout == GdnLayout::TND;
    CHECK_COND(Rank(params.q) == 4 && Rank(params.k) == 4 && Rank(params.v) == 4 &&
                   Rank(params.oOut) == 4 && Rank(params.g) == 3 && Rank(params.beta) == 3,
               ACLNN_ERR_PARAM_INVALID,
               "q/k/v/o must be rank 4 and g/beta must be rank 3.");
    CHECK_COND(ShapeEqual(params.q->GetViewShape(), params.k->GetViewShape()),
               ACLNN_ERR_PARAM_INVALID, "q and k must have identical shapes.");

    if (!info.isSequenceMajor) {
        info.batch = Dim(params.q, 0);
        info.hq = Dim(params.q, 1);
        info.seqlen = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.hv = Dim(params.v, 1);
        info.vDim = Dim(params.v, 3);
        CHECK_COND(HasShape(params.v, {info.batch, info.hv, info.seqlen, info.vDim}),
                   ACLNN_ERR_PARAM_INVALID, "BNSD/NTD expects v as [B,Hv,T,V].");
    } else {
        info.batch = Dim(params.q, 0);
        info.seqlen = Dim(params.q, 1);
        info.hq = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.hv = Dim(params.v, 2);
        info.vDim = Dim(params.v, 3);
        CHECK_COND(HasShape(params.v, {info.batch, info.seqlen, info.hv, info.vDim}),
                   ACLNN_ERR_PARAM_INVALID, "BSND/TND expects v as [B,T,Hv,V].");
    }
    CHECK_COND(HasShape(params.oOut, {info.batch, info.seqlen, info.hv, info.vDim}),
               ACLNN_ERR_PARAM_INVALID, "oOut must use BSND shape [B,T,Hv,V].");
    CHECK_COND(HasShape(params.g, {info.batch, info.seqlen, info.hv}) &&
                   HasShape(params.beta, {info.batch, info.seqlen, info.hv}),
               ACLNN_ERR_PARAM_INVALID, "g and beta must have shape [B,T,Hv].");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckSupportedL2Contract(const ChunkGatedDeltaRuleFwdParams &params)
{
    CHECK_COND(params.layout != nullptr, ACLNN_ERR_PARAM_NULLPTR, "layout must not be nullptr.");
    if (UsePreparePath(params)) {
        CHECK_COND(IsAscend950(), ACLNN_ERR_PARAM_INVALID,
                   "The prepare path is supported on Ascend950 only.");
        CHECK_COND(std::strcmp(params.layout, "BNSD") == 0 || std::strcmp(params.layout, "BSND") == 0 ||
                       std::strcmp(params.layout, "NTD") == 0 || std::strcmp(params.layout, "TND") == 0,
                   ACLNN_ERR_PARAM_INVALID,
                   "The Ascend950 prepare path supports BNSD, BSND, NTD and TND.");
        CHECK_COND(params.useQkL2norm ||
                       (params.qHatOutOptional == nullptr && params.kHatOutOptional == nullptr &&
                        params.qRstdOutOptional == nullptr && params.kRstdOutOptional == nullptr),
                   ACLNN_ERR_PARAM_INVALID,
                   "Q/K L2Norm outputs must be omitted when useQkL2norm is false.");
        return ACLNN_SUCCESS;
    }
    CHECK_COND(IsA2FeatureRequest(params) ||
                   std::strcmp(params.layout, "BNSD") == 0 || std::strcmp(params.layout, "NTD") == 0 ||
                   IsNativeQkvLayout(params),
               ACLNN_ERR_PARAM_INVALID, "This device does not support the requested QKV layout.");
    CHECK_COND((IsA2FeatureRequest(params) && params.useQkL2norm) ||
                    (params.qHatOutOptional == nullptr && params.kHatOutOptional == nullptr &&
                     params.qRstdOutOptional == nullptr && params.kRstdOutOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "Q/K L2Norm intermediate outputs are reserved by the stable ABI but are not supported yet.");
    CHECK_COND(IsA2FeatureRequest(params) || params.betaEffOutOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "betaEffOutOptional is reserved by the stable ABI but is not supported yet.");
    CHECK_COND(IsA2FeatureRequest(params) || params.hOutOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "hOutOptional is reserved by the stable ABI but is not supported yet.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const ChunkGatedDeltaRuleFwdParams &params)
{
    CHECK_COND(params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 ||
                   params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_128,
               ACLNN_ERR_PARAM_INVALID, "chunkSize must be 64 or 128.");
    const aclnnStatus supportedContractStatus = CheckSupportedL2Contract(params);
    if (supportedContractStatus != ACLNN_SUCCESS) {
        return supportedContractStatus;
    }
    GdnShapeInfo info;
    CHECK_RET(ResolveShapeInfo(params, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const size_t qkvRank = 4;
    const size_t scalarRank = 3;
    CHECK_RET(CheckOptionalRank(params.gCumsumOutOptional, scalarRank, "gCumsumOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalRank(params.aOutOptional, qkvRank, "aOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalRank(params.qHatOutOptional, qkvRank, "qHatOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalRank(params.kHatOutOptional, qkvRank, "kHatOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalRank(params.qRstdOutOptional, scalarRank, "qRstdOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalRank(params.kRstdOutOptional, scalarRank, "kRstdOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalRank(params.betaEffOutOptional, scalarRank, "betaEffOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalRank(params.hOutOptional, 5, "hOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    CHECK_COND((params.cuSeqlensOptional == nullptr) == (params.chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID, "cuSeqlens and chunkIndices must be both present or both absent.");
    CHECK_COND(params.cuSeqlensOptional == nullptr || info.batch == 1, ACLNN_ERR_PARAM_INVALID,
               "varlen rank-4 input requires physical B=1.");
    CHECK_RET(CheckMetadata(params, info.seqlen) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    const int64_t chunks = ExpectedChunks(params, info.seqlen);
    CHECK_COND(info.batch > 0 && info.hq > 0 && info.hv > 0 && info.seqlen > 0,
               ACLNN_ERR_PARAM_INVALID, "B/H/T dimensions must be positive.");
    CHECK_COND(info.hv % info.hq == 0, ACLNN_ERR_PARAM_INVALID,
               "Phase 6 GVA requires Hv divisible by Hk.");
    if (IsA2FeatureRequest(params)) {
        CHECK_COND(params.dtBiasOptional == nullptr || params.aLogOptional != nullptr,
                   ACLNN_ERR_PARAM_INVALID, "dtBiasOptional requires aLogOptional.");
        CHECK_COND(!params.allowNegEigval || params.betaEffOutOptional != nullptr,
                   ACLNN_ERR_PARAM_INVALID, "allowNegEigval requires beta sigmoid output.");
        for (const aclTensor *parameter : {params.aLogOptional, params.dtBiasOptional}) {
            CHECK_COND(parameter == nullptr ||
                           (HasShape(parameter, {info.hv}) &&
                            (parameter->GetDataType() == DataType::DT_FLOAT ||
                             parameter->GetDataType() == DataType::DT_FLOAT16 ||
                             parameter->GetDataType() == DataType::DT_BF16)),
                       ACLNN_ERR_PARAM_INVALID, "A_log/dt_bias must be FP32/FP16/BF16 [Hv].");
        }
    }
    CHECK_COND(info.kDim == HEAD_DIM_128 &&
                   (info.vDim == HEAD_DIM_128 ||
                    info.vDim == CHUNK_GATED_DELTA_RULE_FWD_V256),
               ACLNN_ERR_PARAM_INVALID,
               "The Phase 6 composite GDN core supports K=128 and V=128/256.");
    if (params.gCumsumOutOptional != nullptr) {
        CHECK_COND(ShapeEqual(params.gCumsumOutOptional->GetViewShape(), params.g->GetViewShape()),
                   ACLNN_ERR_PARAM_INVALID, "gCumsumOutOptional shape must match g.");
    }
    if (params.aOutOptional != nullptr) {
        const bool valid = HasShape(params.aOutOptional,
                                    {info.batch, info.hv, info.seqlen, params.chunkSize});
        CHECK_COND(valid, ACLNN_ERR_PARAM_INVALID,
                   "aOutOptional must use head-major [B,Hv,T,chunkSize].");
    }
    if (params.qHatOutOptional != nullptr) {
        CHECK_COND(ShapeEqual(params.qHatOutOptional->GetViewShape(), params.q->GetViewShape()),
                   ACLNN_ERR_PARAM_INVALID,
                   "qHatOutOptional shape must match q.");
    }
    if (params.kHatOutOptional != nullptr) {
        CHECK_COND(ShapeEqual(params.kHatOutOptional->GetViewShape(), params.k->GetViewShape()),
                   ACLNN_ERR_PARAM_INVALID,
                   "kHatOutOptional shape must match k.");
    }
    if (params.qRstdOutOptional != nullptr) {
        const bool valid = HasShape(params.qRstdOutOptional, {info.batch, info.hq, info.seqlen});
        CHECK_COND(valid, ACLNN_ERR_PARAM_INVALID, "qRstdOutOptional must have shape [B,Hk,T].");
    }
    if (params.kRstdOutOptional != nullptr) {
        const bool valid = HasShape(params.kRstdOutOptional, {info.batch, info.hq, info.seqlen});
        CHECK_COND(valid, ACLNN_ERR_PARAM_INVALID, "kRstdOutOptional must have shape [B,Hk,T].");
    }
    if (params.betaEffOutOptional != nullptr) {
        CHECK_COND(ShapeEqual(params.betaEffOutOptional->GetViewShape(), params.beta->GetViewShape()),
                   ACLNN_ERR_PARAM_INVALID, "betaEffOutOptional shape must match beta.");
    }
    if (params.hOutOptional != nullptr) {
        const int64_t stateDim0 = params.stateVFirst ? info.vDim : info.kDim;
        const int64_t stateDim1 = params.stateVFirst ? info.kDim : info.vDim;
        const bool valid = HasShape(params.hOutOptional,
                                    {info.batch, chunks, info.hv, stateDim0, stateDim1});
        CHECK_COND(valid, ACLNN_ERR_PARAM_INVALID,
                   "hOutOptional shape must match stateVFirst.");
    }

    const int64_t seqNum = SeqNum(params, info.batch);
    const int64_t stateDim0 = params.stateVFirst ? info.vDim : info.kDim;
    const int64_t stateDim1 = params.stateVFirst ? info.kDim : info.vDim;
    CHECK_RET(CheckStateShape(params.initialStateOptional, "initialState", seqNum, info.hv, stateDim0, stateDim1) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    if (params.finalStateOutOptional != nullptr) {
        CHECK_RET(CheckStateShape(params.finalStateOutOptional, "finalStateOut", seqNum, info.hv,
                                  stateDim0, stateDim1) ==
                      ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }

    const DataType dtype = params.q->GetDataType();
    CHECK_COND(dtype == DataType::DT_FLOAT16 || dtype == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "q/k/v must be float16 or bfloat16.");
    CHECK_COND(params.k->GetDataType() == dtype && params.v->GetDataType() == dtype &&
                   params.oOut->GetDataType() == dtype &&
                   (params.aOutOptional == nullptr || params.aOutOptional->GetDataType() == dtype),
               ACLNN_ERR_PARAM_INVALID, "q/k/v/oOut/aOutOptional must have the same dtype.");
    CHECK_COND((params.beta->GetDataType() == DataType::DT_FLOAT || params.beta->GetDataType() == dtype) &&
                   (params.g->GetDataType() == DataType::DT_FLOAT || params.g->GetDataType() == dtype),
               ACLNN_ERR_PARAM_INVALID, "g and beta must be float32 or match q/k/v.");
    CHECK_COND(params.gCumsumOutOptional == nullptr || params.gCumsumOutOptional->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "gCumsumOutOptional must be float32.");
    CHECK_COND((params.qHatOutOptional == nullptr || params.qHatOutOptional->GetDataType() == dtype) &&
                   (params.kHatOutOptional == nullptr || params.kHatOutOptional->GetDataType() == dtype) &&
                   (params.hOutOptional == nullptr || params.hOutOptional->GetDataType() == dtype),
               ACLNN_ERR_PARAM_INVALID, "qHatOutOptional/kHatOutOptional/hOutOptional must match q dtype.");
    CHECK_COND((params.qRstdOutOptional == nullptr ||
                    params.qRstdOutOptional->GetDataType() == DataType::DT_FLOAT) &&
                   (params.kRstdOutOptional == nullptr ||
                    params.kRstdOutOptional->GetDataType() == DataType::DT_FLOAT) &&
                   (params.betaEffOutOptional == nullptr ||
                    params.betaEffOutOptional->GetDataType() == DataType::DT_FLOAT),
               ACLNN_ERR_PARAM_INVALID, "qRstdOutOptional/kRstdOutOptional/betaEffOutOptional must be float32.");
    if (UsePreparePath(params)) {
        CHECK_COND(dtype == DataType::DT_BF16 && info.vDim == HEAD_DIM_128 &&
                       params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 && info.hv / info.hq <= 4,
                   ACLNN_ERR_PARAM_INVALID,
                   "Ascend950 prepare path requires BF16, K=V=128, chunkSize=64 and Hv/Hk in {1,2,3,4}.");
    }
    if (params.initialStateOptional != nullptr) {
        const DataType stateDtype = params.initialStateOptional->GetDataType();
        CHECK_COND(stateDtype == DataType::DT_FLOAT || stateDtype == dtype, ACLNN_ERR_PARAM_INVALID,
                   "initialState dtype must be float32 or match q/k/v.");
    }
    if (params.finalStateOutOptional != nullptr) {
        const DataType expectedStateDtype = params.initialStateOptional == nullptr
                                                ? DataType::DT_FLOAT
                                                : params.initialStateOptional->GetDataType();
        CHECK_COND(params.finalStateOutOptional->GetDataType() == expectedStateDtype,
                   ACLNN_ERR_PARAM_INVALID,
                   "finalStateOut dtype must match initialState, or be float32 when initialState is absent.");
    }
    return ACLNN_SUCCESS;
}

} // namespace

static aclnnStatus ChunkGatedDeltaRuleFwdGetWorkspaceSizeImpl(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool useExp2,
    bool useQkL2norm,
    bool allowNegEigval,
    bool stateVFirst,
    const aclTensor *oOut,
    const aclTensor *finalStateOutOptional,
    const aclTensor *qHatOutOptional,
    const aclTensor *kHatOutOptional,
    const aclTensor *qRstdOutOptional,
    const aclTensor *kRstdOutOptional,
    const aclTensor *betaEffOutOptional,
    const aclTensor *gCumsumOutOptional,
    const aclTensor *aOutOptional,
    const aclTensor *hOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkGatedDeltaRuleFwdParams params{
        q, k, v, g, beta, aLogOptional, dtBiasOptional, initialStateOptional,
        cuSeqlensOptional, chunkIndicesOptional, layout, scale, chunkSize, useExp2,
        useQkL2norm, allowNegEigval, stateVFirst, oOut, finalStateOutOptional, qHatOutOptional,
        kHatOutOptional, qRstdOutOptional, kRstdOutOptional, betaEffOutOptional,
        gCumsumOutOptional, aOutOptional, hOutOptional};
    CHECK_COND(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");
    CHECK_RET(CheckRequiredInputs(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    if (CheckZeroShape(params, workspaceSize) != ACLNN_SUCCESS) {
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(MakeContiguous(params.q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.beta, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.initialStateOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    GdnShapeInfo info;
    CHECK_RET(ResolveShapeInfo(params, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const int64_t batch = info.batch;
    const int64_t hq = info.hq;
    const int64_t hv = info.hv;
    const int64_t seqlen = info.seqlen;
    const int64_t kDim = info.kDim;
    const int64_t vDim = info.vDim;
    const int64_t seqNum = SeqNum(params, batch);
    const bool outputFinalState = params.finalStateOutOptional != nullptr;
    const bool usePreparedCumsum = UsePreparedCumsum(params, info);
    const bool useA2Features = IsA2FeatureRequest(params);
    const bool useA2Norm = useA2Features && params.useQkL2norm;
    const aclTensor *betaUsed = params.beta;
    const aclTensor *gUsed = params.g;
    // The legacy Phase6 kernels consume FP32 gate/beta. Keep A5's native
    // low-precision prepare contract while normalizing the legacy inputs.
    if (!IsAscend950() || params.beta->GetDataType() == DataType::DT_FLOAT ||
        params.g->GetDataType() == DataType::DT_FLOAT) {
        betaUsed = params.beta->GetDataType() == DataType::DT_FLOAT
                       ? params.beta : l0op::Cast(params.beta, DataType::DT_FLOAT, executorPtr);
        gUsed = params.g->GetDataType() == DataType::DT_FLOAT
                    ? params.g : l0op::Cast(params.g, DataType::DT_FLOAT, executorPtr);
    }
    const aclTensor *gRaw = usePreparedCumsum
                               ? gUsed
                               : (gUsed == nullptr ? nullptr
                                  : TransposeContiguous(gUsed, {0, 2, 1}, executorPtr));
    const aclTensor *betaBht = betaUsed == nullptr ? nullptr
                                 : TransposeContiguous(betaUsed, {0, 2, 1}, executorPtr);
    GDN_STAGE_CHECK(gRaw != nullptr && betaBht != nullptr, 169102);
    const aclTensor *gateParameters = nullptr;
    if (useA2Features && params.aLogOptional != nullptr) {
        gateParameters = MakeA2GateParameters(params.aLogOptional, params.dtBiasOptional, hv, executorPtr);
        GDN_STAGE_CHECK(gateParameters != nullptr, 169122);
    }

    if (UsePreparePath(params)) {
        const op::Shape qkShape = MakeShape({batch, hq, seqlen, kDim});
        const op::Shape scalarQkShape = MakeShape({batch, hq, seqlen});
        const op::Shape scalarHvShape = MakeShape({batch, hv, seqlen});
        const op::Shape wShape = MakeShape({batch, hv, seqlen, kDim});
        const op::Shape vShape = MakeShape({batch, hv, seqlen, vDim});
        const op::Shape aShape = MakeShape({batch, hv, seqlen, params.chunkSize});
        const int64_t stateDim0 = params.stateVFirst ? vDim : kDim;
        const int64_t stateDim1 = params.stateVFirst ? kDim : vDim;
        const op::Shape hShape = MakeShape({batch, ExpectedChunks(params, seqlen), hv, stateDim0, stateDim1});
        const op::Shape stateShape = MakeShape({seqNum, hv, stateDim0, stateDim1});
        const DataType dtype = params.q->GetDataType();
        const DataType stateDtype = params.initialStateOptional == nullptr
                                        ? DataType::DT_FLOAT
                                        : params.initialStateOptional->GetDataType();

        const aclTensor *gCumsumBht = executorPtr->AllocTensor(scalarHvShape, DataType::DT_FLOAT, Format::FORMAT_ND);
        const aclTensor *w = executorPtr->AllocTensor(wShape, dtype, Format::FORMAT_ND);
        const aclTensor *u = executorPtr->AllocTensor(vShape, dtype, Format::FORMAT_ND);
        const aclTensor *a = executorPtr->AllocTensor(aShape, dtype, Format::FORMAT_ND);
        const aclTensor *qHat = params.useQkL2norm
                                    ? executorPtr->AllocTensor(qkShape, dtype, Format::FORMAT_ND)
                                    : nullptr;
        const aclTensor *kHat = params.useQkL2norm
                                    ? executorPtr->AllocTensor(qkShape, dtype, Format::FORMAT_ND)
                                    : nullptr;
        const aclTensor *qRstd = params.useQkL2norm
                                     ? executorPtr->AllocTensor(scalarQkShape, DataType::DT_FLOAT, Format::FORMAT_ND)
                                     : nullptr;
        const aclTensor *kRstd = params.useQkL2norm
                                     ? executorPtr->AllocTensor(scalarQkShape, DataType::DT_FLOAT, Format::FORMAT_ND)
                                     : nullptr;
        const aclTensor *betaEffBht = params.betaEffOutOptional == nullptr
                                          ? nullptr
                                          : executorPtr->AllocTensor(scalarHvShape, DataType::DT_FLOAT,
                                                                     Format::FORMAT_ND);
        const aclTensor *h = executorPtr->AllocTensor(hShape, dtype, Format::FORMAT_ND);
        const aclTensor *vNew = executorPtr->AllocTensor(vShape, dtype, Format::FORMAT_ND);
        const aclTensor *finalState = params.finalStateOutOptional == nullptr
                                          ? executorPtr->AllocTensor(stateShape, stateDtype, Format::FORMAT_ND)
                                          : params.finalStateOutOptional;
        GDN_STAGE_CHECK(gCumsumBht != nullptr && w != nullptr && u != nullptr && a != nullptr &&
                            (!params.useQkL2norm ||
                             (qHat != nullptr && kHat != nullptr && qRstd != nullptr && kRstd != nullptr)) &&
                            h != nullptr && vNew != nullptr && finalState != nullptr &&
                            (params.betaEffOutOptional == nullptr || betaEffBht != nullptr),
                        169103);

        const aclTensor *qHead = params.q;
        const aclTensor *kHead = params.k;
        const aclTensor *vHead = params.v;
        if (info.isSequenceMajor) {
            qHead = TransposeContiguous(params.q, {0, 2, 1, 3}, executorPtr);
            kHead = TransposeContiguous(params.k, {0, 2, 1, 3}, executorPtr);
            vHead = TransposeContiguous(params.v, {0, 2, 1, 3}, executorPtr);
        }
        GDN_STAGE_CHECK(qHead != nullptr && kHead != nullptr && vHead != nullptr, 169108);
        const aclTensor *qCompute = params.useQkL2norm ? qHat : qHead;
        const aclTensor *kCompute = params.useQkL2norm ? kHat : kHead;

        auto prepareResult = l0op::ChunkGatedDeltaRuleFwdPrepare(
            qHead, kHead, vHead, gRaw, betaBht, params.aLogOptional, params.dtBiasOptional,
            params.cuSeqlensOptional, params.chunkIndicesOptional, params.chunkSize,
            params.allowNegEigval, params.useExp2, params.useQkL2norm,
            params.aLogOptional != nullptr || params.dtBiasOptional != nullptr, betaEffBht != nullptr,
            params.aOutOptional != nullptr, gCumsumBht, w, u, a, qHat, kHat, qRstd,
            kRstd, betaEffBht, executorPtr);
        GDN_STAGE_CHECK(prepareResult[0] != nullptr && prepareResult[1] != nullptr &&
                            prepareResult[2] != nullptr &&
                            (!params.useQkL2norm ||
                             (prepareResult[4] != nullptr && prepareResult[5] != nullptr)),
                        169104);

        auto hResult = l0op::ChunkFwdH(
            kCompute, w, u, gCumsumBht, nullptr, params.initialStateOptional,
            params.cuSeqlensOptional, params.chunkIndicesOptional, outputFinalState,
            params.chunkSize, true, params.useExp2, params.stateVFirst, h, vNew, finalState, executorPtr);
        GDN_STAGE_CHECK(hResult[0] != nullptr && hResult[1] != nullptr, 169105);

        auto oResult = l0op::ChunkFwdO(
            qCompute, kCompute, vNew, h, gCumsumBht, params.cuSeqlensOptional,
            params.chunkIndicesOptional, params.scale, params.chunkSize, params.useExp2, params.stateVFirst,
            "BSND", params.oOut, executorPtr);
        GDN_STAGE_CHECK(oResult[0] != nullptr, 169106);

        if (params.qHatOutOptional != nullptr) {
            const aclTensor *qHatExport = qHat;
            if (info.isSequenceMajor) {
                qHatExport = TransposeContiguous(qHatExport, {0, 2, 1, 3}, executorPtr);
            }
            CHECK_RET(ViewCopyIfPresent(qHatExport, params.qHatOutOptional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_INNER_NULLPTR);
        }
        if (params.kHatOutOptional != nullptr) {
            const aclTensor *kHatExport = kHat;
            if (info.isSequenceMajor) {
                kHatExport = TransposeContiguous(kHatExport, {0, 2, 1, 3}, executorPtr);
            }
            CHECK_RET(ViewCopyIfPresent(kHatExport, params.kHatOutOptional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_INNER_NULLPTR);
        }
        if (params.qRstdOutOptional != nullptr) {
            CHECK_RET(ViewCopyIfPresent(qRstd, params.qRstdOutOptional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_INNER_NULLPTR);
        }
        if (params.kRstdOutOptional != nullptr) {
            CHECK_RET(ViewCopyIfPresent(kRstd, params.kRstdOutOptional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_INNER_NULLPTR);
        }
        const aclTensor *aExport = a;
        const aclTensor *hExport = h;
        CHECK_RET(ViewCopyIfPresent(aExport, params.aOutOptional, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(ViewCopyIfPresent(hExport, params.hOutOptional, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        if (params.gCumsumOutOptional != nullptr) {
            const aclTensor *gCumsumBth = TransposeContiguous(gCumsumBht, {0, 2, 1}, executorPtr);
            const aclTensor *gCumsumExport = gCumsumBth;
            CHECK_RET(ViewCopyIfPresent(gCumsumExport, params.gCumsumOutOptional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_INNER_NULLPTR);
        }
        if (params.betaEffOutOptional != nullptr) {
            const aclTensor *betaEffBth = TransposeContiguous(betaEffBht, {0, 2, 1}, executorPtr);
            const aclTensor *betaEffExport = betaEffBth;
            CHECK_RET(ViewCopyIfPresent(betaEffExport, params.betaEffOutOptional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_INNER_NULLPTR);
        }

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto aStorageBhtc = executorPtr->AllocTensor(MakeShape({batch, hv, seqlen, params.chunkSize}),
                                                  params.k->GetDataType(), Format::FORMAT_ND);
    const aclTensor *initialStateCompute = params.initialStateOptional;
    if (useA2Features && params.stateVFirst && initialStateCompute != nullptr) {
        initialStateCompute = TransposeContiguous(initialStateCompute, {0, 1, 3, 2}, executorPtr);
        GDN_STAGE_CHECK(initialStateCompute != nullptr, 169124);
    }
    GDN::QkNormOutputLayout norm{};
    if (useA2Features) {
        GDN::FrontHoSharedLayout baseFront{};
        CHECK_COND(GDN::MakeFrontHoSharedLayout(batch, hv, seqlen, ExpectedChunks(params, seqlen),
                       params.chunkSize, vDim, usePreparedCumsum, baseFront) &&
                       GDN::MakeFrontFeatureOutputLayout(batch, hq, hv, seqlen,
                           IsNativeQkvLayout(params) ? 1 : 0, baseFront, useA2Norm,
                           params.betaEffOutOptional != nullptr, norm),
                   ACLNN_ERR_PARAM_INVALID, "Private Prepare feature output capacity overflows.");
    }
    const aclTensor *finalState = params.finalStateOutOptional;
    if (!outputFinalState) {
        finalState = executorPtr->AllocTensor(MakeShape({1}), DataType::DT_FLOAT, Format::FORMAT_ND);
    } else if (useA2Features) {
        finalState = executorPtr->AllocTensor(MakeShape({SeqNum(params, batch), hv, HEAD_DIM_128, vDim}),
                                               params.finalStateOutOptional->GetDataType(), Format::FORMAT_ND);
    }
    const aclTensor *gCumsumCompute = params.gCumsumOutOptional;
    if (useA2Features) {
        const bool packedFeatures = params.useQkL2norm || params.useExp2 || params.aLogOptional ||
            params.dtBiasOptional || params.betaEffOutOptional || params.allowNegEigval;
        gCumsumCompute = executorPtr->AllocTensor(
            packedFeatures ? MakeShape({static_cast<int64_t>(norm.gElements)}) : MakeShape({batch, seqlen, hv}),
            DataType::DT_FLOAT, Format::FORMAT_ND);
    } else if (gCumsumCompute == nullptr) {
        // A5 Phase6 recognizes this required-output placeholder and skips only
        // the unused public BTH export; its internal BHT cumsum remains intact.
        const auto gCumsumShape = IsAscend950() ? MakeShape({1}) : MakeShape({batch, seqlen, hv});
        gCumsumCompute = executorPtr->AllocTensor(gCumsumShape, DataType::DT_FLOAT, Format::FORMAT_ND);
    } else if (IsAscend950()) {
        // Public descriptors may expose a flat storage shape. Give tiling an
        // executor-owned dense BTH view without changing the caller's descriptor
        // or allocating/copying output data. Preserve the caller's data offset.
        const auto gCumsumShape = MakeShape({batch, seqlen, hv});
        auto *gCumsumView = executorPtr->CreateView(
            gCumsumCompute, gCumsumShape, gCumsumCompute->GetViewOffset());
        if (gCumsumView != nullptr) {
            gCumsumView->SetStorageShape(gCumsumShape);
            gCumsumView->SetOriginalShape(gCumsumShape);
        }
        gCumsumCompute = gCumsumView;
    }
    // 本实验的 DAV_2201 入口统一为 Prepare/H/O，A3 也不再进入旧 phase0。
    // 输入布局和 prepared-cumsum 的选择仍遵循各自原有合同。
    const bool splitFrontHo = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201;
    GDN::FrontHoSharedLayout shared{};
    if (splitFrontHo) {
        CHECK_COND(GDN::MakeFrontHoSharedLayout(batch, hv, seqlen,
                       ExpectedChunks(params, seqlen), params.chunkSize, vDim,
                       usePreparedCumsum, shared), ACLNN_ERR_PARAM_INVALID,
                   "Shared front/H-O storage size overflows.");
        if (useA2Features) shared.columns = norm.frontColumns;
    }
    const aclTensor *sharedFront = executorPtr->AllocTensor(
        splitFrontHo ? MakeShape({batch, hv, seqlen, static_cast<int64_t>(shared.columns)}) : MakeShape({1}),
        params.q->GetDataType(), Format::FORMAT_ND);
    GDN_STAGE_CHECK(sharedFront != nullptr, 169119);
    const aclTensor *aCompute = params.aOutOptional;
    if (aCompute == nullptr || useA2Features) {
        aCompute = executorPtr->AllocTensor(MakeShape({batch, hv, seqlen, params.chunkSize}), params.q->GetDataType(),
                                            Format::FORMAT_ND);
    }
    GDN_STAGE_CHECK(aStorageBhtc != nullptr && finalState != nullptr &&
                        gCumsumCompute != nullptr && aCompute != nullptr,
                    169101);

    const bool sequenceMajorOutput = IsDav2201CandidateSoc();
    const aclTensor *oCompute = executorPtr->AllocTensor(
        sequenceMajorOutput ? MakeShape({batch, seqlen, hv, vDim}) : MakeShape({batch, hv, seqlen, vDim}),
        params.q->GetDataType(), Format::FORMAT_ND);
    GDN_STAGE_CHECK(oCompute != nullptr, 169109);
    const bool nativeQkv = IsNativeQkvLayout(params);
    const bool transposeFeatureQkv = useA2Features && !nativeQkv &&
        (std::strcmp(params.layout, "BSND") == 0 || std::strcmp(params.layout, "TND") == 0);
    const aclTensor *qInput = transposeFeatureQkv ? TransposeContiguous(params.q, {0, 2, 1, 3}, executorPtr) :
        (nativeQkv ? DenseQkvView(params.q, executorPtr) : params.q);
    const aclTensor *kInput = transposeFeatureQkv ? TransposeContiguous(params.k, {0, 2, 1, 3}, executorPtr) :
        (nativeQkv ? DenseQkvView(params.k, executorPtr) : params.k);
    const aclTensor *vInput = transposeFeatureQkv ? TransposeContiguous(params.v, {0, 2, 1, 3}, executorPtr) :
        (nativeQkv ? DenseQkvView(params.v, executorPtr) : params.v);
    GDN_STAGE_CHECK(qInput != nullptr && kInput != nullptr && vInput != nullptr, 169113);
    const aclTensor *qHat = nullptr;
    const aclTensor *kHat = nullptr;
    const aclTensor *qRstd = nullptr;
    const aclTensor *kRstd = nullptr;
    const aclTensor *gNormExport = nullptr;
    const aclTensor *betaEff = nullptr;
    if (useA2Features) {
        gNormExport = NormParentView(gCumsumCompute, MakeShape({batch, seqlen, hv}), 0, executorPtr);
        if (params.betaEffOutOptional != nullptr)
            betaEff = NormParentView(gCumsumCompute, MakeShape({batch, hv, seqlen}), norm.betaEff, executorPtr);
        GDN_STAGE_CHECK(gNormExport && (params.betaEffOutOptional == nullptr || betaEff), 169123);
    }
    if (useA2Norm) {
        qHat = NormParentView(sharedFront, qInput->GetViewShape(), norm.qHat, executorPtr);
        kHat = NormParentView(sharedFront, kInput->GetViewShape(), norm.kHat, executorPtr);
        const auto rstdShape = MakeShape({batch, hq, seqlen});
        qRstd = NormParentView(gCumsumCompute, rstdShape, norm.qRstd, executorPtr);
        kRstd = NormParentView(gCumsumCompute, rstdShape, norm.kRstd, executorPtr);
        GDN_STAGE_CHECK(qHat && kHat && qRstd && kRstd, 169121);
    }
    const aclTensor *qHoInput = useA2Norm ? qHat : qInput;
    const aclTensor *kHoInput = useA2Norm ? kHat : kInput;
    // 无关输出使用各段独立占位，防止executor把同一实际输出重复绑定。
    const aclTensor *frontDummyO = splitFrontHo ? executorPtr->AllocTensor(
        MakeShape({1}), params.q->GetDataType(), Format::FORMAT_ND) : oCompute;
    const aclTensor *frontDummyState = splitFrontHo ? executorPtr->AllocTensor(
        MakeShape({1}), finalState->GetDataType(), Format::FORMAT_ND) : finalState;
    const aclTensor *hDummyO = executorPtr->AllocTensor(MakeShape({1}), params.q->GetDataType(), Format::FORMAT_ND);
    const aclTensor *hDummyShared = executorPtr->AllocTensor(MakeShape({1}), params.q->GetDataType(), Format::FORMAT_ND);
    const aclTensor *oDummyState = executorPtr->AllocTensor(MakeShape({1}), finalState->GetDataType(), Format::FORMAT_ND);
    const aclTensor *oDummyG = executorPtr->AllocTensor(MakeShape({1}), DataType::DT_FLOAT, Format::FORMAT_ND);
    const aclTensor *oDummyShared = executorPtr->AllocTensor(MakeShape({1}), params.q->GetDataType(), Format::FORMAT_ND);
    GDN_STAGE_CHECK(frontDummyO && frontDummyState && hDummyO && hDummyShared &&
                        oDummyState && oDummyG && oDummyShared, 169120);
    const int64_t phaseLayout = (usePreparedCumsum ? 1 : 0) + (splitFrontHo ? 2 : 0) +
        (useA2Norm ? GDN::kFrontQkNormBit : 0) +
        (useA2Features && params.betaEffOutOptional ? GDN::kFrontBetaSigmoidBit | GDN::kFrontOutputBetaBit : 0) +
        (useA2Features && params.aLogOptional ? GDN::kFrontGateBit : 0) +
        (useA2Features && params.dtBiasOptional ? GDN::kFrontDtBiasBit : 0) +
        (useA2Features && params.allowNegEigval ? GDN::kFrontAllowNegBit : 0) +
        (useA2Features && params.useExp2 ? GDN::kFrontExp2Bit : 0);
    const int64_t suffixLayout = (usePreparedCumsum ? 1 : 0) +
        (useA2Norm ? GDN::kFrontQkNormBit : 0) +
        (useA2Features && params.useExp2 ? GDN::kFrontExp2Bit : 0);
    auto phase6Result = l0op::ChunkGatedDeltaRuleFwd(
        qInput, kInput, vInput, betaBht, aStorageBhtc, gRaw, gateParameters,
        initialStateCompute, params.cuSeqlensOptional, params.chunkIndicesOptional,
        splitFrontHo ? false : outputFinalState, params.chunkSize, params.scale, params.gCumsumOutOptional != nullptr,
        frontDummyO, frontDummyState,
        gCumsumCompute, aCompute, phaseLayout, nativeQkv ? 1 : 0,
        sequenceMajorOutput ? 1 : 0, sharedFront, nullptr, executorPtr);
    GDN_STAGE_CHECK(phase6Result[0] != nullptr && phase6Result[2] != nullptr &&
                        phase6Result[3] != nullptr,
                        169112);
    const aclTensor *hAuxExport = nullptr;
    if (splitFrontHo) {
        const aclTensor *suffixGPlaceholder = executorPtr->AllocTensor(
            MakeShape({batch, seqlen, hv}), DataType::DT_FLOAT, Format::FORMAT_ND);
        GDN_STAGE_CHECK(suffixGPlaceholder != nullptr, 169116);
        GDN::HoSharedLayout hoShared{};
        CHECK_COND(GDN::MakeHoSharedLayout(batch, hv, seqlen,
                       ExpectedChunks(params, seqlen), vDim, hoShared), ACLNN_ERR_PARAM_INVALID,
                   "Private H/O storage size overflows.");
        const aclTensor *hoCompute = executorPtr->AllocTensor(
            MakeShape({batch, hv, seqlen, static_cast<int64_t>(hoShared.columns)}),
            params.q->GetDataType(), Format::FORMAT_ND);
        GDN_STAGE_CHECK(hoCompute != nullptr, 169117);
        // 独立H只生产h/vNew/final_state；A/g仍是Prepare的唯一输出。
        phase6Result = l0op::ChunkGatedDeltaRuleFwd(
            qHoInput, kHoInput, vInput, betaBht, sharedFront, gRaw, nullptr,
            initialStateCompute, params.cuSeqlensOptional, params.chunkIndicesOptional,
            outputFinalState, params.chunkSize, params.scale, false,
            hDummyO, finalState, suffixGPlaceholder, hoCompute, suffixLayout + 4,
            nativeQkv ? 1 : 0, sequenceMajorOutput ? 1 : 0, hDummyShared, nullptr, executorPtr);
        GDN_STAGE_CHECK(phase6Result[0] != nullptr && phase6Result[3] != nullptr, 169114);
        // O明确消费两份executor-owned输入，跨kernel不共享局部scratch或flag。
        phase6Result = l0op::ChunkGatedDeltaRuleFwd(
            qHoInput, kHoInput, vInput, betaBht, sharedFront, gRaw, nullptr,
            initialStateCompute, params.cuSeqlensOptional, params.chunkIndicesOptional,
            false, params.chunkSize, params.scale, false,
            oCompute, oDummyState, oDummyG, aStorageBhtc, suffixLayout + 6,
            nativeQkv ? 1 : 0, sequenceMajorOutput ? 1 : 0, oDummyShared, hoCompute, executorPtr);
        GDN_STAGE_CHECK(phase6Result[0] != nullptr && phase6Result[3] != nullptr, 169118);
        if (params.hOutOptional != nullptr) {
            hAuxExport = NormParentView(hoCompute,
                MakeShape({batch, ExpectedChunks(params, seqlen), hv, HEAD_DIM_128, vDim}), 0, executorPtr);
            GDN_STAGE_CHECK(hAuxExport != nullptr, 169125);
            if (params.stateVFirst)
                hAuxExport = TransposeContiguous(hAuxExport, {0, 1, 2, 4, 3}, executorPtr);
            GDN_STAGE_CHECK(hAuxExport != nullptr, 169126);
        }
    }
    // ViewCopy binds a dense output or materializes the caller's strided view.
    const aclTensor *oSequence = sequenceMajorOutput ? oCompute :
        TransposeContiguous(oCompute, {0, 2, 1, 3}, executorPtr);
    GDN_STAGE_CHECK(oSequence != nullptr && l0op::ViewCopy(oSequence, params.oOut, executorPtr) != nullptr,
                    169107);
    if (useA2Features) {
        // 新公开输出在全部Prepare输入读结束后再绑定调用方，避免跨AIV/跨work别名。
        const aclTensor *qHatExport = qHat, *kHatExport = kHat;
        if (useA2Norm && transposeFeatureQkv) {
            qHatExport = TransposeContiguous(qHat, {0, 2, 1, 3}, executorPtr);
            kHatExport = TransposeContiguous(kHat, {0, 2, 1, 3}, executorPtr);
        }
        CHECK_RET(ViewCopyIfPresent(qHatExport, params.qHatOutOptional, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(ViewCopyIfPresent(kHatExport, params.kHatOutOptional, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(ViewCopyIfPresent(qRstd, params.qRstdOutOptional, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(ViewCopyIfPresent(kRstd, params.kRstdOutOptional, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(ViewCopyIfPresent(gNormExport, params.gCumsumOutOptional, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(ViewCopyIfPresent(aCompute, params.aOutOptional, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(ViewCopyIfPresent(hAuxExport, params.hOutOptional, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        if (outputFinalState) {
            const aclTensor *stateExport = params.stateVFirst ?
                TransposeContiguous(finalState, {0, 1, 3, 2}, executorPtr) : finalState;
            CHECK_RET(ViewCopyIfPresent(stateExport, params.finalStateOutOptional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_INNER_NULLPTR);
        }
        if (params.betaEffOutOptional != nullptr) {
            const auto *betaEffBth = TransposeContiguous(betaEff, {0, 2, 1}, executorPtr);
            CHECK_RET(ViewCopyIfPresent(betaEffBth, params.betaEffOutOptional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_INNER_NULLPTR);
        }
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool useExp2,
    bool useQkL2norm,
    bool allowNegEigval,
    bool stateVFirst,
    const aclTensor *oOut,
    const aclTensor *finalStateOutOptional,
    const aclTensor *qHatOutOptional,
    const aclTensor *kHatOutOptional,
    const aclTensor *qRstdOutOptional,
    const aclTensor *kRstdOutOptional,
    const aclTensor *betaEffOutOptional,
    const aclTensor *gCumsumOutOptional,
    const aclTensor *aOutOptional,
    const aclTensor *hOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleFwd,
                   DFX_IN(q, k, v, g, beta, aLogOptional, dtBiasOptional, initialStateOptional, cuSeqlensOptional,
                          chunkIndicesOptional, layout, scale, chunkSize, useExp2, useQkL2norm,
                          allowNegEigval, stateVFirst),
                   DFX_OUT(oOut, finalStateOutOptional, qHatOutOptional, kHatOutOptional, qRstdOutOptional,
                           kRstdOutOptional, betaEffOutOptional, gCumsumOutOptional, aOutOptional, hOutOptional));
    return ChunkGatedDeltaRuleFwdGetWorkspaceSizeImpl(
        q, k, v, g, beta, aLogOptional, dtBiasOptional, initialStateOptional, cuSeqlensOptional, chunkIndicesOptional,
        layout, scale, chunkSize, useExp2, useQkL2norm, allowNegEigval, stateVFirst, oOut,
        finalStateOutOptional, qHatOutOptional,
        kHatOutOptional, qRstdOutOptional, kRstdOutOptional, betaEffOutOptional, gCumsumOutOptional, aOutOptional,
        hOutOptional, workspaceSize, executor);
}

aclnnStatus aclnnChunkGatedDeltaRuleFwd(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleFwd);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkGatedDeltaRuleFwd launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
