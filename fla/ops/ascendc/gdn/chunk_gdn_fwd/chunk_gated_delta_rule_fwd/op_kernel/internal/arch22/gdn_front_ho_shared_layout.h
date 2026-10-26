/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_FRONT_HO_SHARED_LAYOUT_H
#define GDN_FRONT_HO_SHARED_LAYOUT_H
#include <cstdint>
namespace GDN {
// 仅私有三段式 IR 的 feature 编码；旧 2..7 的 phase/layout 含义不变。
constexpr int64_t kFrontQkNormBit = 8;
constexpr int64_t kFrontBetaSigmoidBit = 16;
constexpr int64_t kFrontGateBit = 32;
constexpr int64_t kFrontDtBiasBit = 64;
constexpr int64_t kFrontAllowNegBit = 128;
constexpr int64_t kFrontExp2Bit = 256;
constexpr int64_t kFrontOutputBetaBit = 512;
struct FrontFeatures {
    int64_t phase{}, layout{};
    bool qkNorm{}, betaSigmoid{}, gate{}, dtBias{}, allowNeg{}, exp2{}, outputBeta{};
};
inline bool DecodeFrontFeatures(int64_t encoded, FrontFeatures &out)
{
    constexpr int64_t known = 7 | kFrontQkNormBit | kFrontBetaSigmoidBit | kFrontGateBit |
        kFrontDtBiasBit | kFrontAllowNegBit | kFrontExp2Bit | kFrontOutputBetaBit;
    if (encoded < 2 || (encoded & ~known) != 0) return false;
    out.phase = (encoded & 7) / 2;
    out.layout = encoded & 1;
    out.qkNorm = (encoded & kFrontQkNormBit) != 0;
    out.betaSigmoid = (encoded & kFrontBetaSigmoidBit) != 0;
    out.gate = (encoded & kFrontGateBit) != 0;
    out.dtBias = (encoded & kFrontDtBiasBit) != 0;
    out.allowNeg = (encoded & kFrontAllowNegBit) != 0;
    out.exp2 = (encoded & kFrontExp2Bit) != 0;
    out.outputBeta = (encoded & kFrontOutputBetaBit) != 0;
    return out.phase >= 1 && out.phase <= 3 && (!out.dtBias || out.gate) &&
        (!out.allowNeg || out.betaSigmoid) &&
        (out.phase == 1 || !(out.betaSigmoid || out.gate || out.dtBias || out.allowNeg || out.outputBeta));
}
inline bool DecodeFrontMode(int64_t encoded, int64_t &phase, int64_t &layout, bool &qkNorm)
{
    FrontFeatures features{};
    if (!DecodeFrontFeatures(encoded, features)) return false;
    phase = features.phase;
    layout = features.layout;
    qkNorm = features.qkNorm;
    return true;
}

struct FrontHoSharedLayout {
    uint64_t w, u, g, decay, lastExp, bytes, columns;
};
inline bool MakeFrontHoSharedLayout(uint64_t batch, uint64_t heads, uint64_t tokens,
                                   uint64_t chunks, uint64_t bt, uint64_t valueDim,
                                   bool prepared, FrontHoSharedLayout &out)
{
    auto mul = [](uint64_t x, uint64_t y, uint64_t &z) {
        if (y != 0 && x > UINT64_MAX / y) { return false; }
        z = x * y; return true;
    };
    uint64_t rows = 0, rowBytes = 0, count = 0, bytes = 0, offset = 0;
    if (batch == 0 || heads == 0 || tokens == 0 || chunks == 0 ||
        !mul(batch, heads, count) || !mul(count, tokens, rows) ||
        !mul(rows, 2, rowBytes)) { return false; }
    auto append = [&offset](uint64_t value) {
        if (value > UINT64_MAX - 511) { return false; }
        const uint64_t aligned = (value + 511) / 512 * 512;
        if (offset > UINT64_MAX - aligned) { return false; }
        offset += aligned; return true;
    };
    if (!mul(rowBytes, bt, bytes) || !append(bytes)) { return false; }
    out.w = offset;
    if (!mul(rowBytes, 128, bytes) || !append(bytes)) { return false; }
    out.u = offset;
    if (!mul(rowBytes, valueDim, bytes) || !append(bytes)) { return false; }
    out.g = offset;
    if (!mul(rows, 4, bytes) || !append(bytes)) { return false; }
    out.decay = offset;
    if (prepared && (!mul(rows, 4, bytes) || !append(bytes))) { return false; }
    out.lastExp = offset;
    if (prepared) {
        if (heads > UINT64_MAX - 7) { return false; }
        const uint64_t stride = (heads + 7) / 8 * 8;
        if (!mul(batch, chunks, count) || !mul(count, stride, count) ||
            !mul(count, 4, bytes) || !append(bytes)) { return false; }
    }
    out.bytes = offset;
    out.columns = offset / rowBytes + (offset % rowBytes != 0);
    uint64_t capacity = 0;
    return out.columns <= INT64_MAX && mul(rowBytes, out.columns, capacity) &&
        capacity <= INT64_MAX && capacity >= offset;
}

struct QkNormOutputLayout {
    uint64_t qHat, kHat, qRstd, kRstd, frontColumns, frontBytes, gElements;
    uint64_t betaEff;
};
inline bool MakeFrontFeatureOutputLayout(uint64_t batch, uint64_t hk, uint64_t hv, uint64_t tokens,
                                   uint64_t qkvLayout, const FrontHoSharedLayout &front,
                                   bool qkNorm, bool outputBeta,
                                   QkNormOutputLayout &out)
{
    out = {};
    auto mul = [](uint64_t x, uint64_t y, uint64_t &z) {
        if (y != 0 && x > UINT64_MAX / y) return false;
        z = x * y; return true;
    };
    auto append = [](uint64_t &offset, uint64_t bytes) {
        if (bytes > UINT64_MAX - 511) return false;
        bytes = (bytes + 511) / 512 * 512;
        if (offset > UINT64_MAX - bytes) return false;
        offset += bytes; return true;
    };
    uint64_t rows = 0, keys = 0, rowBytes = 0, keyBytes = 0, rstdBytes = 0;
    uint64_t offset = 0;
    if (!batch || !hk || !hv || !tokens || hv % hk || qkvLayout > 1 ||
        (qkNorm && qkvLayout == 1 && hk > UINT32_MAX / (128 * 2)) ||
        !mul(batch, hv, rows) || !mul(rows, tokens, rows) || !mul(rows, 2, rowBytes) ||
        !mul(batch, hk, keys) || !mul(keys, tokens, keys) ||
        !mul(keys, 128 * 2, keyBytes) || !mul(keys, 4, rstdBytes) ||
        !mul(rowBytes, front.columns, offset) || offset > UINT64_MAX - 511) return false;
    // 追加在旧 parent 完整逻辑容量之后，不借用旧 shape 外的内存。
    if (qkNorm) {
        offset = (offset + 511) / 512 * 512;
        out.qHat = offset;
        if (!append(offset, keyBytes)) return false;
        out.kHat = offset;
        if (!append(offset, keyBytes)) return false;
    }
    out.frontColumns = offset / rowBytes + (offset % rowBytes != 0);
    if (out.frontColumns > INT64_MAX || !mul(rowBytes, out.frontColumns, out.frontBytes) ||
        out.frontBytes > INT64_MAX) return false;
    offset = 0;
    uint64_t gBytes = 0;
    if (!mul(rows, 4, gBytes) || !append(offset, gBytes)) return false;
    if (qkNorm) {
        out.qRstd = offset;
        if (!append(offset, rstdBytes)) return false;
        out.kRstd = offset;
        if (!append(offset, rstdBytes)) return false;
    }
    if (outputBeta) {
        out.betaEff = offset;
        if (!append(offset, gBytes)) return false;
    }
    if (offset > INT64_MAX) return false;
    out.gElements = offset / sizeof(float);
    return true;
}
inline bool MakeQkNormOutputLayout(uint64_t batch, uint64_t hk, uint64_t hv, uint64_t tokens,
                                   uint64_t qkvLayout, const FrontHoSharedLayout &front,
                                   QkNormOutputLayout &out)
{
    return MakeFrontFeatureOutputLayout(batch, hk, hv, tokens, qkvLayout, front, true, false, out);
}

struct HoSharedLayout {
    uint64_t vNew, bytes, columns;
};
inline bool MakeHoSharedLayout(uint64_t batch, uint64_t heads, uint64_t tokens,
                               uint64_t chunks, uint64_t valueDim, HoSharedLayout &out)
{
    auto mul = [](uint64_t x, uint64_t y, uint64_t &z) {
        if (y != 0 && x > UINT64_MAX / y) { return false; }
        z = x * y; return true;
    };
    uint64_t bh = 0, rows = 0, rowBytes = 0, hBytes = 0, vBytes = 0;
    if (!batch || !heads || !tokens || !chunks || !valueDim ||
        !mul(batch, heads, bh) || !mul(bh, tokens, rows) ||
        !mul(rows, 2, rowBytes) || !mul(bh, chunks, hBytes) ||
        !mul(hBytes, 128, hBytes) || !mul(hBytes, valueDim, hBytes) ||
        !mul(hBytes, 2, hBytes) || !mul(rowBytes, valueDim, vBytes) ||
        hBytes > UINT64_MAX - 511 || vBytes > UINT64_MAX - 511) { return false; }
    out.vNew = (hBytes + 511) / 512 * 512;
    vBytes = (vBytes + 511) / 512 * 512;
    if (out.vNew > UINT64_MAX - vBytes) { return false; }
    out.bytes = out.vNew + vBytes;
    out.columns = out.bytes / rowBytes + (out.bytes % rowBytes != 0);
    uint64_t capacity = 0;
    return out.columns <= INT64_MAX && mul(rowBytes, out.columns, capacity) &&
        capacity <= INT64_MAX && capacity >= out.bytes;
}
} // namespace GDN
#endif
