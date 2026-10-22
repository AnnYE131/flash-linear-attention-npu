/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_FRONT_HO_SHARED_LAYOUT_H
#define GDN_FRONT_HO_SHARED_LAYOUT_H
#include <cstdint>
namespace GDN {
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
