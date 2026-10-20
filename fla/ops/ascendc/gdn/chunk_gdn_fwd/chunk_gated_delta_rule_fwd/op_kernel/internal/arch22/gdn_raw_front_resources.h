// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. FrontBatch AIV 的单次 UB 分配和显式事件；不用于 AIC 或 H/O。
#ifndef GDN_ARCH22_RAW_FRONT_RESOURCES_H
#define GDN_ARCH22_RAW_FRONT_RESOURCES_H

#include "kernel_operator.h"
#include <type_traits>

namespace GdnRawFront {
using namespace AscendC;

__aicore__ inline uint32_t Align(uint32_t bytes) { return (bytes + 31) / 32 * 32; }
__aicore__ inline uint32_t Reserve(uint32_t &cursor, uint32_t bytes)
{
    uint32_t offset = cursor;
    cursor += Align(bytes);
    return offset;
}

struct KktLayout {
    uint32_t g, beta, score, out, typed, gate, row, bytes;
    __aicore__ inline void Init(uint32_t bt, uint32_t align, uint32_t outputBytes)
    {
        uint32_t cursor = 0;
        g = Reserve(cursor, align * 4);
        beta = Reserve(cursor, align * 4);
        score = Reserve(cursor, bt * align * 4);
        out = Reserve(cursor, bt * align * 4);
        typed = outputBytes == sizeof(float) ? cursor : Reserve(cursor, bt * align * outputBytes);
        gate = Reserve(cursor, 32 * align * 4);
        row = Reserve(cursor, 32 * 8 * 4);
        bytes = cursor;
    }
};

struct WuLayout {
    uint32_t input, beta, g, out, fp32, betaFp32, brcb, gFp32, bytes;
    __aicore__ inline void Init(uint32_t rows, uint32_t width, uint32_t inputBytes, bool gated)
    {
        uint32_t cursor = 0;
        input = Reserve(cursor, 2 * Align(rows * width * inputBytes));
        beta = Reserve(cursor, 2 * Align(rows * 4));
        g = gated ? Reserve(cursor, 2 * Align(rows * 4)) : 0;
        out = Reserve(cursor, 2 * Align(rows * width * inputBytes));
        fp32 = Reserve(cursor, rows * width * 4);
        betaFp32 = Reserve(cursor, rows * 4);
        brcb = Reserve(cursor, rows * 32);
        gFp32 = gated ? Reserve(cursor, rows * 4) : 0;
        bytes = cursor;
    }
};

class VectorResources {
    TBuf<TPosition::VECCALC> all_;
    TEventID mte2V_[2], vMte2_[2], vMte3_[2], mte3V_[2], mte3Mte2_;
public:
    TPipe pipe;
    KktLayout kkt;
    WuLayout vb, kbg;

    __aicore__ inline void Init(uint32_t bt, uint32_t btAlign, uint32_t k, uint32_t v,
                               uint32_t rows, uint32_t inputBytes)
    {
        kkt.Init(bt, btAlign, inputBytes);
        vb.Init(rows, v, inputBytes, false);
        // 原 KbgExp 实际使用 vbVecRow，保留该取槽和分块合同。
        kbg.Init(rows, k, inputBytes, true);
        uint32_t bytes = 69632; // leaf L32: f2048 + a32768 + prod32768 + row2048。
        if (kkt.bytes > bytes) bytes = kkt.bytes;
        if (vb.bytes > bytes) bytes = vb.bytes;
        if (kbg.bytes > bytes) bytes = kbg.bytes;
        pipe.InitBuffer(all_, bytes);
        // cumsum 同时在途的 ping/pong 需要两个 ID；其余阶段复用已闭合的 ID。
        for (uint32_t i = 0; i < 2; ++i) {
            mte2V_[i] = pipe.AllocEventID<HardEvent::MTE2_V>();
            vMte2_[i] = pipe.AllocEventID<HardEvent::V_MTE2>();
            vMte3_[i] = pipe.AllocEventID<HardEvent::V_MTE3>();
            mte3V_[i] = pipe.AllocEventID<HardEvent::MTE3_V>();
        }
        mte3Mte2_ = pipe.AllocEventID<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline ~VectorResources()
    {
        // 各阶段退出前已消费其全部 token；释放所有权本身不等待硬件事件。
        for (uint32_t i = 0; i < 2; ++i) {
            pipe.ReleaseEventID<HardEvent::MTE2_V>(mte2V_[i]);
            pipe.ReleaseEventID<HardEvent::V_MTE2>(vMte2_[i]);
            pipe.ReleaseEventID<HardEvent::V_MTE3>(vMte3_[i]);
            pipe.ReleaseEventID<HardEvent::MTE3_V>(mte3V_[i]);
        }
        pipe.ReleaseEventID<HardEvent::MTE3_MTE2>(mte3Mte2_);
    }

    template <class T> __aicore__ inline LocalTensor<T> View(uint32_t offset, uint32_t bytes)
    {
        auto tensor = all_.Get<uint8_t>()[offset].template ReinterpretCast<T>();
        tensor.SetSize(Align(bytes) / sizeof(T));
        return tensor;
    }

    template <HardEvent Direction> __aicore__ inline event_t Event(uint32_t slot = 0) const
    {
        if constexpr (Direction == HardEvent::MTE2_V) return static_cast<event_t>(mte2V_[slot]);
        if constexpr (Direction == HardEvent::V_MTE2) return static_cast<event_t>(vMte2_[slot]);
        if constexpr (Direction == HardEvent::V_MTE3) return static_cast<event_t>(vMte3_[slot]);
        if constexpr (Direction == HardEvent::MTE3_V) return static_cast<event_t>(mte3V_[slot]);
        if constexpr (Direction == HardEvent::MTE3_MTE2) return static_cast<event_t>(mte3Mte2_);
    }

    __aicore__ inline void BeginWuPhase()
    {
        // 空闲槽的可写 credit；未使用槽也由 EndWuPhase 消费，不留下孤立 token。
        for (uint32_t slot = 0; slot < 2; ++slot) {
            SetFlag<HardEvent::V_MTE2>(Event<HardEvent::V_MTE2>(slot));
            SetFlag<HardEvent::MTE3_V>(Event<HardEvent::MTE3_V>(slot));
        }
    }
    __aicore__ inline void EndWuPhase()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            WaitFlag<HardEvent::V_MTE2>(Event<HardEvent::V_MTE2>(slot));
            WaitFlag<HardEvent::MTE3_V>(Event<HardEvent::MTE3_V>(slot));
        }
    }
};

// Raw 特化没有队列对象；legacy 模板仍使用原 SDK 类型。
struct Unused {};
template <bool Raw, TPosition Position>
using QueueStorage = std::conditional_t<Raw, Unused, TQue<Position, 1>>;

template <bool Raw> class Buffer;
template <> class Buffer<false> {
    TBuf<TPosition::VECCALC> buffer_;
public:
    __aicore__ inline TBuf<TPosition::VECCALC> &Storage() { return buffer_; }
    template <class T> __aicore__ inline LocalTensor<T> Get() { return buffer_.Get<T>(); }
};
template <> class Buffer<true> {
    LocalTensor<uint8_t> buffer_;
public:
    __aicore__ inline void BindView(VectorResources *resources, uint32_t offset, uint32_t bytes)
    { buffer_ = resources->View<uint8_t>(offset, bytes); }
    template <class T> __aicore__ inline LocalTensor<T> Get()
    { return buffer_.template ReinterpretCast<T>(); }
};

template <bool Raw> class SolveStage;
template <> class SolveStage<false> {
    TPipe pipe_;
public:
    __aicore__ inline explicit SolveStage(VectorResources *) {}
    __aicore__ inline void BindBuffer(Buffer<false> &buffer, uint32_t bytes)
    { pipe_.InitBuffer(buffer.Storage(), bytes); }
};
template <> class SolveStage<true> {
    VectorResources *resources_;
    uint32_t cursor_ = 0;
public:
    __aicore__ inline explicit SolveStage(VectorResources *resources) : resources_(resources) {}
    __aicore__ inline void BindBuffer(Buffer<true> &buffer, uint32_t bytes)
    { buffer.BindView(resources_, Reserve(cursor_, bytes), bytes); }
};

template <bool Raw, HardEvent Direction>
__aicore__ inline event_t SolveEvent(VectorResources *resources)
{
    if constexpr (Raw) return resources->Event<Direction>();
    else return static_cast<event_t>(0);
}
} // namespace GdnRawFront
#endif
