// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. H/O 沿用的顶层资源类型；Prepare 使用独立资源拥有者。
#ifndef GDN_ARCH22_HO_RESOURCES_H
#define GDN_ARCH22_HO_RESOURCES_H

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"

namespace GdnRawFront {
using namespace AscendC;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using Resource = Catlass::Arch::Resource<Catlass::Arch::Ascend950>;
#else
using Resource = Catlass::Arch::Resource<Catlass::Arch::AtlasA2>;
#endif

} // namespace GdnRawFront

#endif // GDN_ARCH22_HO_RESOURCES_H
