// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. 仅A2私有O阶段携带gate指数底，不改变公共CATLASS策略。
#pragma once
#include <type_traits>
namespace GDN {
template <class Base, bool Exp2> struct OGateInputType : Base {
    static constexpr bool kUseExp2 = Exp2;
};
template <class T, class = void> struct OGateUsesExp2 : std::false_type {};
template <class T> struct OGateUsesExp2<T, std::void_t<decltype(T::kUseExp2)>>
    : std::bool_constant<T::kUseExp2> {};
} // namespace GDN
