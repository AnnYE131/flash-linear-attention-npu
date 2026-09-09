add_op_to_compiled_list()
set(chunk_gated_delta_rule_fwd_depends
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare;fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h;fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o"
    CACHE STRING "Kernel source dependencies for chunk_gated_delta_rule_fwd" FORCE)

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_gated_delta_rule_fwd_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_gated_delta_rule_fwd ACLNNTYPE aclnn_exclude)
# COMPUTE_UNIT 使用完整型号；当前选项解析器不接受短名 ascend910b。
# 编译期隔离，避免 A2 同时实例化新旧 Solve；DTYPE_Q 分发保持不变。
add_ops_compile_options(
    OP_NAME ChunkGatedDeltaRuleFwd
    COMPUTE_UNIT Ascend910B1
    OPTIONS -DGDN_A2_TRITON_SOLVE=1
)
add_ops_compile_options(
    OP_NAME ChunkGatedDeltaRuleFwd
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/coefficient_generation
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/coefficient_generation/gated_delta_rule_solve_tri
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/operators/chunk_fwd_o/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/operators/recompute_w_u_fwd/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/chunk_kkt_solve_tri/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/chunk_scaled_dot_kkt/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/solve_tri/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../solve_tri/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/chunk_gated_delta_rule_fwd_h/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/chunk_fwd_o/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/recompute_w_u_fwd/op_kernel
)
