# A2/A3 三段式内部入口

DAV_2201 入口只调度 Prepare、H、O 三次提交。Prepare 的唯一计算实现位于
`chunk_gated_delta_rule_fwd_prepare/op_kernel/arch22/chunk_gated_delta_rule_fwd_prepare.h`，
与该独立算子的 A5 `arch35` 实现并列；这里不保留旧 Cumsum/KKT/Solve/WU 计算路径。

`pipeline/fwd_ho.h` 和 `stages/fwd_h`、`stages/fwd_o` 保留 H/O 实现。`stages/recompute_wu`
仅保留既有 host tiling 使用的元数据定义，不包含 WU kernel。私有 tiling 字段布局暂时保持，
未再使用的旧前段字段置零，低层 phase0 明确拒绝。既有公开 aclnn/Python 调用签名保留；算子 IR 增加共享端口，
Torch 注册增加 `.features` 重载，host、OPP 与适配产物需要配套。

各平台及 ABI 兼容性按仓库门禁验证；目录组织本身不代表性能收益或 A3 硬件验证通过。
