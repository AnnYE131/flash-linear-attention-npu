# A2 GDN backward P154 送审说明

## 验收口径与版本

关联 [issue #661](https://github.com/flashserve/flash-linear-attention-npu/issues/661)。
2026-10-21，提交方确认已与需求提出方对齐本轮验收口径，继续按既定 H20 对照提交本 PR。
此说明不改写 issue 正文，也不将没有执行的 A5 测量或公开入口测试标记通过。

已经执行硬件验证的独立源快照是
`50e3291a478040cb410443d80c0b564a7496a3d4`，旧 P154 为
`7fb346d366929e7916e71dc94ef7198c0eba2797`。送审分支在 upstream
`601bfe8cdb47e2e5a489964e7391497b9a095ac3` 上移植净改动，保留上游其余更新。
独立源没有上游提交祖先关系，因此不是直接推送整个实验历史。

上游已将 `stable_chunk.cpp` 拆分，WY 的 schema/适配函数/参数转发改动同步迁至
`stable_prepare_wy_repr_bwd.cpp`；清除合并造成的重复 `tempfile` import。
本地交接指针、P121/P122/P123 根目录实验结果不进入送审源码。
这些集成调整尚未重新完成送审 head 的 NPU 构建及硬件验证。

## 实现范围

1. DQKWG 的合法 dense/varlen 完整 HK 分区及负载均衡，保留非适用模式回退。
2. WY 可接收复用的 w/u，驻留 KKT 转换结果并合并 D 输入，减少重复搬运与计算。
3. 融合 DG 相加及反向分块扫描，并合并 dk 加法与 Q/K L2Norm 尾段。
4. 私有 QK Python 尾部将合法 strided 输入连续化，输出新连续 Tensor。
5. 独立 A2 Finalize 正确性路径及原有支持范围、构建临时目录选项。

本次含公共 WY ACLNN 入参和 Stable-ABI schema 变化，须配套重建 OPP/适配层/wheel，
不能混用旧版二进制。Python 新增 w/u 为可选关键字；C ABI/dispatcher 参数顺序变化须
Code Owner 检视，不能由 Python 默认值推定二进制兼容。

性能与 500 条 ATK 结果对应 DQ → WY → DG scan → Q/K 组合链，
不是单体 `chunk_gated_delta_rule_bwd_finalize` 入口的同等验收。

## 已封存硬件结果

环境为 A2 910B3、CANN 9.1.0.beta1。执行工具
`72349a1c46cefa0996ce68d07653f5495f819666`；安装 wheel SHA256 为
`44fe559b919b7526460a44d1d0329bf8d56850f7a5214493fbb1aecdf0cf969b`。

| 项目 | 结果与边界 |
| --- | --- |
| 精度 | 500/500；20 份原始 ATK Excel、2500 输出判定 |
| C01 | A2 TaskSUM/H20 = 1.991781229 |
| C02 | A2 TaskSUM/H20 = 1.921650737 |
| C20 | A2 TaskSUM/H20 = 1.967871414 |
| MSS | 16 连续 + 4 布局，memcheck；两个私有 QK 用例仅 dq/dk |
| 非连续 | 50 布局、6 基础用例及 CPU 180 输出检查通过，逐项标注组合链/私有 QK 边界 |
| 内存占用 | 12 组各 100 次，现状观察，未按 issue 额外显存预算判定 |
| 异常 DFX | 14 项 host guards 按预期拒绝，现状观察，不是完整公开 ABI 合同 |
| 特殊 DFX | 60 注入点分类观察 + 2 精确零梯度控制；未定义特殊值传播 oracle |

性能来自物理卡 7 共享测量，沿用用户授权的跨设备输入/beta 存储差异粗比。
普通 Event、设备跨度和任务 gap 均保留。上述三个 TaskSUM 比值不能称为端到端比值、
当前 A5 比值或全 20/120 shape 通过。新源未完成全 120 性能验收。

七类材料为四项 pass、三项 observed；默认审计退出 0，严格验收审计退出 2。
完整转测 ZIP 为 8,662,344 字节，SHA256
`3b231ac2bc2c5dd5a9c1eab71f9480848d193d12d1e9a4e7e1f8fbd2ed602eba`，
2051 个成员经字节数、SHA、CRC 核对，包含原始 ATK Excel 和各项日志、原始失败与中断。
输入/CPU golden/二进制/profile 大件另行封存，未加入源码仓库。

## 送审与合入状态

送审分支的 CPU 静态检查不能替代上述设备结果，也不能把旧快照结果自动标成新 head CI。
本轮先以 Draft 提交并关联 issue；当前 head 的 NPU CI、产品构建矩阵、Example/ST、
ABI 检视和仓库审批尚待完成。旧源、旧转测包和 H20 数字基线均保留。
