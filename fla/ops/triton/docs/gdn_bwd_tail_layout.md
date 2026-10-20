# GDN backward 私有 QK 尾部布局合同

## 调用边界

本说明对应 [`_l2norm_bwd_qk_add`](../triton_core/_gdn_bwd_tail.py)；该函数随 wheel 安装为
`fla_npu.ops.triton.triton_core._gdn_bwd_tail._l2norm_bwd_qk_add`。
P154 的 ATK 组合链先执行 `chunk_bwd_dqkwg`、`prepare_wy_repr_bwd` 和 reverse scan，
再在 `norm=True` 时调用本函数。该组合链不等同于单体
`ascendc.chunk_gated_delta_rule_bwd_finalize` 的 ACLNN 入口。

## 输入与输出

- `q`、`dq`、`k`、`dk0`、`dk1` 的 shape 必须相同；矩阵同为 BF16 或同为 FP16。
- 矩阵至少二维，最后一维满足 `0 < D <= 512`；两个 FP32 rstd 的元素数均等于矩阵行数。
- 七个输入必须位于同一 NPU。原 shape、dtype、device 校验和异常消息保持不变，校验先于布局复制。
- 接受满足上述条件的逻辑 strided 输入视图；wrapper 内部对七个输入各调用 `contiguous()`，
  后续 view 和 kernel 统一使用转换后的局部引用。输入不被原位改写，不做 dtype 转换。
- 连续输入（包括仍为连续布局的非零 storage offset 视图）沿用 `contiguous()` 的原 Tensor 快路径。
- 返回两个与 q/k 分别同 shape、dtype、device 的新连续 Tensor；不保留输入 strides，不与输入共享输出存储。
  此 Python API 没有 caller-provided out 参数，因此不新增 out 签名或 ViewCopy。
- 零行在原校验后返回空连续 Tensor，不启动 QK kernel；这不表示整个 DQ/WY 组合链支持所有空 shape。

内部 kernel 仍只处理连续二维行和连续 rstd。所有复制在现有 raw-stream drain 前排队，
使 torch_npu 生产者及复制完成后再提交 Triton。原 `dk0 + dk1` 的存储 dtype 舍入、
FP32 norm 计算顺序、BT/BD、分核、stream drain 和 fallback synchronize 均不改变。

## 其它消费者

DQ/WY 的 ACLNN L2 已分别处理其输入连续化；这些转换是各调用内部的临时值，
并不会更新组合链保留的原 q/k 引用，所以私有 QK 尾部仍需自己的布局边界。
本次不移动组合链、不修改 DQ/WY、reverse scan、单体 Finalize 或公共 ctypes/Stable ABI。
reverse scan 已传入 dg0/dg1 的实际 strides 并分配连续 FP32 输出。

## 验证状态与范围

已测独立源 `50e3291a478040cb410443d80c0b564a7496a3d4` 的 500 条 ATK、布局检查、
MSS memcheck 与三核心性能结果见
[`P154 送审说明`](../../ascendc/gdn/chunk_gdn_bwd/docs/p154_validation.md)。
本送审分支移植到新 main 后尚未重新完成 NPU 构建和硬件验证；不能将已测快照结果
自动升级为送审 head 通过。封存证据分别核对以下边界：

1. 连续输入相对 7fb 的五输出 bitwise 回归，以及原 ATK 精度范围回归。
2. 真实入口上的非连续 q/k/rstd、transpose、slice2、offset 对照；扩展到其余实际输入和组合视图，
   五输出逐项比较，确认输入及非目标存储不被改写。
3. 私有尾部单独覆盖非连续 dq/dk0/dk1、FP16/BF16、空行、同设备与 dtype 拒绝边界。
4. MSS 实际原生日志及真实父子 wait、连续快路径性能与非连续布局复制开销。

不得把本地 stub 结果、旧 7fb 的证据或单体 Finalize 的接口文档当作本候选设备验收。
