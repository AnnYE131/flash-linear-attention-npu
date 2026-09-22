# GDN 状态布局

GDN 的独立算子、融合副本和组合调用统一使用 NT-first h/dh：dense 为
`[B,NT,HV,K,V]`，packed 为 `[totalNT,HV,K,V]`。支持 `state_v_first` 时交换末两维。
NT 为每条序列的 chunk 数，totalNT 为各序列 chunk 数之和；h0/ht/dh0/dht 无 chunk 轴。

FwdH→FwdO、Dhu→dqkwg/Finalize 直接传递状态，内部需要 rank-5 时只补 B=1 视图。
旧 head-first 数据需要实际换轴，不能只 reshape；不能靠 NT=HV 的 shape 检测其内容。
token 布局、dtype、平台、gate 与可选状态支持域以各算子 API/README 为准。

实现检查与待执行设备矩阵见 [迁移记录](../../../../docs/architecture/h-dh-nt-first-contract.md)。
本轮完成源码迁移与离线检查，NPU 编译、精度、内存和性能验收待 P6。
