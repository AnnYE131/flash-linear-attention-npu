# NT-first 地址迁移设计

接口依据：[api.md](api.md)。按仓库设计规则 V2，仅修改状态地址和 descriptor。

dense 状态基址为 `((b*NT+chunk)*HV+hv)*K*V`；packed 的 chunk 为全局 chunk 索引、b=0。
普通与 preload 的 scheduler 同步使用该地址，下一块状态步长是 `HV*K*V`。
首块初始化写入 sequence 的首个全局 chunk，末块继续走原 finalState 分支，不写越界的下一块。
finalState/initialState 没有 chunk 轴，地址保持不变。L2 的 V-first 转换仅交换末两维。

GDN 融合的普通、arch22、arch35 副本同步改写；内嵌 FwdO 同时读取新地址。
KDA 复用该 FwdH 的路径与自身 prepare/post_wu/finalize 状态地址同步更新。
workspace 总元素数不变；UB/L1/L0、任务分配、流水、事件、跨核同步和数值公式不变。
不引入额外搬运；末维转置仍按原路径执行。

离线检查真实地址表达式的多 batch、跨 chunk、packed 起点及 NT/HV 相等场景。
设备阶段必须覆盖普通/preload 的可达分支、初末态、尾块、两个 Python 后端和 fast launch。
当前未构建 OPP/wheel，不能据离线地址检查宣称 DMA 或设备精度通过。
