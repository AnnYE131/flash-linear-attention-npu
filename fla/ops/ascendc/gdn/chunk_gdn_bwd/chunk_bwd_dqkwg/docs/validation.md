# P154 开发期验证计划

唯一父 P152 `816ff1a8c938e201bd338f063cdcedbe1a4337d5`。本候选只扩两DQ host的合法varlen资格、修helper注释、补两整数driver项和两docs，不继承P153四头或P149/P151驻留。当前P154只有本地源码/静态证据，尚未编译C++、构建all6或取得硬件/精度/性能结果。P152 C01首项实测通过不代表其全20或精度已通过；P152 C02裁决及全轮关闭由root另记。

参考角色CHUNK_INDEPENDENT、版本V1、来源commit `a78efa9512dfa64c20b2a58a75e84e88a054d694`，仅复用完整输入下独立task与资源闭环核对方法，不引入参考的Stage/tile/buffer/runtime。

## 本地与固定 CPU driver

- 两选择点仅 `isVarLen==0 && T%BT==0`→`isVarLen!=0 || T%BT==0`；BT原64/128校验、formal非950 guard、最终varlen/numChunks时机保持父。helper评分/约数枚举/防溢出正文逐字父，其余父tracked文件逐字不变，六文件精确逆映射由owned回执绑定。
- `tests/hk_partition_cpu.cpp` 保留原18项原文，新增 `varlen_full_chunks`（N256/HK16/HV128/C20/parent1/allowtrue→P8）及 `varlen_tail_chunks`（N4/HK4/HV8/C3/parent1/allowtrue→P2），合计20项。它们直接调用实际无SDK helper，仅核整数选择；driver不接收seq值，名称不代表已执行尾块浮点或device测试。原 `varlen_disabled` 保留为allow=false的父回退控制。
- 本地未运行C++编译器；预期stdout只是准备资产。固定refs后由唯一runner从Git blob物化driver/helper/父schedule三文件到独立CPU目录，`g++ -std=c++17 -O2 -Wall -Wextra -Werror`一次编译运行，保留精确argv/三SHA/stdout/wait；首次失败原样留存，CPU未过不启动NPU。

## 后续门禁（待 root 另 GO）

当前唯一设备轮仍P152；P154不提交/同步/设备，等待P152实际裁决、完整关闭及root核验。root若确认需要才固定新源和DQ原三头/profile，再独立all6/native/输入/工具准入，同卡fresh FLA/P90/P154/P144的C01/C02/C20完整冻结尾段TaskSUM各严格<2.3×H20，等号拒绝；任一miss完整关闭后淘汰。

核心全过再补全20，必要CPU/Windows原件封存及root核验后固定50、两角色100DG真FP64、同轮CPU saved20；不跑500。固定资产实际覆盖以本轮manifest/摘要为准，不声称完整dtype/norm笛卡尔覆盖。保留全部慢/失败样本、普通Event100、独立Task全部CSV/末10/span/gap、实际native/源/tools/输入/设备与真实wait。共享如实标记，既有H20原件只读，不从静态头数推时延或因果。若P152已全目标及精度通过，停止无用P154硬件尝试。
