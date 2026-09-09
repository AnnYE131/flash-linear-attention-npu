# A2 Solve 迁移开发期验证

状态：设计已按当前调用、资源和数值边界自查；允许开始接入实验，尚未通过候选硬件门禁。
无公开ABI变化，不使用基线通过代替候选结果。性能和正式交付仍未放行。

## 固定来源与范围

- 目标基线：chw_a2_a5@86299afd，ATK适配后fa381c6f。
- 私有算法来源：f2fbb243，pipeline blob SHA256 9351fe4bac429373056d989db670702f50467b163d0ff5241654dc4f23994d3d。
- CATLASS 41bf90da，CANN 9.1.0-beta.1 / A2 910B3。
- 整体参考CHUNK_DEPENDENT；独立Solve参考CHUNK_INDEPENDENT V1/a78efa95，仅采用独立任务和grid-stride思想，
  不照搬A5的VF、owner head或UB写出模板。既有Solve的数学/flag协议保留。
- V2兼容保留项：整mega物理blockDim不变、A2中间GM relay不变；详见design。
- DTYPE_Q模板、H/WU/O、CPU FP64 golden、六ACLNN链、输入域和阈值冻结。
- 用户允许ATK26.7.8，仅运行环境覆盖最低版本；用户允许共享卡精度，不作为隔离性能证据。

## 追溯与门禁顺序

1. CPU：对照源码核对task/地址/尾块/分核、scratch、A2编译宏、旧新数学序列；
   每stage计算单独核对；grid-stride必须覆盖无任务、少于核数、尾批、环形复用。
2. 构建：同七算子/jobs8/无cache，基线527.860秒/27设备对象；记录新wall、模板对象和实际宏。
   比较DTYPE_Q分发合同，不以整个已编辑entry文件hash不变为条件。
3. 隔离部署核对wheel SHA及真实加载路径；最小ATK smoke。
4. 受影响stage需采集X/D16/D32/D64/A，按独立FP64公式和固定输入核验；
   原旧stage门禁只是来源证据，迁移后的地址/分发仍需复验。
5. 500×双模式×三实际种子；完整原始ATK输出，1500模式pair逐位有限值，
   每例q/k/v/g/beta实际hash不同；历史377原输入及50次原生DC。
6. 保留现有六MSS例，补16种dtype/BT/varlen/V组合与长任务。
   独立调试workspace记录实际task/slot/版本/ready/free/blockDim；无新同步，分core。
   先回收证明首轮/切换/回绕/尾部再关闭记录测普通精度和性能。
7. 两模型各训练/推理，固定输入与模式。计划预热10、每进程25次采样，3次交替A/B；
   汇报逐轮及汇总分布，目标kernel取chunk_gated_delta_rule_fwd完整MIX Task Duration。
   精确的新公开入口profile脚本在实验前固定，旧Phase6入口脚本不得直接当新版证据。
   性能若共享占用则只记录诊断数据，不给出确定提速结论。
8. 公开Example/ST、非A2编译分支影响核对，全部通过后再归档正式结论。

## 当前证据

迁移前3000原生ATK全部Pass、6300共同输出逐位有限值、实际三种子hash检查通过。
这些绑定fa381c6f，属于前置基线，不是新Solve通过。
CPU旧计划500+2模型过；宏探针确认COMPUTE_UNIT需完整Ascend910B1名称；
两者尚非真实新kernel执行证明。新候选结果待填。

## 实验变量和推翻条件

本轮变量：将私有Solve接入新主仓，配合V2的CG1 grid-stride；不得同时更改KKT/H/WU/O。
任何新精度/非有限/重复不一致、地址越界、超时、模板数量异常均先定位，
禁止放宽ATK阈值、换种子掩盖或修改独立标杆。
workspace显存增量和编译耗时必须随性能一起报告。
