# Kimi-K3：理论计算与以小验大

本例整理自用户《MoE 模型显存建模与资源估算》的实践记录。它保留方法和测量来源，不代表对当前公开模型或当前 NPU 软件的重新验收。模型/框架/算子 commit 及完整原始日志未随文档保存，新环境需重新核对。以下 GiB 数值默认按展示精度四舍五入。

## 结构和输入

总参数按约 2.8×10¹² 记账；93 层，其中 92 个 MoE 层、69 个 KDA 层、24 个 MLA 层。H=7168，KDA N=96、D=128；E=896，K=16；专家输入输出宽度 3584、中间宽度 3072；共享专家总中间宽度 6144。

BF16 权重/激活，16 B/param 的训练状态格式，逐层 Full Recompute，B=1、输入 1024、输出 512、S=1536，TP1/CP1。每节点 16 个设备，每设备容量 64 GiB，设备与 die 口径按本例记录。

\[
p_e=3\times3584\times3072=33\,030\,144,
\quad P_E=92\times896\times p_e=2\,722\,740\,830\,208.
\]

\(P_D=2.8\times10^{12}-P_E=77\,259\,169\,792\)。公共参数由约数总量相减，不能因显示完整整数就称为精确参数统计。

## 方法一：理论估算

总状态 \(16\times2.8\times10^{12}/2^{30}\approx41723.251\) GiB。公共参数按 93 层均摊、临时系数取案例的 8 B/param 时：

\[
M_{param,tmp}\approx
\frac{8(77.26\times10^9/93+896\times33\,030\,144/EP)}{2^{30}}
\approx6.190+220.5/EP\quad\mathrm{GiB}.
\]

这是平均层粗估；FSDP 包装组、预取与大型 embedding/输出头可能使真实峰值不同。

\[
M_{ckpt}=93\times2\times1536\times7168/2^{30}\approx1.907227\quad\mathrm{GiB}.
\]

原文分析的公开 KDA/FLA 训练路径使用 BF16、全秩输出门控、块长 C=64、无初始循环状态及默认重算策略。该路径单层保存量为：

\[
A_{KDA,saved}\approx2BS(H+D+10ND+2NC)+4BS(5N).
\]

10ND 包括三份短卷积输入、归一化后的 Q/K 和 V、衰减门控输入、核心输出、输出门控及输出投影输入；2NC 是两份块内矩阵。FP32 的 5N 包括两份 beta 门控量、Q/K 的两份归一化统计及输出归一化统计。权重引用和共享存储不重复计入。原分析的默认路径会重算部分中间量；改变内核或重算开关会改变这份清单。

\[
A_{KDA,saved}\approx
\frac{2\times1536(7168+128+10\times96\times128+2\times96\times64)+4\times1536\times5\times96}{2^{30}}
=0.410339\quad\mathrm{GiB}.
\]

这只是单层主要保存张量，额外反向/重算临时量另算；NPU 实际内核未由本案例核实。Full Recompute 下不乘 69，共享 checkpoint 入口只计一次。

LatentMoE 的结构项为 \(0.3453c_{MoE}\) GiB；展开 Q/K/V 并将 V 补齐至 D=192 的 FA 路径为 \(0.1260c_{FA}\) GiB。系数待校准，不能将二者默认取 1 当作完整活动峰值。

\[
F_{min}\approx\left\lceil\frac{41723}{64\eta-8.097-220.5/EP-M_{act,live}}\right\rceil.
\]

缺少活动量时，原文另用“整块分母取 80% 容量”的粗估，\(\lceil41723/(64\times0.8)\rceil=815\) 卡；该方法不再重复扣除临时量和 checkpoint，815 也不是已验证配置。

## 方法二：五步外推

1. **减层减专家并测量。** 保留前 12 层（9 KDA、3 MLA，首层 Dense、11 MoE），896 选 16 改为 16 选 8。16 卡 FSDP16/EP16/eFSDP1/TP1/CP1，B=1、S=1536，逐层重算，已单机跑通。用户提供 allocated=19.79、reserved=24.84375；原分析核对对应 verl 输出按 1024³ 除，单位为 GiB。两者不相加，残差使用 allocated。未包含此处未统计的设备外部分配。
2. **计算已知量。** 小模型约 15.60B 参数：专家约 5.81B、公共约 9.79B；公共包含词表/输出头约 2.35B、KDA/MLA 约 4.69B、其余约 2.75B。这是结构估算，非用户实测参数计数。状态 \(16\times15.60\times10^9/(16\times2^{30})\approx14.528632\) GiB，checkpoint \(12\times2\times1536\times7168/2^{30}=0.246094\) GiB。原文先圆整为 14.53+0.25=14.78 GiB。
3. **残差外推。** 按原文展示精度，\(R_s=19.79-14.78=5.01\) GiB；专家任务从 1536×8 增至 1536×16，故 \(R_l\approx10.02\) GiB。临时参数等留在残差内，不再次相加。若中间值不圆整则残差分别约 5.015274 和 10.030548 GiB，差异来自舍入，均为近似外推。
4. **估算卡数。** 预算 64×0.95=60.8 GiB，\(F_{min}=\lceil41723/(60.8-1.907-10.02)\rceil=854\)。选择满足每卡一专家的 896 卡网格。
5. **推导配置。** 保持每卡一专家、eFSDP1、TP1/CP1，得到 FSDP896/EP896。\(\widehat M_{train}=41723/896+1.907+10.02\approx58.5\) GiB。它是 allocated 口径外推，不是全尺寸实测峰值。

## 推理候选

小模型 TP16/EP16/DP1。全模型每卡一专家需 TP16/EP896/DP56；用户的当时 NPU `moedispatchv2` 部署要求单实例不跨超节点，896 卡不可行，448 卡实例可放入一个超节点。因此尝试每卡两个专家 TP16/EP448/DP28。这个限制仅记录为该环境条件；未给出超节点精确容量，计算示例用 448 卡作为已确认可用的单实例规划上限，不冒充硬件理论上限。

按公共权重全部均匀 TP 的简化假设：

\[
M_{weight}=\frac{2P_E/448+2P_D/16}{2^{30}}\approx20.3145\quad\mathrm{GiB}.
\]

经验常驻量 0.15 GiB/层，固定框架量 10 GiB；合计 23.95 GiB。TP16、每 DP 副本 16 序列的 FP32 KDA 状态取材料值 0.43 GiB。这些不是通用常数。

MLA 按该缓存格式，每 token 每 die：

\[
24\times(512+64)\times2=27648\ \mathrm{B}=27\ \mathrm{KiB}.
\]

它已经包含 24 层的压缩状态和额外缓存维度，不能再乘 K/V、head 数或再次除 TP。16×1536 token 需要 0.6328125 GiB。静态合计约 45.3273 GiB，物理余量约 18.6727 GiB；未提供完整 prefill/decode 动态峰值，因此仅判断静态预算及已知拓扑满足。

每卡四专家 EP224/TP16/DP14 时，相同开销假设下静态约 56.648 GiB，物理余量约 7.352 GiB；余量更小，需动态实测。不能仅据此断言必然 OOM。若更换 EP 引起缓冲变化，应逐候选更新常驻和动态预算。

## 复算与来源

[计算输入](kimi-k3-calculations.json)沿用原文外推的展示精度；工具输出不会把静态推理量标为完整峰值。输入中相同常驻量用于不同候选是明确的案例假设。

原文的模型来源为 [配置](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)与[实现](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py)。KDA 张量清单参考 [chunk.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk.py)、[短卷积](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/causal_conv1d/ops.py)、[输出归一化](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/norm/fused_norm_gate.py)。这些链接未固定历史 commit，复用时应核对目标版本。

测量与 NPU 部署条件来源于用户实践记录；没有编造完整日志、实验日期或源码版本。
