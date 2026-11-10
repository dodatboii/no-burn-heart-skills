# 显存账本与理论估算

## 参数和单位

下列纯变量公式按字节计，数值报告转为 GiB。1 GiB=2³⁰ B，1 GB=10⁹ B，1 KiB=1024 B。容量 65536 MiB=64 GiB；界面标为 MB 时先确认其实际单位。统计设备时区分物理卡、die 和框架可见 device，容量、并行度、每节点数量必须使用同一种粒度。

相同 MoE 层的 routed-expert 参数：

\[
P_E=L_E E p_e,\qquad P_{total}=P_D+P_E.
\]

三矩阵 gated FFN 且输入输出宽度相同时，\(p_e=3H_{latent}H_e\)；有偏置、异形矩阵时按实际形状统计。共享专家、路由器和升降维属于公共参数。异构层用逐层求和，绑权存储只计一次。每 token 激活参数影响计算量，常驻权重仍按全部实际保存的参数计。

## 训练

\[
M_{train}(t)=M_{state}(t)+M_{param,tmp}(t)+M_{act}(t)+M_{comm/work}(t)+M_{runtime}(t).
\]

这是同一时刻的账本，峰值为该和随时间的最大值；相加各项独立峰值只是保守上界。

### 状态

\[
b=b_w+b_g+b_{master}+b_m+b_v,
\quad M_{state}=\frac{b_D P_D}{F_D}+\frac{b_E P_E}{EP\,eFSDP}.
\]

示例：BF16 权重和梯度、FP32 master 与两个 Adam 状态合计 16 B/param；FP32 梯度、量化优化器、offload、参数冻结等需要改账本。仅当有效分片数相同且状态格式一致时，简化为 \(bP_{total}/F\)。TP/PP 存在时先计算本 rank/stage 的参数，再按真实 FSDP 组分片，避免重复除法。

### 临时参数

\[
M_{param,tmp}\approx b_{tmp}\left(P_{D,group}+P_{E,group}/EP\right).
\]

group 是实际恢复参数的 FSDP 包装组。统计额外同时存在的 all-gather、梯度、精度转换和预取存储，去掉已在状态项内的重复存储；不能默认 b_tmp=8。整层均值仅作粗估，首层、embedding、输出头或多组预取可能决定峰值。

### 激活

先写保存张量的形状、dtype、生命周期、别名及重计算策略，再按唯一存储求和。通用结构式用于确定规模，系数 c 吸收保存份数与实现差异，必须注明校准依据：

\[
A_{FA}\approx c_{FA}b_{act}BS[H+ND+N_{kv}D],
\]

\[
A_{GDN}\approx c_{GDN}b_{act}BS[H+N_kD_k+N_vD_v+N_vC],
\]

\[
A_{MoE}\approx c_{MoE}b_{act}BS[H+H_s+E+K(H_e+H_{latent})].
\]

FA 避免保存完整 S×S 注意力矩阵，但保存输入、Q/K/V、输出及部分统计量。Q/K/V 维度不同时展开计算，不能硬套同一 D。MLA 训练若展开 Q/K/V，其激活不能按推理压缩缓存估计。

GDN/KDA 是 token 交互模块，不是在 FA 后额外串接一次。分块项由 \((BS/C)N_vC^2=BSN_vC\) 得到；具体实现是否保存块内矩阵、循环状态、FP32 门控或统计量，需检查 forward 和 backward。KDA 的细粒度门控与 GDN 不同，不能沿用同一份张量清单。

MoE 公共和共享部分按 BS 计；每个 token 分发 K 份任务，专家中间量随 BSK 增长。路由分数是否物化为 BS×E、容量填充、丢 token、负载不均和融合算子都会改变数值。Dense FFN 按实际门控和中间宽度单独估算。

\[
M_{act,no\_recompute}\approx\sum_\ell A_\ell,
\qquad
M_{act,full\_recompute}\approx\sum_j A_{ckpt,j}+M_{act,live}.
\]

逐层保存一个相同形状输入时，checkpoint 简化为 \(L b_{act}BSH\)。其他分段方式按实际边界数与存储计算；嵌套 checkpoint 和 view 按唯一存储去重。活动项取反向重算期间同时存活的最大额外张量，包括相关梯度及临时量。整层重算后，Attention 的保存量可能一直保留到 MoE 反向完成；不能默认取 max(Attention,MoE)。不用总层数乘活动分段。检查输出 logits、loss、首层 Dense 及其是否纳入 checkpoint。

## 推理

\[
M_{infer}(t)=M_{weight}+M_{resident}+M_{cache}(t)+M_{temporary}(t).
\]

在 routed experts 仅按 EP 分配、公共权重分为 TP 分片及复制部分的布局：

\[
M_{weight}=b_E P_E/EP+b_D(P_{D,TP}/TP+P_{rep}).
\]

量化时另计 scale、zero point、对齐或高精度残留。若专家内部也有 TP、PP 或冗余专家，按实际存储布局重算。

常驻量可由逐层 workspace/缓冲和框架固定量组成；是否常驻由实际生命周期决定，不应在 resident 与 temporary 中重复出现。图池、KV 池预留也按真实分配归类。

MHA/GQA 在 K/V 同宽同精度时：

\[
M_{KV}=2b_{KV}L_{local}T_{cached}N_{kv,local}D_{head}.
\]

MLA 按每 token 实际保存的压缩维度求和，不再额外乘 K/V 的 2 或 head 数。缓存复制还是分片由内核决定，不能统一除 TP。混合模型为：

\[
M_{cache}=M_{KV}+N_{seq}M_{recurrent,per\ sequence}+M_{metadata}.
\]

T_cached 是一个 DP 副本的总缓存 token 数，包含输入和已生成输出；N_seq 是该副本活跃序列数。每条序列的循环状态已包含本卡所有相关层，主要随并发增长。另查卷积历史等状态。实际缓存池可能高于需求量；已计池容量后不再把池内已用量加一次。

Prefill 临时激活主要取决于本轮执行 token 数（包括 chunked prefill 设置），decode 取决于活跃请求和算子路径。使用对应阶段实测扣除同口径已知项，不把训练残差代入推理。

## 卡数与 RL

设每卡可用预算 \(C_{allow}=\eta C\)。状态均匀分片、其他开销给定时：

\[
F_{min}=\left\lceil\frac{M_{state,total}}{C_{allow}-M_{param,tmp}-M_{act}-M_{other}}\right\rceil.
\]

分母非正表示该假设下加卡也不能解决。临时量等依赖 EP 时需逐个候选重算，不能把一个候选的下界视为全局下界。80% 容量直接留给状态是另一种粗估，不能再重复扣除同一批开销。按整机、专家分配、head 切分、PP 层分配、通信范围和可用卡数筛选。

Colocate 串行阶段且非活动角色已卸载或计入驻留账本时，RL 峰值可取各阶段总占用的最大值。`async` 的名字不证明训推计算重叠，需查看调度与进程布局。真正重叠时计算重叠期间总占用并去重；Separate 的训练与推理集群分别核算。初始化、优化器首次创建、权重同步也可能成为峰值，按任务需要验证。

## 实现核验入口

- [PyTorch FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)：分组、恢复、预取。
- [PyTorch Checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html)：重计算实际行为。
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)、[FLA](https://github.com/fla-org/flash-linear-attention)：保存张量与重算。
- [vLLM EP](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)：以目标版本布局为准。

这些是查阅入口；分析具体运行时应引用实际版本或 commit，不能用最新 main 代替历史运行环境。
