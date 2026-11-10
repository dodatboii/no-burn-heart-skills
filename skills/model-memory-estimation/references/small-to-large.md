# 以小验大的五步

## 1. 减层、减专家，尽量单机跑通

可减少层数、专家数及 top-k；保留层内宽度、attention 形状、词表、精度、优化器、算子和重计算分段。混合模型尽量覆盖每种层型及首层、输出头，不能只测轻量层后代表全模型。

记录每层 E/K、训练 FSDP/EP/eFSDP/TP/CP/PP、推理 DP/TP/EP、每卡 micro-batch、实际输入/输出或打包 token 数、推理并发和 prefill token 数、offload 与 cache 配置。每卡专家数是 E/EP，与 eFSDP 无关。

测量需要覆盖优化器状态创建后的完整训练步，以及所需的 prefill/decode 阶段。确认峰值计数器重置范围，异步设备操作同步后读取，取所有 rank 的最大值，保留阶段和 metric。allocated、reserved、设备总占用分别记录，不相加。没有实测时只提供方案，不填写假造峰值。

## 2. 计算小模型参数和已知开销

优先统计实际裁剪模型的参数；按保留层型重建时标为估算，尤其注意 embedding 与输出头不会随层数等比例缩减。

\[
M_{known,small}=M_{state,small}+M_{ckpt,small}+M_{other\ known,small}.
\]

只扣除属于所测口径、峰值时确实驻留的项。只扣状态与 checkpoint 时，其余项留在残差；单独扣除某项后，大模型必须相应重算并加回。

## 3. 扣除已知项并外推残差

\[
R_{small}=M_{measured,small}-M_{known,small}.
\]

若结果为负，先查单位、状态格式、offload、峰值阶段和参数统计，不截断成零继续外推。

在 TP1、CP1、每个 EP rank 输入不同的 BS 个 token、路由均衡时，一个 EP 组生成 EP×BSK 份任务，平均每卡为 BSK。保持每份任务的维度、精度、内核和重计算分段不变，且残差主要由专家激活贡献时：

\[
R_{large}\approx R_{small}\frac{B_{large}S_{large}K_{large}}{B_{small}S_{small}K_{small}}.
\]

这里 B 是每卡 micro-batch，不是全局 batch，也不乘梯度累积次数。TP/CP 或 token 分配变化时，用实际每卡专家任务数之比；不要再额外乘 E、EP 或层数比例。非均衡路由用最忙 rank 的任务量评估。

固定开销、临时参数和通信量未必遵循这个比例。能拆出的部分单独计算；不能拆出时将结果标为一阶外推。有条件可再测一个 batch、长度或 top-k 点检查增长趋势，不默认引入复杂回归。维度或内核大幅改变时，应重新测量基准。

## 4. 推导完整模型卡数

恢复完整模型的状态、checkpoint 和其他已知项。若状态均匀分片：

\[
F_{min}=\left\lceil\frac{M_{state,total}}{C_{allow}-M_{ckpt,large}-M_{other\ known,large}-R_{large}}\right\rceil.
\]

残差中已有的临时量不再重复添加。预算比例由用户目标、实测口径及剩余不确定性选定并写明；allocated 外推未覆盖的设备运行时与分配器余量仍需预算。没有满足条件的分母或网格时，报告哪个项限制了配置。

## 5. 推导并行配置并复核峰值

### 训练

优先保持已验证的每层每卡专家数、eFSDP、TP、CP 和重计算方式：

\[
EP_{large}=EP_{small}E_{large}/E_{small}.
\]

在两类状态均采用同一有效分片数的布局中，F=EP×eFSDP；TP1/CP1/PP1 的相应简单网格才可把 F 直接当总设备数。其他布局应展开进程组，检查维度重叠关系。

筛选卡数下界、整机粒度、专家分配和通信约束，逐候选代入：

\[
\widehat M_{large}=M_{known,large}+R_{large}.
\]

改变每卡专家数或 eFSDP 后，重新核算临时参数和通信；不自动继承原残差完全不变的假设。

### 推理

对公共 attention 在 TP 内切分、专家在 DP×TP 设备间分配的布局：

\[
EP=E/(\text{每卡专家数}),\quad DP=EP/TP,\quad W_{instance}=DP\times TP.
\]

先保持小模型 TP，尝试每卡 1、2、4 等可整除的专家数；检查实际后端是否支持该布局、单实例卡数和超节点通信限制。规则不覆盖所有后端，某些实现允许不均匀分配或额外专家复制，需要按其规则计算。

逐候选重算专家权重、公共权重、逐层驻留、固定框架开销、KV 与循环状态，再估 prefill/decode 动态量。不同 EP 的通信缓冲可能改变，不能只改变权重一项。若没有动态量，结论只能是“静态预算满足，余量为 X，动态峰值待验证”。

训练、prefill、decode 分别使用自己的测量基准和缩放依据。最后按实际 RL 调度检查阶段切换与重叠，给出可部署的候选及最小验证工作量。
