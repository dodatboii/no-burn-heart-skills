# no-burn-heart-skills


## Skills

| Skill | 用途 |
| --- | --- |
| [verl-ci-diagnose](skills/verl-ci-diagnose/SKILL.md) | 排查 verl PR CI 和 nightly_ci 失败、精度回退，对比历史成功运行并定位可疑 PR；nightly 额外交付 Ascend NPU 复现环境包。 |

## verl-ci-diagnose

提供失败 CI 的链接，Codex 按以下流程开展排查：

1. **分析失败运行**：读取 run/job 元数据、完整日志和相关产物，确定失败步骤、原始异常、实际 checkout SHA 与最终启动命令。
2. **寻找成功对照**：核验用户提供的成功 run，或向前查找同 workflow、job/matrix 的成功运行，建立回归区间。
3. **对比与归因**：比较代码、关联 PR、训练配置、镜像、依赖、硬件、模型、数据和 baseline；区分已确认根因、高度怀疑与待排查假设。
4. **交付报告**：给出证据、反例、修改建议和控制变量验证方案。
5. **按任务交付下一步**：nightly 生成完整 NPU 复现包；PR CI 给出建议后询问用户如何继续，已有明确授权则执行相应后续工作。

### 安装

在 Codex 中发送以下内容，让内置的 skill-installer 从本仓库安装：

```text
$skill-installer
请安装这个 skill：
https://github.com/dodatboii/no-burn-heart-skills/tree/main/skills/verl-ci-diagnose
```

安装后在下一轮对话中调用 `$verl-ci-diagnose`。如果没有被识别，可以重启 Codex 后重试。有关 skill 的发现与安装机制，参见 [OpenAI 官方说明](https://learn.chatgpt.com/docs/build-skills)。

如果已安装同名 skill，更新时明确告诉 Codex“从该仓库更新已安装的 verl-ci-diagnose”，并说明是否需要保留本地改动。

### 示例：PR CI 失败

将下面的 `RUN_ID` 替换为实际 run ID；也可以直接提供指向具体 job 的完整链接。

```text
$verl-ci-diagnose
帮我排查这个 PR CI：
https://github.com/verl-project/verl/actions/runs/RUN_ID

请对比之前成功的同类 job，定位可疑改动，给出分析报告和修改建议，然后询问我下一步怎么做。
```

### 示例：nightly 精度回退

```text
$verl-ci-diagnose
这次 nightly_ci 的精度明显回退：
失败：https://github.com/verl-project/verl/actions/runs/FAILED_RUN_ID
之前成功：https://github.com/verl-project/verl/actions/runs/GOOD_RUN_ID

请对比两次实际执行环境、代码和配置，分析可能涉及的 PR，并给出修改建议。
同时生成可在 Ascend NPU 上解压使用、完整模拟这个 nightly job 的复现环境包和中文使用说明。
```

没有已知成功 run 时，只提供失败链接即可，skill 会尝试向前查找。若硬件、网络、模型缓存或数据路径已有约束，可以一并说明。

### 交付内容

| 场景 | 交付 |
| --- | --- |
| 所有 CI 排查 | 故障摘要、成功/失败对照、候选 PR 与证据、修改建议、验证方案、已完成验证和限制。 |
| PR CI | 完成上述报告后询问下一步；明确授权后再进行对应修复或提交工作。 |
| Nightly CI | 上述报告，以及复现目录、压缩包、中文 README、环境与版本清单、预检、宿主/容器运行入口、环境采集及结果回传说明。 |


## 目录结构

```text
.
├── README.md
└── skills/
    └── verl-ci-diagnose/
        ├── SKILL.md
        ├── agents/
        │   └── openai.yaml
        └── references/
            └── nightly-npu-repro.md
```

- [`SKILL.md`](skills/verl-ci-diagnose/SKILL.md)：触发范围、排查流程、报告要求与后续动作。
- [`nightly-npu-repro.md`](skills/verl-ci-diagnose/references/nightly-npu-repro.md)：NPU 复现包的环境等价性、文件交付、运行行为和验证规范。
- [`openai.yaml`](skills/verl-ci-diagnose/agents/openai.yaml)：skill 显示名称、简述和默认调用提示。

本工作流根据实际 verl CI 排查经验整理，使用了 AI 辅助编写。
