# no-burn-heart-skills

将实际开发与排障经验整理成可复用的 Codex skills。

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

### 使用前提

- Codex 能读取目标 GitHub Actions 的元数据、日志和相关仓库代码。优先使用 GitHub 连接器；能力不足时使用已认证的 `gh` 或 GitHub API。
- 日志或 artifacts 已过期、访问受限时，可以提供已下载的日志和配置；报告会明确剩余证据缺口。
- 分析和生成复现包不要求当前电脑有 NPU。执行完整 nightly 复现需要符合该 job 要求的 Ascend NPU 机器；具体型号、节点数、卡数、驱动、镜像和输入要求会写入当次生成的 README。

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

Nightly 默认复现失败版本的完整流程，保留原始卡数、训练步数、batch、数据处理和 baseline 检查。候选修复或缩小规模的 smoke test 使用独立模式。

本仓库包含**工作流指令及复现包规范**；具体运行脚本、环境包和诊断报告在每次排查时依据对应 run 生成，不是一份适用于所有 CI 的固定 NPU 镜像或预制复现包。

### 证据与验证原则

- 使用日志中的实际 checkout 和最终配置，不把当前 main 或脚本默认值当成历史 run 的真实状态。
- 不把时间上最近合入的 PR 直接认定为根因；明确反例、证据强度与待验证假设。
- 相同镜像 tag、包版本或模型路径不代表内容完全一致；无法核验的历史环境信息会标明未知。
- 没有实际运行 NPU 时，只报告已完成的静态/本地检查，不声称已复现故障或已验证修复。
- 修复和提交仍遵循目标仓库贡献规范；skill 不代替人类审阅与测试。

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
