# Contributing to vime

[中文版](#开源协作范围说明)

Thank you for your interest in contributing to vime! We deeply appreciate every contribution from the community. Vime is derived from [slime](https://github.com/THUDM/slime), so the collaboration scope below preserves the upstream project's expectations while focusing this repository on Vime-specific maintenance.

## Collaboration Scope

slime is the RL training infrastructure behind [GLM-4.5 through GLM-5.1](https://z.ai) and a large number of internal experiments at Z.ai. Vime builds on that training stack and adapts the rollout side around vLLM.

Our goal for open-source collaboration is focused on **bug fixes** and **general-purpose large-scale RL optimizations**. We have had several successful collaborations with the community in this area, including:

- Speculative decoding in RL
- Low-precision training: fp8 rollout + bf16/fp8 training, int4 rollout + int4 QAT training
- Deterministic training

### What We Welcome

| Category | Examples |
|----------|----------|
| **Bug reports** | Crashes, incorrect results, documentation errors |
| **Bug fixes** | PRs that fix existing issues with tests or clear reproduction |
| **General RL optimizations** | Performance improvements with clear benchmarks that can be verified through CI or standard training runs |

### What's Currently Outside Our Scope

| Category | Reason |
|----------|--------|
| **Large-scale code refactoring** | This would add considerable overhead to syncing between the internal and open-source versions, particularly in coordinating with internal algorithm teams. |
| **Design / abstraction proposals** | e.g., universal data standards, eval standards, tool base classes. Standard-setting involves non-technical factors; slime intentionally avoids such content to keep things flexible for both the community and internal teams, and Vime follows that scope. |
| **Features that cannot be clearly verified** | Correctness is critically important for a training framework. If a feature cannot be verified through CI or routine internal training, it becomes difficult for us to ensure timely fixes, which could affect the project's long-term reliability. |
| **Features independent of the RL framework** | e.g., full algorithm reproduction pipelines. While these lower the barrier to entry, they are difficult to include in routine verification. vime aims to be lightweight — more like Flask than Django. We recommend building such pipelines in separate repositories; we are happy to reference them in the README. |
| **Major modifications to Megatron** | We do not plan to maintain a Megatron fork through vime. The goal is to switch Megatron versions relatively painlessly; Megatron performance optimization and feature completion are not primary objectives. |

### Why This Policy?

The upstream slime design is tightly coupled with large-scale post-training requirements, and Vime keeps that focused scope so the fork can remain maintainable while adapting the rollout backend.

Thank you for your understanding and patience. We truly appreciate the effort community contributors put in, and we're sorry if this policy causes any inconvenience.

---

## 开源协作范围说明

感谢你对 vime 的关注和支持！社区的每一份贡献我们都非常珍视。Vime 由 [slime](https://github.com/THUDM/slime) 衍生而来，因此下面的协作范围保留上游项目的基本预期，同时聚焦本仓库中的 Vime 维护工作。

### 背景

slime 承担了智谱内部的大量实验，包括 GLM 4.5 至 5 的全部 RL 流程，以及大量的日常实验。Vime 基于这套训练栈，并将 rollout 侧适配到 vLLM。

在我们的已知范围内，目前只有极少的前沿大模型团队愿意公开如此核心且完整的 Infra 组件。Vime 在继承 slime 训练栈的同时，保持聚焦的协作范围，以便更稳定地维护 vLLM 相关适配。

### 协作范围

我们将开源协作的范围限制在 **bug fix** 和一些**通用的大规模 RL 优化**上。在这方面我们也和社区达成了多次成功的合作，例如：

- RL 中的投机采样
- 低精度训练：fp8 rollout + bf16/fp8 training，int4 rollout + int4 QAT training
- 确定性训练

### 我们欢迎的

| 类别 | 说明 |
|------|------|
| **Bug 报告** | 崩溃、结果错误、文档错误等 |
| **Bug 修复** | 带有测试或清晰复现步骤的修复 PR |
| **通用 RL 优化** | 有明确 benchmark 且可通过 CI 或常规训练验证的性能优化 |

### 暂时不在协作范围内的

| 类别 | 原因 |
|------|------|
| **较大范围的代码重构** | 会给内外部版本同步带来较多额外工作，尤其是在与内部算法团队的沟通协调上。 |
| **带有项目规划建议的标准或抽象** | 例如引入某种通用数据标准、eval 标准、工具构建基类等。标准的设立在大多数团队中会涉及到非技术因素；slime 的设计中故意避开了类似的内容，Vime 也沿用这一范围。 |
| **无法进行明确验证的功能** | 训练框架的正确性至关重要。如果一个功能不能通过 CI 或智谱内部常规训练进行验证，我们就难以及时发现和修复问题，这对项目的长期可靠性会带来不小的风险。 |
| **与 RL 框架较为独立的功能** | 例如整套算法复现流程。这类内容较难纳入日常验证流程，不太容易持续保证正确性。vime 是一个相对轻量的框架，更像是 Flask 而非 Django。建议在独立的 repo 中搭建，我们也非常愿意在 README 中引用所有使用了 vime 的项目链接。 |
| **对 Megatron 的大幅度改动** | 目前我们没有计划通过 vime 维护一套 Megatron fork。vime 的目标是能够相对无痛地切换 Megatron 版本，Megatron 的性能优化和功能补全不在主要目标中。 |

### 为什么需要这样的策略？

上游 slime 的设计与大规模后训练需求有很强的绑定关系。Vime 保留这一聚焦策略，是为了在适配 vLLM rollout 后端的同时保持项目可维护。

感谢大家的理解与支持，如果这一策略给您带来了不便，我们深表歉意。
