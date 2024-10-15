# Auto Risk Rule Search

## 项目介绍

Auto Risk Rule Search 是一个定制化的风控策略工具，旨在帮助信贷风控策略的开发与优化。策略组合的设计和变量搜索往往会随着变量增多变得复杂、耗时，尤其在搜索过程中涉及到多维度变量组合及优化策略效果（如 lift 提升等）。为了简化这些工作，我开发了这个工具，通过自动化搜索每层决策树中的最优变量，减少了手动调整的麻烦。

## 功能特点

- **灵活的树构建控制**：支持设置最大深度、优先特征、固定阈值、动态步长等多种配置。
- **自动变量选择**：在构建过程中，自动选择每层最优变量进行分裂，或者根据预设顺序搜索最优阈值。
- **丰富的参数设置**：通过参数控制分裂策略，满足不同业务场景的需求。
- **策略效果优化**：旨在提高逾期率预测的准确性，并优化整体风控策略。

## 示例

详细案例展示在 `example/example.md` 中，包括如何使用该工具进行策略构建和优化。

## 安装方式

你可以通过以下命令直接从 GitHub 安装该包：

```bash
pip install git+https://github.com/feifeigeaha/auto_risk_rule_search.git
