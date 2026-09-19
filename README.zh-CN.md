# OmegaPRM 数据生成 baseline（v2，独立实现）

[English](README.md) | 简体中文

本项目独立实现论文 **[Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/abs/2406.06592)**（Luo et al., 2024）提出的 OmegaPRM 数据生成部分，属于非官方实现。

这版用于**按论文算法生成小规模过程监督数据**。默认采用独立的32次筛题、
8次MC估计、100次搜索；不再使用上一版的按行/词切分和“任何UNKNOWN就丢弃整个节点”规则。

这是论文数据生成部分的独立实现，**不是作者官方代码，也不保证复现论文的相同数据或性能**。
已明确的算法细节尽量对齐；论文未固定的操作细节以及替换生成模型等差异列在
[ALIGNMENT.zh-CN.md](ALIGNMENT.zh-CN.md)。不包含PRM训练或测试集评估。

## 1. 小规模真实模型baseline

需要Python 3.10+。在支持vLLM的GPU环境安装模型依赖：

```bash
python -m pip install -r requirements.txt
```

准备训练题目的JSON/JSONL文件后，先用少量候选题运行：

```bash
python generate_prm_data.py \
  --problem data/train_questions.jsonl \
  --output runs/omega_small \
  --backend vllm \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --limit 20 \
  --seed 1234
```

默认已配置：筛题32条/题；MC采样8条/前缀；每题最多100次搜索；长度目标16份；
Math-Verify验证器；temperature=0.8；top_p=0.95。
**前四项是对齐目标，生成模型与采样温度等不是论文原实验的等同设置。**

`--limit 20`表示从输入中用固定seed随机选20道**候选题**，然后筛题，不保证保留20道。
默认limit为100；`--limit 0`处理全部输入。不会自动下载或混用MATH测试题。
比对其他方法时，用manifest中的`selected_ids`和同一份输入固定候选题，再使用一致的筛选/校准。

Qwen2.5-Math-7B-Instruct配置的上下文长度是4096，本版默认`--max-model-len 4096`，
每次续写上限`--max-tokens 2048`。如果实际题目/续写超长，程序会明确报错；
应重新选择适合的模型、上下文和续写上限，新建run，而不是把截断当成数学推理错误。
不能仅提高`max-model-len`而假定模型支持更长上下文。

8条续写是一个MC调用，不是总共只生成8条。100次搜索可能涉及数百次MC调用；
先检查少量题目的耗时与`summary.json`中的实际调用/输出token数。
若只是调试，可设置`--search 5 --limit 5`，但报告实验时应注明偏离默认预算。
`--max-calls`是可选的搜索调用上限，默认不启用；触发时状态为`call_limit`。

## 2. 输入

JSON数组或扩展名为`.jsonl`的逐行JSON：

```json
{"id":"train-001","problem":"What is 2 + 3?","final_answer":"5"}
```

必需字段：`problem`、`final_answer`；`id`可选，但建议提供稳定且唯一的题目ID。
答案应为最终值/LaTeX表达式，不是完整解答。缺字段、空答案、重复ID会报错。
本代码不负责重建论文12K训练题与500题测试划分；需要使用者提供明确来源的训练子集。

## 3. 生成流程

1. **筛题**：独立采样32条完整解答，只有同时出现正确和错误答案的题目被保留。
2. **长度校准**：默认用保留题目的完整筛题解答计算平均token长度，除以16，
   得到整个run固定的二分停止阈值。也可用`--mean-solution-tokens`传入外部校准平均值。
3. **建树/搜索**：重新采样8条根续写；从混合MC节点的未访问错误rollout中按Q+U选择；
   沿token中点二分，MC>0向右（包括MC=1），MC=0向左。原文空格、换行和Unicode均保留。
4. **建边与导出**：保留正向切分点到首个定位零边界的显式边；长边仍保存供审计，
   只导出符合单步长度约定的边，使用目标前缀的MC作为soft label。

32次筛题和8次根采样使用独立seed，分别记录成本。通过筛题的题目仍可能在新的8次采样中
全对/全错，状态为`root_no_candidates`；不会静默追加采样来改变这个结果。

但在后续二分过程中，8条全对（MC=1）或全错（MC=0）只决定向右或向左继续定位，
不会因此丢弃整道题。这类节点不进入下一轮候选池，因为候选池要求`0<MC<1`。
通过筛题的题目总生成量为`32 + 8 + 8m`，其中m是额外评估的新前缀数；缓存前缀无需重新生成。

## 4. 验证器及失败政策

`--verifier auto`：使用Math-Verify。
也可以明确指定`math-verify`，或用`conservative`选择无依赖数值验证器。后者不适合覆盖完整MATH的符号答案。
验证器不会把标准答案是否为输出子串当作正确性判断。

MC分母始终为配置的k，不会只保留容易解析的输出后缩小分母：

| 情况 | 默认处理 |
| --- | --- |
| 完整输出且最终答案正确 | 1 |
| 完整输出且答案错误 | 0 |
| 完整输出但未提取/未解析出答案 | 0，并记录原因 |
| 生成到长度上限、输出未结束 | 报错，不默默加入训练集 |
| prompt超出上下文预算 | 报错 |
| 验证器执行异常/超时 | 报错 |

`--unknown-policy error`可把完整但无法验证的答案也改为报错。
`--incomplete-policy incorrect`可明确把被截断的输出算0；这是可选实验设置，不能不加说明使用。
这些政策属于论文没有完全公开的操作细节；不要把它们写成论文的规定。
Math-Verify需要支持`raise_on_error`的近期版本；不满足时会提示升级。

## 5. 输出

| 路径 | 内容 |
| --- | --- |
| `manifest.json` | 源码/输入指纹、配置、版本、所选题目ID |
| `screening/*.json` | 每题32条筛选输出与判定，包含被过滤的题目 |
| `calibration.json` | 平均长度、阈值、计算来源和样本数 |
| `problems/*.json` | 保留题目的节点、边、所有rollout、搜索事件和标注 |
| `samples.jsonl` | 去重后的单步soft-label训练候选 |
| `summary.json` | 筛题结果、样本正负数、调用/续写/token计数、异常映射次数 |

单步样本含有：

```text
question, parent_prefix, step, prefix
parent_id, child_id, step_tokens
label, hard_label, mc_source, n_correct, n_rollouts
problem_id, sample_id, sampling_seed, gold_answer
```

`parent_prefix + step == prefix`严格成立；`label`为child节点的MC，`hard_label=int(label>0)`。
`gold_answer`供审计，不能作为PRM输入。训练输入是题目、父前缀和当前步骤。
多步边保存在题目文件中，不会伪装成一个单步样本。

若边终点是已经验证答错的完整终止解答，MC按吸收终止状态解释为0，
`mc_source=terminal_answer`、`n_rollouts=0`；不会假装为它又生成了8条续写。
其余估计为`mc_source=monte_carlo`，保留真实分子、分母。

正负比例由搜索结果产生，本版没有额外进行1:1重采样。
数据太少时也不会重复复制样本。默认导出全部小规模样本，不做论文大规模实验的下采样。
**训练/验证划分按题目分组**，不能把同一题的不同前缀随机拆到不同集合。

## 6. 续跑、模型与提示词

原命令添加`--resume`：

```bash
python generate_prm_data.py --problem data/train_questions.jsonl \
  --output runs/omega_small --backend vllm --limit 20 --seed 1234 --resume
```

筛题和搜索各有独立的逐题原子检查点；最多重做当前未完成题目。
汇总可从检查点重建。输入、配置、源代码或真实模型运行依赖版本变化时拒绝续跑。
v1输出不兼容v2，请新建run目录。无需GPU也能运行标准库测试。

一个run只允许一个写入进程。强制终止可能遗留`.run.lock`；确认旧进程停止后手动删除再续跑。

默认使用chat template，并原样追加解答前缀。base模型可用`--prompt-mode plain`。
自定义few-shot格式可用`--prompt-template prompts/custom.txt`；文件必须各包含一次
`{{problem}}`和`{{prefix}}`，并以`{{prefix}}`结尾（其后不加换行）。原始LaTeX花括号无需转义。
模板内容计入run指纹。代码未捏造或内置论文的Gemma2四样本提示。
使用`--revision <commit>`固定模型和tokenizer版本；默认模型名对应的远程版本仍可能变化。

## 7. 验证范围与复现实验措辞

已验证：标准库回归测试、所有受控错误边界、公式值、筛题/校准/建边流程、断点续跑。
未验证：真实GPU/vLLM推理，真实Math-Verify安装集成；本环境的安装源未提供该依赖。
`VALIDATION.txt`记录实际执行结果。没有运行论文原模型，也没有声称模型准确率提升。

建议在实验中称为：**OmegaPRM data-generation baseline (our implementation)**。
同时报告生成模型、数据子集、32/8/100设置、长度校准、答案验证器和失败政策。
详细的已对齐项及仍需报告的差异见[ALIGNMENT.zh-CN.md](ALIGNMENT.zh-CN.md)。

## 可视化保存的搜索结构

不需要安装额外依赖：

```bash
python tools/visualize_problem.py "runs/omega_small/problems/<hash>.json" -o tree.html
```

用浏览器打开 `tree.html`，可离线使用。支持拖动、缩放、按 ID 查找节点；点击节点查看完整推理前缀和采样续写，点击边查看新增文本。节点颜色表示 MC，实线表示已导出训练样本的边。勾选 **Show disconnected probes** 可显示没有保存连接的探测节点。只绘制实际保存的边，不把续写展开成虚构分支；共享状态使结构可能是 DAG，而非严格的树。

点击 **Export SVG** 导出静态图；覆盖已有 HTML 时加 `--force`。项目包含可直接打开的 dummy 后端示例 [examples/search_graph.html](examples/search_graph.html)。HTML 内嵌完整题目检查点内容，分享时也会分享其中的数据。
