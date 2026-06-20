# LeanRank-Based Knowledge Graph 设计计划

## 1. 目标

本计划从 [`erbacher/LeanRank-data`](https://huggingface.co/datasets/erbacher/LeanRank-data) 出发，构建一个面向 formal theorem proving 的 theorem-proofstate-premise knowledge graph。

该 knowledge graph 的核心目标是：把 Lean/mathlib 中的 theorem、proof state、positive premise、negative premise、file/module、tactic step 和 weak proof-technique labels 组织成可训练、可检索、可解释的图结构，用于支持 lemma retrieval、premise ranking、proof dependency analysis、proof-technique abstraction 和 theorem similarity learning。

## 2. 关于 `erbacher/LeanRank-data`

`erbacher/LeanRank-data` 是 Hugging Face 上的一个 Lean/mathlib proof-search 数据集，采用 parquet 格式发布，规模约为 2.09M rows，包含较大的 train split 和较小的 validation split。

该数据集围绕 Lean proof state 的 premise selection 任务组织。每条记录包含当前 theorem 和 proof location 的 metadata、formal proof context、一个或多个 useful premises，以及一组 negative candidate premises。关键字段包括：

```text
file_path
full_name
start
tactic_idx
context
all_pos_premises
pos_premise
neg_premises
```

这些字段非常适合构建 knowledge graph。`full_name` 和 `file_path` 可以识别当前 theorem 和 module；`context` 表示当前 proof state；`pos_premise` 和 `all_pos_premises` 表示有用的 lemmas 或 declarations；`neg_premises` 则提供 supervised ranking 所需的对比候选。

对本项目而言，`LeanRank-data` 提供三类关键监督信号：

- Proof dependency supervision：哪些 premises 对某个 proof state 有用。
- Retrieval supervision：哪些 premises 应该排在 negative candidates 前面。
- Structural supervision：theorem、file、proof state 和 premise 在 mathlib 中如何连接。

### 2.1 数学领域覆盖

因为 `LeanRank-data` 来源于 Lean/mathlib，它覆盖的 theorem 范围跟随 mathlib 的目录组织，而不是某一个单独数学专题。数据集中可能包含来自以下 formalized mathematics 领域的 theorem、lemma、definition 和 declaration：

```text
Algebra
LinearAlgebra
RingTheory / FieldTheory
GroupTheory
ModuleTheory
SetTheory
Logic / Foundations
OrderTheory
Topology
Analysis
MeasureTheory
Probability
NumberTheory
Combinatorics
CategoryTheory
Data / Basic Structures
```

该数据集本身不提供干净的显式 `domain` 标签。因此，domain 信息应从 `file_path` 中解析得到。

示例：

```text
Mathlib/LinearAlgebra/Finsupp/LinearCombination.lean
-> domain_tag = LinearAlgebra
-> subdomain_tag = Finsupp

Mathlib/SetTheory/ZFC/Basic.lean
-> domain_tag = SetTheory
-> subdomain_tag = ZFC

Mathlib/Algebra/Module/Injective.lean
-> domain_tag = Algebra
-> subdomain_tag = Module

Mathlib/Data/Set/Image.lean
-> domain_tag = Data
-> subdomain_tag = Set
```

实现时应基于 sample 或 full data 生成 domain coverage report：

```text
top-level domain counts
second-level subdomain counts
example theorem names per domain
example file paths per domain
premise usage frequency by domain
```

这个报告可以帮助我们说明 KG 的数学覆盖范围，并指导后续 sampling、evaluation 和展示选择。

## 3. 数据定位

`LeanRank-data` 不是自然语言 theorem-proof 数据集，而是一个 formal premise-ranking 数据集。每条记录大致表示：

```text
当前 theorem / proof state
+ 一个 positive premise
+ 若干 negative premises
+ 文件路径、定理名、proof step index 等 metadata
```

因此，本项目的 KG 以 formal proof dependency 为核心组织原则，重点建模 theorem、proof state、premise 及其检索关系。

## 4. 核心节点类型

### 4.1 Theorem

表示 Lean/mathlib 中的 theorem、lemma、definition 或 declaration。

主要字段：

```text
full_name
file_path
start
namespace
statement_or_context
domain_tag
embedding
```

### 4.2 ProofState

表示某个 theorem 在某个 tactic step 下的 proof context / goal。

主要字段：

```text
proof_state_id
theorem_full_name
tactic_idx
context
goal_text
local_hypotheses
embedding
```

### 4.3 Premise

表示可用于证明当前 proof state 的 premise。它可以是 theorem、lemma、definition 或 simp rule。

主要字段：

```text
premise_id
full_name
code
path
pid
premise_type
embedding
```

### 4.4 FileModule

表示 Lean 文件或 mathlib module。

主要字段：

```text
file_path
module_name
top_level_area
namespace
```

### 4.5 TacticStep

表示 proof 中的某一步 tactic index。LeanRank-data 不直接提供 tactic 文本，但 `tactic_idx` 可以作为 proof progression 的位置标记。

主要字段：

```text
tactic_step_id
theorem_full_name
tactic_idx
context_before
positive_premises
negative_premises
```

### 4.6 ProofTechnique

表示从 premise name、theorem name、premise code 或 context pattern 中弱标注出来的证明技法。

本文档中，`Strategy` 明确指 `ProofTechnique`。`linear_algebra_reasoning`、`set_reasoning`、`topology_reasoning` 这类 domain-oriented labels 不应作为 strategy labels，而应作为从 `file_path` 和 namespace 中解析出来的 `domain_tag`、`subdomain_tag` 或 reasoning-domain metadata。

初始 weak proof-technique labels 包括：

```text
simplification
rewriting_or_coercion
typeclass_resolution
definition_unfolding
theorem_application
case_or_constructor_reasoning
extensionality
induction
contradiction
computation
automation
```

## 5. 核心边类型

```text
Theorem --has_proof_state--> ProofState
Theorem --appears_in_file--> FileModule
ProofState --positive_uses--> Premise
ProofState --negative_candidate--> Premise
Theorem --invokes_premise--> Premise
Premise --defined_in_file--> FileModule
ProofState --at_tactic_step--> TacticStep
ProofState --uses_proof_technique--> ProofTechnique
Theorem --similar_to_theorem--> Theorem
Premise --co_occurs_with--> Premise
```

## 6. 边的来源与标注需求

上述边可以分为三类：从 `LeanRank-data` 直接观测得到的边、通过程序聚合自动生成的边，以及需要项目自己定义 weak labels 或规则的边。

### 6.1 直接观测得到的边

这些边可以直接由 LeanRank 字段构建，不需要人工标注：

```text
Theorem --has_proof_state--> ProofState
```

来源：

```text
full_name + tactic_idx + context
```

```text
Theorem --appears_in_file--> FileModule
```

来源：

```text
file_path
```

```text
ProofState --positive_uses--> Premise
```

来源：

```text
pos_premise
all_pos_premises
```

```text
ProofState --negative_candidate--> Premise
```

来源：

```text
neg_premises
```

```text
Premise --defined_in_file--> FileModule
```

来源：

```text
pos_premise.path
neg_premise.path
```

### 6.2 程序自动派生的边

这些边不需要人工标注，但实现时需要定义 deterministic aggregation rules。

```text
Theorem --invokes_premise--> Premise
```

由以下关系聚合得到：

```text
Theorem --has_proof_state--> ProofState
ProofState --positive_uses--> Premise
```

如果某个 theorem 的任意 proof state 使用某 premise 作为 positive premise，则添加该 theorem 到 premise 的 invokes 边。

```text
ProofState --at_tactic_step--> TacticStep
```

由以下字段得到：

```text
tactic_idx
```

其中 `TacticStep` 节点是 KG pipeline 引入的建模抽象。

```text
Premise --co_occurs_with--> Premise
```

由共现规则生成，例如：

```text
两个 premises 出现在同一个 all_pos_premises list 中
两个 premises 被同一个 theorem 的 proof states 使用
```

### 6.3 需要 weak label 或规则定义的边

这些边需要项目自己设计 labeling rules 或 similarity definitions。

```text
ProofState --uses_proof_technique--> ProofTechnique
```

`LeanRank-data` 没有显式 proof-technique labels。因此这些边需要根据 premise name、premise code、theorem name 和 context patterns 通过 weak-labeling rules 生成。

示例：

```text
premise name contains simp / @[simp] -> simplification
Eq, congr, coe, cast -> rewriting_or_coercion
ext, extensionality -> extensionality
induction, rec_on -> induction
by_contra, not_not, contradiction -> contradiction
norm_num, decide, omega, ring, linarith -> computation_or_automation
```

```text
Theorem --similar_to_theorem--> Theorem
```

`LeanRank-data` 没有显式 theorem similarity labels。因此 similarity edges 需要通过定义 similarity function 生成，可以使用以下信号：

```text
shared positive premises
same file_path or namespace
proof state context embedding similarity
shared weak proof-technique labels
similar difficulty vectors
```

因此，本项目最核心的自定义标注/规则部分是：

```text
proof-technique weak labels
theorem similarity function
```

## 7. 从 LeanRank 字段到 KG 的映射

| LeanRank 字段 | KG 用途 |
|---|---|
| `file_path` | `FileModule` 节点；`Theorem appears_in_file` 边 |
| `full_name` | 当前 `Theorem` 节点 ID |
| `start` | theorem 在源文件中的位置 metadata |
| `tactic_idx` | `TacticStep` 节点或 proof progression metadata |
| `context` | `ProofState` 文本和 embedding 输入 |
| `all_pos_premises` | `ProofState positive_uses Premise` 边 |
| `pos_premise` | premise-ranking 正例 |
| `neg_premises` | premise-ranking 负例 |
| `pos_premise.code` | `Premise` 节点文本和 embedding 输入 |
| `pos_premise.full_name` | `Premise` 节点 ID |
| `pos_premise.path` | `Premise defined_in_file FileModule` 边 |

## 8. 图的主要学习任务

### 8.1 任务 1：Proof-Technique Abstraction

从 formal premise usage 中抽象出 proof-technique labels。这个任务负责定义 proof-technique vocabulary，并为 proof states 分配可解释标签。

例如：

```text
[simp] theorem / simp namespace -> simplification
Nat / Int arithmetic lemmas, norm_num, ring -> computation_or_automation
coe / subtype / ext lemmas -> coercion_or_extensionality
induction / rec_on lemmas -> induction
by_contra / contradiction lemmas -> contradiction
```

这些 proof-technique labels 不是 gold labels，而是 weak labels，用于中期可解释展示和后续 retrieval。

### 8.2 任务 2：Proof-Technique Retrieval

给定一个 proof state，从冻结后的 proof-technique candidate pool 中检索或排序最可能的 proof techniques。

输入：

```text
ProofState context
goal_text
positive premise names and code, for training only
candidate proof-technique pool
```

输出：

```text
top-k proof techniques
confidence scores
rule or model explanation
```

这个任务可以通过 rule-based weak labeling、LLM-assisted labeling，或基于 weak labels 训练的 supervised multi-label classifier 实现。Label space 必须保持 closed：系统只能从冻结后的 `ProofTechnique` pool 中选择。

### 8.3 任务 3：Premise Ranking

给定一个 proof state，从候选 premises 中排序出最相关的 premise。

输入：

```text
ProofState context
Candidate Premise code/name/path
Graph neighborhood features
```

输出：

```text
score(ProofState, Premise)
```

这是 LeanRank-data 最自然的 supervised task。

### 8.4 任务 4：Lemma Retrieval

给定 theorem 或 proof state，检索 top-k 可用 lemmas/premises。

输出：

```text
top-k positive premises
relevance scores
explanation features
```

### 8.5 任务 5：Theorem Similarity

两个 theorem 如果使用相似 premises、处在相似 file/module、拥有相似 proof states，则认为相似。

相似度来源：

```text
shared positive premises
shared premise namespaces
context embedding similarity
file/module proximity
proof-technique label overlap
```

## 9. Difficulty 表示

在 LeanRank setting 中，difficulty 主要刻画 proof search complexity，即一个 proof state 对 premise retrieval 和 proof continuation 的难度。

建议 difficulty vector：

```text
context_length_score
num_local_hypotheses
num_positive_premises
avg_positive_premise_length
premise_namespace_rarity
tactic_step_index_score
negative_candidate_hardness
retrieval_entropy
```

含义：

- `context_length_score`：proof state 越长，通常越复杂。
- `num_local_hypotheses`：local context 越大，检索和证明越难。
- `num_positive_premises`：需要多个 premise 的 state 可能更复杂。
- `premise_namespace_rarity`：少见 namespace 的 premise 可能更难。
- `negative_candidate_hardness`：如果负例和正例很相似，ranking 更难。
- `retrieval_entropy`：模型对候选 premise 不确定时，proof state 更难。

## 10. Representation Learning

### 10.1 Baseline 表示

先构造 hybrid embedding：

```text
ProofState embedding = text_embedding(context) + metadata features
Premise embedding = text_embedding(code + full_name + path)
Theorem embedding = aggregate(proof state embeddings + positive premise embeddings)
```

### 10.2 Graph 表示

后续训练 graph model：

```text
R-GCN
GraphSAGE
HGT
bi-encoder + graph reranker
```

中期建议先实现：

```text
TF-IDF / sentence embedding baseline
+ graph features
+ nearest-neighbor retrieval
+ supervised premise ranking classifier
```

## 11. 中期交付物

中期交付应包括：

- LeanRank subset loader
- Normalized theorem/proofstate/premise schema
- Heterogeneous KG construction pipeline
- Positive/negative premise edges
- Baseline embeddings
- Premise ranking baseline
- Proof-technique retrieval demo
- Lemma retrieval demo
- Theorem similarity demo
- Proof-technique weak-labeling demo
- Difficulty feature report

## 12. 成功标准

到中期节点，系统应能完成：

```text
Input: a Lean proof state context
Output: top-k relevant premises

Input: a Lean proof state context
Output: top-k proof techniques from the frozen ProofTechnique pool

Input: a theorem full_name
Output: similar theorems based on shared premises/proof states

Input: a proof state
Output: difficulty vector and proof-search complexity bucket
```

一句话总结：该 KG 围绕 Lean formal proof search 构建。它的核心是利用 LeanRank-data 中的 proof state、positive premise 和 negative premise 构建 theorem-premise dependency graph，并训练可用于 premise ranking 和 lemma retrieval 的 representation model。
