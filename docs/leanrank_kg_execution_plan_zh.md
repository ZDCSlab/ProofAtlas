# LeanRank-Based Knowledge Graph 执行计划

## 1. 目标

基于 [`erbacher/LeanRank-data`](https://huggingface.co/datasets/erbacher/LeanRank-data) 构建一个可运行的 formal proof knowledge graph prototype。该 prototype 应支持从 Hugging Face parquet 数据中抽样、规范化记录、构建 theorem-proofstate-premise graph、生成 difficulty features、训练 premise-ranking baseline，并提供 lemma retrieval 和 theorem similarity 查询。

## 2. 推荐目录结构

```text
leanrank_kg/
  data/
    raw/
    sample/
    processed/
  schemas/
    theorem.schema.json
    proof_state.schema.json
    premise.schema.json
    file_module.schema.json
    proof_technique.schema.json
  src/
    download_or_sample.py
    normalize.py
    parse_context.py
    build_graph.py
    compute_difficulty.py
    weak_label_proof_technique.py
    embed.py
    train_ranker.py
    retrieve.py
    evaluate.py
  outputs/
    graph/
    embeddings/
    models/
    reports/
  notebooks/
    leanrank_kg_demo.ipynb
  README.md
```

## 3. Implementation Phases

实现应分为两个明确阶段。

### 3.1 Phase 1: KG + Embedding Index + Ranker

Phase 1 是中期 baseline，不需要 graph neural model。KG 在这一阶段作为结构化数据层，用于抽取 nodes、edges、labels、features 和 retrieval supervision。

Phase 1 组件：

```text
theorem-proofstate-premise KG
proof-technique weak labeler
difficulty feature computation
proof state and premise embeddings
premise embedding index
cosine retrieval baseline
supervised premise ranker
retrieval and evaluation API
```

应用到 test data 的流程：

```text
1. 只用 train split 构建 training KG。
2. 在 train data 上 fit vectorizers、normalizers、weak labelers 和 rankers。
3. 冻结 proof-technique candidate pool、labeling rules、LLM prompt、thresholds 和 ranker parameters。
4. 将每个 test proof state 作为 query node。
5. 使用 train-fitted encoders 编码 test proof state。
6. 从选定 premise index 中 retrieve/rank candidate premises。
7. Test positive premises 只在 evaluation 时揭晓。
```

评估只使用 closed-index setting：

```text
Closed-index setting: candidate premises = train premises only
```

如果某个 test gold premise 不在 train premise index 中，则该样本标记为 `out_of_index_gold`，并从需要可检索性的 Recall@k/MRR 分子分母计算中排除。报告中必须包含 gold premise coverage：

```text
gold_premise_coverage = test examples whose gold premise appears in train index / all test examples
```

Phase 1 成功标准：

```text
在 test proof states 上计算 Recall@k 和 MRR
使用 frozen rules 或 frozen hybrid rules 生成 proof-technique labels
retrieval examples 包含 scores 和 explanations
closed-index setting 报告 gold premise coverage
```

### 3.2 Phase 2: Graph Model

Phase 2 在 heterogeneous KG 上增加 graph representation model。这是 post-baseline enhancement，而不是把 KG 应用到 test data 的前置条件。

候选模型：

```text
GraphSAGE
R-GCN
Heterogeneous Graph Transformer
bi-encoder with graph-based reranker
```

训练目标：

```text
ProofState-Premise link prediction
premise ranking with graph-enhanced embeddings
Theorem-Theorem similarity prediction
ProofState-ProofTechnique prediction
```

Phase 2 应复用 Phase 1 的 theorem-level train/validation/test split。Test proof states 仍作为 query nodes，其 gold `positive_uses` edges 只能用于最终评估。

Phase 2 成功标准：

```text
graph model 相比 Phase 1 baseline 提升 Recall@k 或 MRR
ablation 比较 text-only、feature-only、KG baseline 和 graph model
模型能处理 unseen test theorem nodes，且训练时不使用 test positive edges
```

## 4. 加载 LeanRank-data

使用 Hugging Face datasets 或直接读取 parquet。

建议先不要加载全部 2M+ rows，先抽样：

```text
train sample: 50k rows
val sample: 5k rows
```

抽样策略：

```text
按 file_path 分层抽样
保留 theorem full_name 的多条 tactic_idx 记录
优先保留 all_pos_premises 非空的记录
```

### 4.1 Train/Validation/Test 划分

使用 theorem-level split，而不是 row-level split。同一个 theorem 在 `LeanRank-data` 中可能对应多条 records，因为同一个 `full_name` 可能包含多个 proof states，也可能在同一个 proof state 下有多个 positive premises。如果按 row 随机划分，会导致同一个 theorem 泄漏到 train 和 evaluation 中。

默认划分：

```text
train: 80% theorem full_name groups
validation: 10% theorem full_name groups
test: 10% theorem full_name groups
```

分组 key：

```text
theorem_group_key = full_name
```

推荐划分流程：

```text
1. 按 full_name 将 sampled rows 分组。
2. 将每个 theorem group 分配到 train、validation 或 test。
3. 同一个 theorem 的所有 proof states、positive premises 和 negative premises 必须留在同一个 split。
4. 按从 file_path 解析出的 top-level domain_tag 做分层抽样。
5. 使用固定 random seed，并写出 split assignment files。
```

输出文件：

```text
data/sample/train_rows.parquet
data/sample/val_rows.parquet
data/sample/test_rows.parquet
data/sample/split_assignments.json
```

Leakage checks：

```text
no full_name appears in more than one split
no proof_state_id appears in more than one split
train/val/test domain distributions are reported
```

如果 sample 很小，使用：

```text
train: 70%
validation: 15%
test: 15%
```

Validation split 用于 hyperparameter choices、weak-label rule tuning 和 threshold selection。Test split 只用于最终报告。

验收标准：

- 能读取 train 和 val split。
- 能保存 deterministic sample。
- 输出字段包括 `file_path`、`full_name`、`start`、`tactic_idx`、`context`、`all_pos_premises`、`neg_premises`、`pos_premise`。
- Split assignment 是 theorem-level 且 deterministic。
- 没有 theorem `full_name` 同时出现在多个 split 中。
- 输出 train、validation 和 test 的 domain distribution statistics。

## 5. 定义规范化 Schema

定义以下 JSON schemas。

### 5.1 Theorem

```json
{
  "id": "thm:Submodule.mem_span_set",
  "full_name": "Submodule.mem_span_set",
  "file_path": "Mathlib/LinearAlgebra/Finsupp/LinearCombination.lean",
  "start": [450, 1],
  "namespace": "Submodule",
  "domain_tag": "LinearAlgebra"
}
```

### 5.2 ProofState

```json
{
  "id": "ps:Submodule.mem_span_set:0",
  "theorem_id": "thm:Submodule.mem_span_set",
  "tactic_idx": 0,
  "context": "R : Type ... ⊢ m ∈ span R s ↔ ...",
  "goal_text": "m ∈ span R s ↔ ...",
  "local_hypotheses": ["R : Type", "M : Type"]
}
```

### 5.3 Premise

```json
{
  "id": "premise:Set.image_id",
  "full_name": "Set.image_id",
  "code": "theorem image_id (s : Set α) : id '' s = s",
  "path": "Mathlib/Data/Set/Image.lean",
  "pid": 52652
}
```

### 5.4 ProofTechnique

```json
{
  "id": "proof_technique:simplification",
  "name": "simplification",
  "source": "weak_rule",
  "description": "Uses simplification lemmas or simp-tagged declarations.",
  "candidate_patterns": ["@[simp]", ".simp", "_simp"],
  "priority": 10
}
```

验收标准：

- 所有 normalized records 都能通过 schema validation。
- ID construction deterministic。
- Missing premise fields 能被记录或跳过，不导致 pipeline 崩溃。

## 6. 规范化 LeanRank Records

实现 `normalize.py`。

输入：sampled LeanRank rows。
输出：

```text
theorems.jsonl
proof_states.jsonl
premises.jsonl
positive_edges.jsonl
negative_edges.jsonl
file_modules.jsonl
```

处理逻辑：

- `full_name` 生成 theorem node。
- `(full_name, tactic_idx, context hash)` 生成 proof state node。
- `pos_premise` 和 `all_pos_premises` 生成 premise nodes。
- `neg_premises` 生成 candidate premise nodes。
- `file_path` 和 premise `path` 生成 file module nodes。

验收标准：

- 同一个 theorem 只生成一个 theorem node。
- 同一个 premise full_name 只生成一个 premise node。
- 每个 proof state 至少连接当前 theorem。
- Positive 和 negative edges 都保留 label。

## 7. 解析 Context

实现 `parse_context.py`。

从 Lean context 中提取：

```text
local hypotheses
goal text
symbols
namespace hints
typeclass hints
```

简单规则：

- `⊢` 之后作为 `goal_text`。
- `⊢` 之前按行或变量声明切分为 local hypotheses。
- 从 theorem full_name 和 file_path 提取 namespace/domain。

验收标准：

- 至少 95% sampled rows 能解析出 goal_text。
- 解析失败的 rows 有 fallback，不中断 pipeline。

## 8. 构建 Graph

实现 `build_graph.py`。

节点类型：

```text
Theorem
ProofState
Premise
FileModule
TacticStep
ProofTechnique
```

边类型：

```text
Theorem --has_proof_state--> ProofState
Theorem --appears_in_file--> FileModule
ProofState --positive_uses--> Premise
ProofState --negative_candidate--> Premise
Theorem --invokes_premise--> Premise
Premise --defined_in_file--> FileModule
ProofState --at_tactic_step--> TacticStep
ProofState --uses_proof_technique--> ProofTechnique
Premise --co_occurs_with--> Premise
Theorem --similar_to_theorem--> Theorem
```

输出：

```text
nodes.parquet
edges.parquet
graph_stats.json
```

验收标准：

- 所有 edge endpoints 存在。
- Graph stats 包括每种 node/edge type 的数量。
- Graph construction deterministic。

## 9. 构造 Theorem Similarity Edges

相似度公式：

```text
similarity(T1, T2) =
  w1 * shared_premise_jaccard
+ w2 * file_or_namespace_overlap
+ w3 * proof_state_text_similarity
+ w4 * proof_technique_overlap
+ w5 * tactic_index_profile_similarity
```

中期先为每个 theorem 保留 top-k similar theorem edges。

验收标准：

- 无 self edge。
- 每个 theorem 最多 k 条 similarity edges。
- Similarity score 写入 edge attributes。

## 10. 弱标注 ProofTechnique

实现 `weak_label_proof_technique.py`。

基于 premise name、premise code、theorem name 和 context 生成 weak proof-technique labels。

### 10.1 Proof-Technique Candidate Pool

Proof-technique candidate pool 应在正式打标签之前生成。它应该是一个受控 vocabulary，而不是每个 proof state 自由生成的标签集合。

初始候选池：

```text
simplification
rewriting_or_coercion
typeclass_resolution
definition_unfolding
theorem_application
extensionality
case_or_constructor_reasoning
logical_reasoning
induction
contradiction
computation
automation
```

候选池由三类来源生成：

```text
manual_seed_proof_techniques
rule_mined_proof_techniques_from_names
frequency_filtered_proof_techniques
```

`manual_seed_proof_techniques` 是上面的固定 proof-technique labels。`rule_mined_proof_techniques_from_names` 通过 Lean declaration names、annotations 和 code patterns 映射到 proof-technique family。`frequency_filtered_proof_techniques` 是可选的 technique labels，只有在 sampled data 中出现频率足够高时才保留。

Name/code-to-technique 示例：

```text
@[simp], .simp, _simp -> simplification
Eq, congr, coe, cast, Subtype.coe -> rewriting_or_coercion
ext, extensionality -> extensionality
cases, constructor, rec -> case_or_constructor_reasoning
induction, rec_on, casesOn -> induction
by_contra, contradiction, not_not -> contradiction
norm_num, decide, omega, ring, linarith, nlinarith -> computation or automation
inferInstance, inst, typeclass -> typeclass_resolution
```

Frequency filtering：

```text
minimum_support = 50 proof states in train
maximum_proof_technique_count = 20 labels for the midterm prototype
```

如果某个 candidate label 低于 `minimum_support`，则映射到更宽的 parent label，例如 `theorem_application`、`automation` 或 `logical_reasoning`。

Candidate pool 保存为：

```text
outputs/reports/proof_technique_candidate_pool.json
```

每个 proof-technique entry 应包含：

```json
{
  "id": "proof_technique:rewriting_or_coercion",
  "name": "rewriting_or_coercion",
  "source": "manual_seed+name_rule",
  "patterns": ["Eq", "congr", "coe", "cast", "Subtype.coe"],
  "minimum_support": 50,
  "parent": "theorem_application"
}
```

### 10.2 Weak Label Assignment

使用 deterministic rules 分配 proof-technique labels。一个 proof state 可以获得多个 proof-technique labels。

初始 assignment rules：

```text
[simp] or .simp or theorem tagged simp -> simplification
rw, Eq, congr, coe -> rewriting_or_coercion
ext, extensionality -> extensionality
cases, rec, constructor -> case_or_constructor_reasoning
induction, rec_on, casesOn -> induction
by_contra, contradiction, not_not -> contradiction
norm_num, decide, omega, ring, linarith, nlinarith -> computation_or_automation
```

Rule inputs：

```text
positive premise full_name
positive premise code
negative premise full_name and code, used only for hardness features
proof state context
theorem full_name
file_path
```

Conflict handling：

```text
allow multi-label assignment
store rule provenance for every assigned label
if more than 5 labels fire, keep the top 5 by rule priority and support
if no rule fires, assign no label rather than guessing
```

Rule priority：

```text
explicit Lean annotation or tactic-like name > declaration name pattern > premise code pattern > context keyword
```

不要在 test split 上调 weak-label rules。Rule thresholds 和 priority choices 只能使用 train 和 validation data finalized。

`LinearAlgebra`、`SetTheory`、`Topology`、`CategoryTheory` 这类 domain-oriented signals 应保存为 `domain_tag` 或 `subdomain_tag`，不能作为 proof-technique labels。

### 10.3 LLM-Assisted Labeling

LLM 可以作为可选的第二阶段标注器，用于处理 deterministic rules 无法高置信标注的 proof states。LLM 不能发明新标签，只能从第 10.1 节冻结后的 proof-technique candidate pool 中选择。

适合使用 LLM-assisted labeling 的情况：

```text
没有 rule-based label 的 proof states
存在多个低优先级冲突标签的 proof states
用于 human-readable demo 的 proof states
用于改进 rule coverage 的小规模 validation samples
```

不要在看到 test performance 后再用 LLM 为 test split 创建新标签。如果需要在 test examples 上使用 LLM labeling，则 prompt、candidate pool 和 decision rules 必须提前冻结。

LLM 输入应包含：

```text
proof_state_id
theorem full_name
proof state context
goal_text
positive premise full_name
positive premise code
candidate proof-technique pool
rule-based labels, if any
```

LLM 输出必须是 strict JSON：

```json
{
  "proof_state_id": "ps:Submodule.mem_span_set:0",
  "labels": [
    {
      "name": "rewriting_or_coercion",
      "confidence": 0.78,
      "rationale": "The positive premise involves coercion/coercion-normalization patterns."
    }
  ],
  "abstain": false
}
```

Decision policy：

```text
accept LLM label only if confidence >= 0.70
accept at most 3 labels per proof state
reject labels not in frozen candidate pool
prefer deterministic rule labels over LLM labels when confidence is high
store provenance as llm_assisted
```

推荐 hybrid pipeline：

```text
1. 对所有 proof states 先应用 deterministic rules。
2. 只把 unlabeled 或 ambiguous train/validation examples 发送给 LLM。
3. 可选：用一小批 human-reviewed sample 校准 LLM prompt。
4. 冻结 prompt、candidate pool、thresholds 和 conflict policy。
5. 用 frozen hybrid labeler 标注 train、validation 和 test。
```

输出：

```text
outputs/reports/proof_technique_llm_labels.jsonl
outputs/reports/proof_technique_label_provenance.csv
outputs/reports/llm_label_abstention_rate.json
```

验收标准：

- 每个 proof state 可以有零个或多个 proof-technique labels。
- 每个 label 附带 rule provenance。
- Proof-technique distribution 写入报告。
- Proof-technique candidate pool 保存 patterns、source、support 和 parent labels。
- 分别报告 train、validation 和 test 上的 weak-label coverage。
- Test labels 由 frozen rules 生成，不允许 test-set tuning。
- 如果启用 LLM-assisted labeling，prompt、candidate pool、thresholds 和 conflict policy 必须在 test labeling 前冻结。
- LLM labels 包含 confidence、rationale 和 provenance。
- 自动拒绝 frozen proof-technique pool 之外的标签。

## 11. 计算 Difficulty Features

实现 `compute_difficulty.py`。

为每个 proof state 计算：

```text
context_length_score
num_local_hypotheses
num_positive_premises
avg_positive_premise_length
premise_namespace_rarity
tactic_step_index_score
negative_candidate_hardness
```

为每个 theorem 聚合：

```text
mean_proof_state_difficulty
max_proof_state_difficulty
num_proof_states
num_unique_positive_premises
```

验收标准：

- ProofState 和 Theorem 都有 difficulty vector。
- 数值归一化到 `[0, 1]`。
- 生成 `easy`、`medium`、`hard` bucket。

## 12. 生成 Embeddings

实现 `embed.py`。

Baseline embedding：

```text
ProofState embedding = TF-IDF(context + goal_text)
Premise embedding = TF-IDF(full_name + code + path)
Theorem embedding = average(proof_state embeddings + positive premise embeddings)
```

可选增强：

```text
structured features
namespace one-hot
proof-technique multi-hot
difficulty vector
degree features
```

验收标准：

- ProofState、Premise、Theorem 都有 embedding。
- Embedding rows 和 node IDs 有 metadata mapping。
- 输出保存到 `outputs/embeddings/`。

## 13. 训练 Premise Ranker

实现 `train_ranker.py`。

训练样本：

```text
(proof_state, pos_premise, label=1)
(proof_state, neg_premise, label=0)
```

Baseline model：

```text
cosine similarity baseline
logistic regression on pair features
small MLP reranker
```

Pair features：

```text
cosine(context_embedding, premise_embedding)
same_namespace
same_file_area
premise_frequency
proof_technique_overlap
difficulty_features
```

验收标准：

- 报告 Recall@k、MRR、AUC 或 accuracy。
- 模型可保存和加载。
- Val split 上能跑完整评估。

## 14. 实现 Retrieval API

实现 `retrieve.py`。

必要函数：

```python
retrieve_premises(proof_state_id: str, k: int = 10) -> list[dict]
retrieve_similar_theorems(theorem_id: str, k: int = 10) -> list[dict]
explain_premise_match(proof_state_id: str, premise_id: str) -> dict
get_proof_technique_labels(proof_state_id: str) -> list[dict]
get_difficulty_profile(entity_id: str) -> dict
```

验收标准：

- 输入 proof state，返回 top-k premise。
- 输入 theorem，返回 similar theorem。
- 输出包含 score 和简短 explanation。

## 15. 评估

实现 `evaluate.py`。

指标：

```text
Recall@1, Recall@5, Recall@10
MRR
AUC for positive/negative premise ranking
proof-technique label coverage
difficulty distribution
theorem similarity sanity checks
```

报告：

```text
metrics.json
retrieval_examples.md
proof_technique_distribution.csv
difficulty_distribution.csv
graph_stats.json
```

验收标准：

- 至少 20 个 retrieval examples。
- 每个 example 包括 proof state、gold positive premise、top retrieved premises、score。
- 报告能直接用于中期展示。

## 16. Demo

创建 notebook 或 CLI demo。

Demo 流程：

```text
Load sampled LeanRank data
Build KG
Show graph stats
Select one theorem
Show its proof states
Retrieve top-k premises for one proof state
Show positive premise rank
Show similar theorems
Show weak proof-technique labels
Show difficulty vector
```

验收标准：

- Demo 可端到端运行。
- 不依赖外部服务。
- 可在小样本上 5 分钟内完成。

## 17. Midterm Definition of Done

中期完成标准：

- 已从 LeanRank-data 抽样并生成 processed dataset。
- 已构建 theorem-proofstate-premise heterogeneous KG。
- 已生成 positive/negative premise edges。
- 已完成 weak proof-technique labeling。
- 已计算 proof state 和 theorem difficulty vectors。
- 已生成 baseline embeddings。
- 已训练 premise ranking baseline。
- 已实现 premise retrieval 和 theorem similarity 查询。
- 已生成 evaluation report 和 demo。

## 18. 后续扩展

中期后可扩展：

- 使用全量 LeanRank-data。
- 训练 heterogeneous GNN 或 graph transformer。
- 加入真实 tactic text，如果可从 LeanDojo 或 mathlib 源码补充。
- 与 ProofNet 合并，加入自然语言 theorem/proof。
- 将 weak proof-technique labels 升级为人工审核 labels。
- 用 graph retrieval 作为 Lean prover 的 premise selection module。
