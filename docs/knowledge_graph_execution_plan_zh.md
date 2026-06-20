# Knowledge Graph 实施计划

## 目标

构建一个可执行原型：difficulty-aware theorem/proof/strategy knowledge graph。该原型应能够读取 curated seed dataset，将其规范化为 typed graph schema，生成 graph edges 和 features，训练 baseline representations，并提供相似定理检索、证明策略推荐和难度估计功能。

本文档应作为工程落地计划使用。

## 预期目录结构

创建如下项目结构：

```text
kg/
  data/
    raw/
    processed/
    seed/
  schemas/
    theorem.schema.json
    proof.schema.json
    lemma.schema.json
    strategy.schema.json
    math_object.schema.json
    concept.schema.json
  src/
    ingest.py
    normalize.py
    extract_objects.py
    annotate_strategies.py
    build_graph.py
    compute_difficulty.py
    embed.py
    train_baseline.py
    retrieve.py
    evaluate.py
  notebooks/
    kg_demo.ipynb
  outputs/
    graph/
    embeddings/
    reports/
  README.md
```

如果已有 repo 使用其他结构，可以调整路径，但应保留相同的逻辑模块。

## 数据输入

原型至少应支持三类输入。

### Theorem Records

使用 JSONL 作为主要交换格式：

```json
{
  "id": "thm_0001",
  "statement_text": "Let K be a number field defined by ...",
  "domain": "number_theory",
  "subdomain": "monogenicity",
  "source": "curated",
  "status": "proved",
  "objects": ["obj_0001"],
  "concepts": ["monogenicity", "number_field"]
}
```

### Proof Records

```json
{
  "id": "proof_0001",
  "theorem_id": "thm_0001",
  "proof_text": "The proof proceeds by checking local obstructions ...",
  "lemmas": ["lem_0001", "lem_0002"],
  "strategy_labels": ["local_to_global", "case_analysis"],
  "verification_status": "informal_checked",
  "source": "curated"
}
```

### Math Object Records

```json
{
  "id": "obj_0001",
  "object_type": "number_field",
  "canonical_representation": "x^3 - 3x + 1",
  "lmfdb_id": null,
  "known_properties": {
    "degree": 3,
    "monogenicity_status": "unknown"
  }
}
```

## Step 1: 定义 JSON Schemas

为以下节点类型实现 JSON schemas：

- `Theorem`
- `Proof`
- `Lemma`
- `ProofStrategy`
- `MathObject`
- `Concept`

每个 schema 应校验 required fields、field types，以及必要的 enum values。

验收标准：

- 非法 records 会校验失败，并给出有用错误信息。
- 合法 seed records 能通过校验。
- 每种节点类型至少有一个 valid 和一个 invalid schema test。

## Step 2: 构建 Seed Dataset

在 `kg/data/seed/` 下创建一个小而连贯的 seed dataset。

最低目标规模：

- 20 条 theorem records
- 20 条 proof records
- 20 条 lemma records
- 10 条 proof strategy records
- 20 条 math object records
- 10 条 concept records

Seed dataset 应聚焦 number theory，并尽量包含 monogenicity、primality certification 和 modular square root 相关例子。

验收标准：

- 每个 proof 都引用一个存在的 theorem。
- 每个被引用的 lemma 都存在。
- 每个 theorem 至少引用一个 concept 或 math object。
- 每个 strategy label 都能映射到一个已知 `ProofStrategy` 节点。

## Step 3: 规范化输入数据

实现 `normalize.py`。

职责：

- 加载 raw JSONL files。
- 根据 schemas 校验 records。
- 规范化 IDs 和 labels。
- 规范化 strategy names、domains 和 subdomains。
- 移除重复 records。
- 将 normalized records 写入 `kg/data/processed/`。

验收标准：

- Normalization 输出是 deterministic 的。
- 能检测 duplicate IDs。
- 能报告 missing foreign keys。
- 脚本可重复运行，且输出顺序稳定。

## Step 4: 抽取 Objects 和 Concepts

实现 `extract_objects.py`。

初始实现可以是 rule-based，用于从 theorem 和 proof 文本中检测并规范化常见 number-theoretic objects 和 concepts。

示例：

- Polynomial expressions，例如 `x^3 - 3x + 1`
- Number field mentions
- Prime mentions
- Discriminant mentions
- Monogenicity mentions
- Quadratic residue mentions
- Modular square root mentions

验收标准：

- Extractor 能产生候选 `MathObject` 和 `Concept` references。
- 抽取结果会附加到 theorem 和 proof records。
- Manual annotations 优先级高于 extracted annotations。

## Step 5: 标注 Proof Strategies

实现 `annotate_strategies.py`。

初始实现支持两种模式：

- 读取人工提供的 `strategy_labels`。
- 使用 keyword 或 pattern rules 添加 weak labels。

示例 weak-label rules：

```text
assume for contradiction -> contradiction
induct on -> induction
case -> case_analysis
local obstruction -> local_to_global
certificate -> certificate_verification
random sample / probability -> probabilistic_checking
interactive -> interactive_proof
```

验收标准：

- 每个 proof 可以有零个或多个来自 strategy vocabulary 的 strategy labels。
- Weak labels 需要标明 provenance，例如 `manual` 或 `rule_based`。
- Unknown strategy labels 在校验阶段应被拒绝。

## Step 6: 计算 Difficulty Features

实现 `compute_difficulty.py`。

为每个 theorem 和 proof 计算可解释的 difficulty vector：

```text
proof_length_score
dependency_depth_score
number_of_invoked_lemmas
number_of_distinct_strategies
strategy_rarity_score
formal_verification_cost_proxy
search_complexity_proxy
```

初始 proxy 定义：

- `proof_length_score`：proof text 的 normalized token count。
- `dependency_depth_score`：lemma dependency 的 longest path；如果不可用则设为 1。
- `number_of_invoked_lemmas`：引用 lemma 的数量。
- `number_of_distinct_strategies`：strategy labels 数量。
- `strategy_rarity_score`：proof strategies 在 seed dataset 中的 inverse frequency。
- `formal_verification_cost_proxy`：基于 proof length、dependency 数量和 verification status 的 heuristic。
- `search_complexity_proxy`：基于 strategy rarity 和 theorem similarity sparsity 的 heuristic。

验收标准：

- 每个 theorem 和 proof 都有 numeric difficulty vector。
- Difficulty values 归一化到稳定范围，优先使用 `[0, 1]`。
- 每条 record 都有 difficulty bucket：`easy`、`medium` 或 `hard`。

## Step 7: 构建 Heterogeneous Graph

实现 `build_graph.py`。

Graph 应包含以下节点类型：

- `Theorem`
- `Proof`
- `Lemma`
- `ProofStrategy`
- `MathObject`
- `Concept`

Graph 应包含以下边类型：

- `theorem_has_proof`
- `proof_uses_strategy`
- `proof_invokes_lemma`
- `lemma_supports_theorem`
- `theorem_about_object`
- `object_has_property`
- `theorem_related_to_concept`
- `theorem_similar_to_theorem`
- `strategy_transfers_to`

原型建议存储格式：

- Nodes：CSV 或 Parquet table，包含 `node_id`、`node_type` 和 attributes。
- Edges：CSV 或 Parquet table，包含 `src`、`dst`、`edge_type` 和 attributes。
- Optional graph object：如后续训练需要，可保存 NetworkX pickle 或 PyTorch Geometric object。

验收标准：

- Graph construction 是 deterministic 的。
- 所有 edge endpoints 都存在。
- 输出 node 和 edge counts。
- Graph artifacts 写入 `kg/outputs/graph/`。

## Step 8: 创建 Similarity Edges

实现初始 rule-based theorem similarity。

Similarity score 应综合：

- Shared domain 或 subdomain
- Shared concepts
- Shared math objects
- Shared proof strategies
- Similar difficulty vector
- Text similarity，如果 embeddings 可用

初始公式：

```text
similarity = w1 * domain_overlap
           + w2 * concept_jaccard
           + w3 * object_jaccard
           + w4 * strategy_jaccard
           + w5 * difficulty_similarity
           + w6 * text_similarity
```

验收标准：

- 每个 theorem 最多有 `k` 条 similarity edges 指向其他 theorem。
- 排除 self-edges。
- Similarity scores 作为 edge attributes 保存。
- 公式权重可配置。

## Step 9: 生成 Baseline Embeddings

实现 `embed.py`。

使用以下信息生成 hybrid node embeddings：

- Theorem/proof text features
- Multi-hot strategy features
- Multi-hot concept features
- Difficulty vectors
- Local graph degree features

先使用简单可复现的 baseline：

- TF-IDF，或可用时使用 local sentence embedding
- Concatenated structured features
- 如有必要，使用 PCA 或 truncated SVD 降维

验收标准：

- 每个 theorem、proof、strategy、object 和 concept 都有 embedding vector。
- Embedding dimension 可配置。
- Embeddings 保存到 `kg/outputs/embeddings/`。
- Metadata file 能将 node IDs 映射到 embedding rows。

## Step 10: 训练 Baseline Predictive Tasks

实现 `train_baseline.py`。

至少训练两个 baseline models：

- Theorem-to-strategy prediction
- Difficulty bucket prediction

可选附加任务：

- Theorem-proof 或 theorem-strategy link prediction

中期原型可使用简单模型：

- Logistic regression
- Random forest
- Shallow MLP
- Cosine nearest-neighbor retrieval

验收标准：

- 脚本报告 train/dev metrics。
- 结果保存到 `kg/outputs/reports/`。
- Model artifacts 可保存并重新加载。

## Step 11: 实现 Retrieval Functions

实现 `retrieve.py`。

必须包含以下函数：

```python
retrieve_similar_theorems(theorem_id: str, k: int = 5) -> list[dict]
recommend_strategies(theorem_id: str, k: int = 5) -> list[dict]
recommend_lemmas(theorem_id: str, k: int = 5) -> list[dict]
estimate_difficulty(theorem_id: str) -> dict
```

每个输出应包含：

- 返回实体 ID
- Score
- Human-readable label 或 text snippet
- 可用时提供 explanation features

验收标准：

- Functions 能从已保存的 graph 和 embedding artifacts 中运行。
- 固定 seed 时 retrieval results deterministic。
- Demo 可在 seed dataset 上运行，无需外部服务。

## Step 12: 评估 Graph Quality

实现 `evaluate.py`。

自动指标：

- Strategy recommendation 的 top-k precision
- Lemma recommendation 的 top-k recall，如果有 labels
- Difficulty bucket prediction 的 accuracy 或 macro-F1
- Link prediction 的 AUC，如果已实现
- 基于 shared concepts 和 strategies 的 retrieval sanity checks

专家验证输出：

- 导出 CSV 或 Markdown 格式的 sample retrievals。
- 包含 theorem、retrieved theorem、score、shared concepts、shared strategies 和 difficulty comparison。

验收标准：

- Evaluation 生成 machine-readable metrics file。
- Evaluation 生成 human-readable report。
- 至少包含 10 个 example retrieval cases。

## Step 13: 构建 Demo

创建 `kg/notebooks/kg_demo.ipynb` 或等价 CLI demo。

Demo 应展示：

- 加载 graph
- 显示 graph statistics
- 查询一个 theorem
- 返回 similar theorems
- 返回 recommended proof strategies
- 返回 relevant lemmas
- 显示 estimated difficulty vector

验收标准：

- Demo 能从 processed data 端到端运行。
- Demo 不需要手动修改。
- Demo 输出足够清楚，可用于中期展示。

## Step 14: 增加 Batch Update 支持

实现简单的新数据更新路径。

必要行为：

- 接收新的 theorem/proof/lemma/object records。
- 校验 records。
- 添加到 processed data。
- 重建 graph edges。
- 重新计算 similarity edges。
- 重新生成 embeddings。

验收标准：

- 一个小的新 batch 可以加入，且不破坏已有 IDs。
- 更新后的 graph statistics 能反映新增 records。
- 更新后 retrieval 仍可运行。

## Step 15: 文档

编写 `kg/README.md`。

README 应包含：

- 项目目的
- 数据格式
- Graph schema
- Setup instructions
- 每个 pipeline step 的运行命令
- Example retrieval query
- Evaluation command
- Known limitations

验收标准：

- 新开发者只阅读 README 就能运行 pipeline。
- 中期 demo 所需的每条命令都有文档说明。

## 推荐 Pipeline Commands

最终原型应支持类似流程：

```bash
python kg/src/normalize.py --input kg/data/seed --output kg/data/processed
python kg/src/extract_objects.py --input kg/data/processed --output kg/data/processed
python kg/src/annotate_strategies.py --input kg/data/processed --output kg/data/processed
python kg/src/compute_difficulty.py --input kg/data/processed --output kg/data/processed
python kg/src/build_graph.py --input kg/data/processed --output kg/outputs/graph
python kg/src/embed.py --graph kg/outputs/graph --output kg/outputs/embeddings
python kg/src/train_baseline.py --graph kg/outputs/graph --embeddings kg/outputs/embeddings --output kg/outputs/reports
python kg/src/evaluate.py --graph kg/outputs/graph --embeddings kg/outputs/embeddings --output kg/outputs/reports
```

## 中期 Definition of Done

当以下条件满足时，中期实现可视为完成：

- Seed dataset 存在并通过 schema validation。
- Heterogeneous graph 可由 seed data 构建。
- Theorem 和 proof 节点拥有 difficulty vectors。
- Theorem 之间生成 similarity edges。
- Graph nodes 拥有 baseline embeddings。
- 至少两个 downstream tasks 已完成评估。
- Retrieval functions 能返回 similar theorems、strategies、lemmas 和 difficulty estimates。
- Demo 和 README 可用。

## 后续扩展

中期之后可以继续扩展：

- Heterogeneous GNN training
- Graph updates 的 continual learning
- LMFDB-scale data integration
- Lean 或 Isabelle proof dependency graphs integration
- Self-Proving Model verifier transcripts integration
- Expert annotation 的 active learning
