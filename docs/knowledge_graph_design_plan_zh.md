# Knowledge Graph 设计计划

## 目标

中期目标是构建一个面向选定数学领域的、带有难度信息的 theorem/proof/strategy knowledge graph。这个图会把数学命题、证明、引理、证明策略、数学对象和概念标注组织到统一表示中，用于支持相似定理检索、证明策略推荐、难度估计，以及后续的 prover training。

这个 knowledge graph 是 RO1 中结构化数据集和 RO2 中 prover training pipeline 之间的桥梁。它的作用是把数学推理模式显式暴露给学习系统，而不是只依赖原始 theorem/proof 文本。

## 范围

初始版本应聚焦在一个可控的数学领域。建议优先选择 number theory，重点包括：

- Number fields
- Monogenicity
- Primality certification
- Modular square root
- 相关 LMFDB 对象和 metadata

这个领域足够聚焦，便于做高质量标注；同时又能自然连接 RO3 中 Self-Proving Model 的实验任务。

## 图结构 Schema

Knowledge graph 应建模为 heterogeneous knowledge graph，即包含多种节点类型和边类型。

### 节点类型

- `Theorem`：数学命题、定理、lemma 或 challenge problem。
- `Proof`：完整证明、证明草稿、certificate 或 verification transcript。
- `Lemma`：证明中使用的可复用中间结论。
- `ProofStrategy`：证明方法或推理模式。
- `MathObject`：数学对象，例如 number field、polynomial、ideal、prime 或 elliptic curve。
- `Concept`：更高层的数学概念，例如 monogenicity、local obstruction、quadratic residue 或 discriminant。

### 边类型

- `theorem_has_proof`：连接 theorem 与对应 proof。
- `proof_uses_strategy`：连接 proof 与其中使用的 proof strategy。
- `proof_invokes_lemma`：连接 proof 与依赖的 lemma 或 intermediate claim。
- `lemma_supports_theorem`：连接 lemma 与其支持证明的 theorem。
- `theorem_about_object`：连接 theorem 与涉及的数学对象。
- `object_has_property`：连接数学对象与其已知属性。
- `theorem_related_to_concept`：连接 theorem 与相关数学概念。
- `theorem_similar_to_theorem`：连接数学上相似的 theorem。
- `strategy_transfers_to`：表示某种 proof strategy 可以迁移到另一类 theorem 或 domain。

## 节点属性

每类节点都应使用稳定 schema，以便后续查询、训练和更新。

### Theorem 属性

```text
id
statement_text
formal_statement_optional
domain
subdomain
source
status
difficulty_vector
strategy_labels
embedding
```

### Proof 属性

```text
id
theorem_id
proof_text
formal_proof_optional
proof_length
proof_dag_features
strategy_labels
difficulty_vector
verification_status
embedding
```

### Strategy 属性

```text
id
strategy_name
description
applicable_domains
example_theorems
embedding
```

### Math Object 属性

```text
id
object_type
canonical_representation
LMFDB_id_optional
known_properties
certificate_status
embedding
```

## 难度表示

Graph 中应为 theorem 和 proof 节点加入 difficulty vector。初始版本可以使用可解释的 proxy features：

```text
proof_length_score
dependency_depth_score
number_of_invoked_lemmas
number_of_distinct_strategies
strategy_rarity_score
formal_verification_cost_proxy
search_complexity_proxy
```

Difficulty vector 也可以进一步离散化为 `easy`、`medium`、`hard` 等 bucket，用于 curriculum learning 和评估。

## 相似度表示

Graph 应同时通过显式边和 learned embeddings 表示 theorem/proof similarity。

Similarity 应综合以下因素：

- Theorem statement 和 proof text 的文本相似度
- 共享的数学对象
- 共享的数学概念
- 共享的 proof strategies
- 相似的 difficulty profile
- 重叠的 dependency structure
- Graph-neighborhood similarity

初始版本中的 `theorem_similar_to_theorem` 边可以先用规则生成。后续版本可以根据 learned graph embeddings 更新这些边。

## Representation Learning

Representation learning pipeline 建议分两阶段进行。

### 阶段 1：Feature-Based Baseline

第一版先用 hybrid embeddings，包含：

- Theorem/proof 文本 embedding
- Strategy labels 的 one-hot 或 multi-hot 表示
- Difficulty vectors
- Object 和 concept features
- 局部 graph-neighborhood statistics

这个 baseline 可以快速验证 graph schema 和 annotations 是否支持有效检索。

### 阶段 2：Graph Neural Representation Learning

第二版在 heterogeneous graph 上训练 graph representation model。候选模型包括：

- R-GCN
- GraphSAGE
- Heterogeneous Graph Transformer
- 其他 heterogeneous GNN variants

可选训练目标包括：

- Link prediction
- Proof strategy prediction
- Similar theorem retrieval
- Difficulty prediction
- Lemma recommendation

中期 demo 应优先展示：

- 给定一个 theorem，检索 top-k similar theorems。
- 给定一个 theorem，推荐可能的 proof strategies。
- 给定一个 theorem，估计 difficulty profile。

## 查询接口

Knowledge graph 应暴露一个最小查询接口，供后续 prover training 调用。

必要查询包括：

```text
Input: theorem statement
Output: top-k similar theorems

Input: theorem statement
Output: recommended proof strategies

Input: theorem or math object
Output: relevant lemmas and concepts

Input: theorem statement
Output: estimated difficulty vector
```

初始版本可以实现为 notebook、command-line script 或 lightweight Python API。

## 持续更新

当 prover 产生新的 proof attempt、lemma、certificate 或 verifier transcript 时，graph 应支持更新。

中期版本可以先支持 batch update：

- 添加新的 theorem、proof、lemma 和 object 节点。
- 添加新的 proof-strategy 和 proof-dependency 边。
- 重新计算 similarity edges。
- 重新计算或 fine-tune node embeddings。
- 更新 difficulty estimates。

后续版本可以进一步使用 continual learning 方法处理动态图 embedding。

## 中期交付物

到中期节点，项目应交付：

- `KG schema v1`
- 一个选定数学领域中的 curated seed dataset
- 一个构建完成的 heterogeneous theorem/proof/strategy knowledge graph
- 初始 theorem、proof、strategy、object 和 concept embeddings
- Similar theorem retrieval demo
- Strategy recommendation demo
- Difficulty estimation demo
- 包含自动指标和专家验证的 evaluation report

## 成功标准

到中期节点，我们将交付一个面向选定数学领域的 difficulty-aware theorem/proof/strategy knowledge graph，以及训练好的 graph embeddings 和 retrieval functions。系统应能把一个新 theorem 映射到相似定理、可能证明策略、相关 lemmas 和 estimated difficulty profile。
