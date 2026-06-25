# ProofAtlas Homepage Design

Source report: `outputs/reports/ProofAtlas_Project_Report.md`

## Goal

The homepage should present ProofAtlas as a research system for Lean premise retrieval, not as a marketing landing page or interactive dashboard.

The main story is:

> ProofAtlas improves Lean proof-state to premise retrieval by fusing proof-state evidence with theorem-neighborhood evidence.

The page should be static. It should use graph visualization and concise result charts to make the system structure, knowledge graph, and held-out test results understandable at a glance.

## Evidence to Highlight

Use the held-out test split results from the report.

Primary method:

```text
weighted_rrf_llm_theorem_tuned
```

Main comparison:

| Metric | Dense baseline | Primary method | Absolute gain | Relative gain |
| --- | ---: | ---: | ---: | ---: |
| Covered Recall@100 | 0.2362 | 0.4746 | +0.2384 | +101.0% |
| All-positive Recall@100 | 0.1851 | 0.4074 | +0.2223 | +120.1% |
| MAP | 0.0494 | 0.0981 | +0.0487 | +98.6% |
| nDCG@10 | 0.0697 | 0.1236 | +0.0539 | +77.4% |

Important nuance:

- `weighted_rrf_llm_theorem_tuned` is the primary non-BGE method and has the strongest MAP and nDCG@10 among the main reported variants.
- `weighted_rrf_llm_pretrained_tuned` is the auxiliary BGE fusion and has the strongest candidate-pool recall on the test split: Covered Recall@100 `0.4813`, All-positive Recall@100 `0.4124`.
- The homepage should show BGE as an auxiliary recall-maximizing variant, not as the primary claim.

## Page Structure

### 1. Header

The first viewport should contain:

- Project name: `ProofAtlas`
- One-line claim: `Theorem-neighborhood retrieval for Lean premise selection`
- Short supporting sentence: `A static view of how dense, lexical, proof-state expansion, and LLM-enriched theorem-neighborhood evidence are fused for premise retrieval.`
- Resource links:
  - Repository: `https://github.com/ZDCSlab/ProofAtlas`
  - Dataset: `https://huggingface.co/datasets/ZDCSlab/proofatlas-enriched`
  - Report: `outputs/reports/ProofAtlas_Project_Report.md`
- Four compact metric cards:
  - Covered Recall@100: `0.2362 -> 0.4746`
  - All-positive Recall@100: `0.1851 -> 0.4074`
  - MAP: `0.0494 -> 0.0981`
  - nDCG@10: `0.0697 -> 0.1236`

Do not use a generic product hero. The system graph should be the visual focus of the first page.

### 1.1 Resource Links

The homepage should include explicit links to the repo, dataset, and report near the header and again in a compact footer.

| Resource | Link target | Purpose |
| --- | --- | --- |
| GitHub repo | `https://github.com/ZDCSlab/ProofAtlas` | Source code, pipeline commands, and implementation details |
| HuggingFace dataset | `https://huggingface.co/datasets/ZDCSlab/proofatlas-enriched` | Exported enriched dataset artifact |
| Project report | `outputs/reports/ProofAtlas_Project_Report.md` | Full held-out test results and ablation details |

If the homepage is deployed outside the repository, the report link should point to the GitHub-rendered markdown file instead of a local relative path.

### 1.2 Layout Control

The layout should be carefully controlled so the page feels balanced, readable, and intentional on both desktop and mobile.

Desktop layout rules:

- Use a centered page container with a maximum content width around `1180px` to `1280px`.
- Keep the header concise: project title, one-sentence claim, resource links, and metric cards should fit before the main graph without crowding it.
- The system pipeline graph should be the dominant first visual element and should span the main content width.
- Place metric cards in a single row on desktop, with equal widths and consistent numeric alignment.
- Use two-column layouts only when the content naturally pairs:
  - dataset summary next to domain distribution
  - T1 main comparison next to T1 ablation progression, if chart labels remain legible
  - the two demo case graphs side by side
- Do not place large graph visualizations inside decorative cards. Use section bands or unframed constrained content instead.
- Keep chart heights stable, approximately `260px` to `360px`, so the page rhythm stays predictable.
- Keep the knowledge graph visually separate from the system pipeline graph; it should feel like a second figure, not an extension of the first.

Mobile layout rules:

- Stack all sections vertically.
- Metric cards should become a two-column grid, then one column on narrow screens.
- The system graph should either scale down with readable labels or switch to a simplified vertical graph.
- Long theorem names in demo cases must wrap cleanly without overflowing their node boxes.
- Charts should keep axis labels readable; if necessary, use horizontal bars instead of dense grouped bars.

Spacing rules:

- Use consistent vertical section spacing.
- Avoid dense text blocks immediately above or below complex graphs.
- Every graph should have a short title and one concise caption explaining what to look for.
- No label, chart legend, node, or edge should overlap another element.

### 2. System Pipeline Graph

Use a static graph visualization rather than an interactive flowchart.

Recommended graph layout:

```text
LeanRank processed data
        |
        v
theorem-disjoint train / test split
        |
        +------------------+--------------------+----------------------+
        |                  |                    |                      |
        v                  v                    v                      v
proof states        theorem profiles       premises          positive edges
        |                  |                    |
        v                  v                    |
dense retrieval      LLM enrichment             |
        |                  |                    |
        v                  v                    |
proof-state       theorem-neighborhood          |
evidence          evidence                      |
        |                  |                    |
        +----------+-------+--------------------+
                   |
                   v
          weighted RRF fusion
                   |
                   v
        ranked premise candidates
                   |
                   v
     Recall@100 / MAP / nDCG@10
```

The visual grammar should distinguish:

- Data nodes: proof states, premises, theorem profiles, positive edges
- Retrieval source nodes: dense retrieval, lexical TF-IDF, proof-state expansion, theorem-neighborhood premise source
- Fusion node: weighted RRF
- Output nodes: ranked premise candidates, theorem neighbors, guidance bundles
- Evaluation nodes: Recall@100, MAP, nDCG@10

The center of the graph should be `weighted RRF fusion`, with the primary method label `weighted_rrf_llm_theorem_tuned` placed near it.

### 3. Report Result Charts

The homepage should include charts derived from the report, not only text tables.

#### Chart A: Dense vs Primary Method

Best chart type: grouped bar chart.

Metrics:

- Covered Recall@100
- All-positive Recall@100
- MAP
- nDCG@10

Series:

- Dense baseline
- Primary method

This chart makes the main claim immediately visible.

#### Chart B: T1 Ablation Progression

Best chart type: line chart or connected dot plot.

X axis methods:

1. `dense`
2. `lexical`
3. `dense_lexical_rrf`
4. `full_union_rrf`
5. `weighted_rrf_tuned_recall`
6. `weighted_rrf_theorem_tuned`
7. `weighted_rrf_llm_theorem_tuned`
8. `weighted_rrf_llm_pretrained_tuned`

Y axis:

- Covered Recall@100 as the primary line
- MAP as a secondary smaller line or adjacent mini chart

Values:

| Method | Covered Recall@100 | MAP |
| --- | ---: | ---: |
| dense | 0.2362 | 0.0494 |
| lexical | 0.2713 | 0.0523 |
| dense_lexical_rrf | 0.2915 | 0.0581 |
| full_union_rrf | 0.3888 | 0.0705 |
| weighted_rrf_tuned_recall | 0.4408 | 0.0861 |
| weighted_rrf_theorem_tuned | 0.4618 | 0.0929 |
| weighted_rrf_llm_theorem_tuned | 0.4746 | 0.0981 |
| weighted_rrf_llm_pretrained_tuned | 0.4813 | 0.0964 |

This chart should show the progression from simple retrieval to theorem-neighborhood fusion.

#### Chart C: T2 Theorem Retrieval Comparison

Best chart type: grouped bar chart.

Metrics:

- Neighbor premise Recall@10
- Neighbor premise Recall@50
- Neighbor premise Recall@100
- MAP
- nDCG@10

Series:

- Baseline theorem profile
- LLM-enriched TF-IDF profile
- LLM-enriched BGE profile

Values:

| Method | Recall@10 | Recall@50 | Recall@100 | MAP | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline theorem profile | 0.1728 | 0.2841 | 0.3185 | 0.1102 | 0.1499 |
| LLM-enriched TF-IDF profile | 0.1985 | 0.3279 | 0.3774 | 0.1257 | 0.1717 |
| LLM-enriched BGE profile | 0.1757 | 0.2875 | 0.3369 | 0.1082 | 0.1480 |

This chart supports the homepage claim that LLM-enriched theorem profiles improve theorem-neighborhood retrieval.

#### Chart D: Dataset Overview

Best chart type: compact table plus proportional bars.

| Split | Theorems | Proof states | Premises | Positive edges |
| --- | ---: | ---: | ---: | ---: |
| train | 8000 | 23723 | 127561 | 54897 |
| test | 1000 | 3053 | 38332 | 7054 |

#### Chart E: Top Theorem Domains

Best chart type: horizontal bar chart.

Use test split shares:

| Domain | Test theorems | Share |
| --- | ---: | ---: |
| Algebra | 149 | 14.9% |
| Data | 146 | 14.6% |
| Analysis | 116 | 11.6% |
| Topology | 78 | 7.8% |
| RingTheory | 70 | 7.0% |
| MeasureTheory | 66 | 6.6% |
| CategoryTheory | 54 | 5.4% |
| LinearAlgebra | 50 | 5.0% |
| Order | 46 | 4.6% |
| Other | 225 | 22.5% |

## Knowledge Graph Visualization

The homepage should include a static knowledge graph visualization that explains what ProofAtlas retrieves and aggregates.

The knowledge graph should not be a complete graph dump. It should be a representative subgraph with semantically meaningful node types.

Recommended node types:

- Theorem
- Proof state
- Premise
- Theorem neighbor
- Strategy facet
- Difficulty bucket
- Domain
- Retrieval method

Recommended edge types:

- `has_proof_state`
- `uses_or_suggests_premise`
- `similar_to_theorem`
- `has_strategy_facet`
- `has_difficulty_bucket`
- `belongs_to_domain`
- `retrieved_by`
- `fused_by`

Suggested graph composition:

```text
                  Domain: AlgebraicGeometry
                              |
                              v
      AlgebraicGeometry.HasRingHomProperty.stalkwise
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
   theorem neighbors    premise suggestions   guidance profile
          |                   |                   |
          v                   v                   v
  10 similar theorems   10 suggested premises   strategy facets
                                              difficulty bucket
```

The report's example bundles can seed this view:

| Theorem | Domain | Neighbors | Premise suggestions | Strategy facets | Difficulty bucket |
| --- | --- | ---: | ---: | ---: | --- |
| AList.lookup_to_alist | Data | 10 | 10 | 8 | medium |
| AbsoluteValue.Completion.extensionEmbedding_of_comp_coe | Analysis | 10 | 10 | 8 | easy |
| Action.full_res | CategoryTheory | 10 | 10 | 5 | easy |
| AddChar.to_mulShift_inj_of_isPrimitive | NumberTheory | 10 | 10 | 8 | easy |
| AddCircle.continuousAt_equivIoc | Topology | 10 | 10 | 8 | easy |

Use two harder mathematically focused bundles as central demo graphs:

- `AlgebraicGeometry.HasRingHomProperty.stalkwise` from AlgebraicGeometry
- `Algebra.Presentation.aeval_val_relation` from RingTheory

These are better homepage examples than `AList.lookup_to_alist` because they foreground deeper mathematical dependencies: stalkwise/localization criteria in algebraic geometry, and generators-and-relations reasoning in commutative algebra.

## Demo Case Section

Use two static demo case graphs rather than an interactive selector.

Recommended demo case A:

```text
AlgebraicGeometry.HasRingHomProperty.stalkwise
Domain: AlgebraicGeometry
Difficulty: hard
Neighbors: 10
Premise suggestions: 10
Strategy facets: 8
Theme: stalkwise/localization criteria for ring-hom properties
```

Recommended demo case B:

```text
Algebra.Presentation.aeval_val_relation
Domain: RingTheory
Difficulty: hard
Neighbors: 10
Premise suggestions: 10
Strategy facets: 8
Theme: relations in algebra presentations vanish under evaluation
```

Each demo graph should show:

```text
query theorem
    |
    +--> similar theorem neighbors
    |
    +--> premise suggestions
    |
    +--> strategy facets
    |
    +--> difficulty bucket
```

Demo case captions:

- `AlgebraicGeometry.HasRingHomProperty.stalkwise`: shows how a ring-homomorphism property in algebraic geometry can be checked through stalkwise/localized conditions. This demo should emphasize scheme-local reasoning, localization, prime spectra, and morphism-property transport.
- `Algebra.Presentation.aeval_val_relation`: shows that relations declared in an algebra presentation vanish under algebraic evaluation and belong to the span/ideal generated by the presentation relations. This demo should emphasize generators, relations, polynomial evaluation, ideals, and span membership.

The report does not include the full neighbor and premise names for each example bundle in the markdown, so the design should avoid inventing those labels unless the implementation reads `outputs/proofatlas/t3_test_llm_enriched_guidance_bundles.json`.

If using only the report markdown as the data source, show counts and bundle metadata. If implementation is allowed to use the JSON artifact, expand the graph with concrete neighbor and premise names.

## Visual Style

The page should be beautiful, professional, concise, and elegant. It should read like a polished research artifact: rigorous enough for an academic audience, but clean enough for a public project homepage.

- Dense but clean layout
- White or near-white background
- Dark text with restrained accent colors
- Use color to encode node type, not decoration
- Avoid gradients, decorative blobs, and marketing-style hero sections
- Keep chart labels short and legible
- Use compact cards only for metrics and repeated evidence items
- Prefer generous whitespace around major figures, while keeping tables and metric summaries compact
- Use a restrained typography system: one sans-serif family, clear hierarchy, no oversized decorative display text
- Keep graph lines thin, node shapes simple, and labels aligned to make the visualization feel precise rather than ornamental
- Make the main graph look publication-quality: balanced spacing, consistent edge routing, no overlapping labels, no unnecessary visual effects
- Keep chart palettes muted and consistent with graph node colors
- Use concise section headings and avoid explanatory text that repeats what a chart already shows
- The overall impression should be refined and technical, not flashy

Suggested color mapping:

- Data nodes: neutral gray
- Retrieval source nodes: blue
- Theorem-neighborhood nodes: green
- LLM enrichment nodes: violet or indigo, used sparingly
- Fusion node: dark charcoal
- Metric/result nodes: amber or red accent

## Static Implementation Recommendation

Because no interaction is required, the homepage can be implemented with static HTML/CSS/SVG or a small React/Vite page.

Preferred structure if implemented later:

```text
site/
  index.html
  src/
    App.tsx
    data.ts
    components/
      SystemGraph.tsx
      ResultCharts.tsx
      KnowledgeGraph.tsx
      DemoCaseGraph.tsx
```

Recommended chart implementation:

- Use SVG or a lightweight chart library for bar and line charts.
- Keep graph visualizations as custom SVG for exact layout control.
- Store report-derived values in a small typed data file.

Avoid D3 unless dynamic graph layout becomes necessary. The design calls for a static, publication-quality graph, not an exploratory graph browser.

## Final Homepage Outline

1. Header with project name, one-line claim, and four metric cards
2. Large system pipeline graph centered on weighted RRF fusion
3. Main report result charts:
   - Dense vs primary method grouped bars
   - T1 ablation progression line chart
   - T2 theorem retrieval comparison bars
4. Dataset and domain summary
5. Knowledge graph visualization showing theorem, proof-state, premise, strategy, difficulty, and domain relationships
6. Static demo case based on one report example bundle
7. Resource links to GitHub, HuggingFace dataset, and full report
8. Short conclusion:

```text
The held-out test result supports the design hypothesis: theorem-theorem retrieval is not just a side task, but a useful source of premise evidence for Proof-State -> Premise retrieval.
```
