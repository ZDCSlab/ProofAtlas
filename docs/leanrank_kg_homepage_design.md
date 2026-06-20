# LeanRank KG Homepage Design

## Purpose

The homepage should be a research demo dashboard, not a marketing landing page.

Its goal is to let users immediately understand two things:

1. How Lean/mathlib theorem and proof objects are organized into a proof knowledge graph.
2. How a new theorem can receive proof guidance from that graph.

The page should make the project objective visible:

> Visualize Lean/mathlib proof relations as a knowledge graph, then use that graph as a retrieval and recommendation engine to help users or provers find a proof path for a new theorem.

## Core User Story

A user opens the homepage, enters or selects a theorem, and sees:

- relevant lemmas and premises
- similar theorems
- likely proof techniques
- related proof states and proof patterns
- difficulty profile
- premise ranking explanations
- the graph relations behind these recommendations

## Page Structure

### 1. Header And Theorem Input

The first screen should show the actual demo interface.

Suggested title:

```text
LeanRank Proof Knowledge Graph
```

Suggested subtitle:

```text
Visualizing Lean/mathlib proof relations and retrieving proof guidance for new theorems.
```

The header should include:

- theorem input box
- `Get Proof Guidance` action
- sample theorem buttons

Sample theorem buttons are important because users should be able to try the demo without writing Lean syntax themselves.

### 2. Knowledge Graph Overview

This section should show the proof knowledge graph directly.

It should visualize nodes such as:

- Theorem
- ProofState
- Premise
- FileModule
- ProofTechnique

It should visualize edges such as:

- theorem uses premise
- theorem has proof state
- theorem belongs to module
- theorem is similar to theorem
- proof state suggests proof technique
- proof state connects to previously used premise

The graph should include a legend explaining node and edge types.

This section should also show graph statistics, for example:

```text
Theorems: 12,000
Proof States: 48,000
Premises: 30,000
Edges: 180,000
Proof Techniques: 12
```

The purpose of this section is to make the theorem relations visible.

### 3. Proof Guidance Panel

This is the most important functional section of the homepage.

After a user enters or selects a theorem, the system should show several kinds of guidance.

#### Relevant Lemmas / Premises

Show top-k recommended lemmas, theorems, or definitions that may be useful for proving the new theorem.

Each item should include:

- premise name
- score
- short statement if available
- explanation of why it was recommended

Example explanation signals:

- shared symbols
- similar theorem statement
- similar proof state
- same namespace
- previously used as a positive premise in related proof states

#### Similar Theorems

Show mathlib theorems that are structurally or semantically close to the input theorem.

Each item should include:

- theorem name
- theorem statement
- similarity score
- shared structure, symbols, premises, or namespace

These theorems can serve as proof templates.

#### Likely Proof Techniques

Show likely proof strategy hints, such as:

- `simp`
- `rw`
- `induction`
- `cases`
- `contradiction`
- algebraic manipulation

These can come from weak proof-technique labels or retrieval from similar proof states.

#### Related Proof States / Proof Patterns

Show historical proof states whose context or goal is similar to the new theorem.

For each related proof state, show:

- goal text
- theorem source
- premises used successfully
- associated proof technique labels

The goal is to help users see reusable proof patterns.

#### Difficulty Profile

Show a compact difficulty estimate for the new theorem.

Possible features:

- goal complexity
- dependency count
- namespace or domain overlap
- symbol overlap
- negative candidate hardness
- estimated proof-search difficulty bucket

The display can use small bars, badges, or a compact chart.

#### Premise Ranking And Explanation

The system should not only return lemma names. It should explain why each lemma is likely useful.

Possible explanation signals:

- theorem statement similarity
- goal text similarity
- shared namespace
- shared symbols
- positive-premise history in similar proof states
- high baseline ranker score

## Explanation View

The homepage should include a section named something like:

```text
Why were these premises recommended?
```

This view should make the KG-based reasoning visible.

Example relation path:

```text
New Theorem
  -> similar proof state
  -> previously used premise
  -> recommended lemma
```

This section is important because it shows that the system is not a black-box recommender. It uses graph relations, retrieval scores, and historical proof usage to explain guidance.

## Pipeline Summary

The homepage should include a concise method summary.

Suggested pipeline:

```text
LeanRank Data
  -> Normalize Records
  -> Build Proof KG
  -> Add Weak Labels + Difficulty Features
  -> Generate Embeddings
  -> Train Premise Ranker
  -> Retrieve Proof Guidance
  -> Homepage Demo
```

This section helps users understand that the homepage is generated from a real data pipeline, not manually written examples.

## Evaluation And Examples

The homepage should show evidence that the system works.

Possible items:

- premise retrieval metrics
- ranking metrics such as Recall@k, MRR, or MAP
- theorem similarity examples
- at least 20 retrieval examples
- smoke test status showing the pipeline runs end to end

Example table:

| Query theorem | Top premise | Similar theorem | Suggested technique |
| --- | --- | --- | --- |
| Example theorem A | Example premise A | Example theorem B | `simp` |
| Example theorem C | Example premise C | Example theorem D | `rw` |

## Recommended Layout

The homepage should use a dashboard-style layout.

Recommended structure:

```text
Header
  Project title
  Theorem input
  Sample theorem buttons

Main demo area
  Left: knowledge graph visualization
  Right: proof guidance results

Supporting sections
  Explanation paths
  Difficulty profile
  Pipeline summary
  Evaluation examples
```

The graph visualization is the visual center.

The proof guidance panel is the functional center.

## Visual Style

The homepage should feel like an academic research tool.

Recommended style:

- clean white or light-gray background
- clear node colors by type
- compact dashboard layout
- readable theorem and premise text
- no decorative hero-only section
- no purely marketing-style introduction
- all demo results should come from real files under `homepage/assets/`

Suggested generated assets:

```text
homepage/assets/graph_stats.json
homepage/assets/metrics.json
homepage/assets/retrieval_examples.json
homepage/assets/domain_coverage.json
```

## Final Homepage Goal

The homepage should demonstrate both project goals:

1. Show the Lean/mathlib proof knowledge graph and its theorem relations.
2. Show how a new theorem receives proof guidance from the graph.

In one sentence:

> The homepage should be an interactive proof guidance demo: it visualizes the Lean/mathlib proof knowledge graph and shows how a new theorem can receive relevant premises, similar theorems, proof techniques, proof patterns, difficulty signals, and ranking explanations.
