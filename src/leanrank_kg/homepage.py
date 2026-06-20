from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from .report import build_summary
from .utils import write_json

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{{ dataset.project_name }}</title>
  <style>
    :root{
      color-scheme:light;
      --ink:#17202a;--muted:#657180;--line:#d9e0e8;--bg:#f5f7fb;--panel:#fff;
      --blue:#2563eb;--green:#0f8a5f;--teal:#0e7490;--amber:#b7791f;--rose:#be123c;
      --soft-blue:#e8f0ff;--soft-green:#e7f6ee;--soft-amber:#fff4db;--soft-teal:#e6f6f8;
    }
    *{box-sizing:border-box}
    body{font-family:Inter,Arial,sans-serif;margin:0;color:var(--ink);background:var(--bg);line-height:1.45}
    header{background:#182433;color:white;padding:34px 7vw 28px;border-bottom:5px solid #2dd4bf}
    header h1{font-size:48px;line-height:1;margin:0 0 12px;letter-spacing:0}
    header p{max-width:900px;color:#dbe6f3;font-size:18px;margin:0}
    header .demo-form{margin-top:18px;max-width:980px}
    header .demo-form label{color:#dbe6f3}
    header .demo-form textarea,header .demo-form input,header .demo-form select{border-color:#42546b;background:#f8fbff;color:var(--ink)}
    header .sample-row{display:flex;gap:8px;flex-wrap:wrap}
    main{max-width:1220px;margin:auto;padding:20px}
    section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:22px;margin:18px 0}
    h2,h3{margin:0 0 12px;letter-spacing:0}
    h2{font-size:22px}
    h3{font-size:16px;color:#263547}
    table{border-collapse:collapse;width:100%;font-size:14px}
    td,th{border-bottom:1px solid #e6ebf1;padding:8px;text-align:left;vertical-align:top}
    th{color:#3b4654;background:#f7f9fc}
    code{background:#eef2f7;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}
    .muted{color:var(--muted)}
    .eyebrow{text-transform:uppercase;letter-spacing:.08em;font-size:12px;color:#8fb3d9;font-weight:700;margin-bottom:8px}
    .hero-grid{display:grid;grid-template-columns:1.3fr .9fr;gap:18px;margin-top:22px}
    .hero-panel{border:1px solid rgba(255,255,255,.18);border-radius:8px;padding:16px;background:rgba(255,255,255,.06)}
    .hero-panel h2{color:white}
    .hero-panel .chip{color:#27364a}
    .kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
    .kpi{background:white;border:1px solid var(--line);border-radius:8px;padding:14px;min-height:104px}
    .kpi.dark{background:#223047;border-color:#34445f;color:white}
    .label{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:700}
    .dark .label{color:#b8c7da}
    .value{font-size:30px;font-weight:800;line-height:1.1;overflow-wrap:anywhere;margin-top:6px}
    .note{font-size:13px;color:var(--muted);margin-top:6px}
    .dark .note{color:#d0d9e6}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}
    .two-col{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}
    .workbench{display:grid;grid-template-columns:minmax(0,1.12fr) minmax(360px,.88fr);gap:16px;align-items:start}
    .status-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
    .status{border:1px solid var(--line);border-radius:8px;padding:12px;background:#fbfcfe}
    .status.ok{border-color:#badfcb;background:var(--soft-green)}
    .status.warn{border-color:#f1d49a;background:var(--soft-amber)}
    .pill{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:5px 9px;font-size:12px;font-weight:700;background:#edf2f7;color:#27364a}
    .pill.ok{background:var(--soft-green);color:#166534}
    .pill.info{background:var(--soft-blue);color:#1d4ed8}
    .chips{display:flex;flex-wrap:wrap;gap:8px}
    .chip{border:1px solid var(--line);border-radius:999px;padding:5px 10px;background:#fafcff;font-size:13px}
    .chart{display:grid;gap:10px}
    .bar-row{display:grid;grid-template-columns:126px minmax(0,1fr) 84px;gap:10px;align-items:center;font-size:14px}
    .bar-track{height:12px;background:#edf2f7;border-radius:999px;overflow:hidden}
    .bar{height:100%;border-radius:999px;background:var(--blue)}
    .bar.green{background:var(--green)}
    .bar.teal{background:var(--teal)}
    .bar.amber{background:var(--amber)}
    .bar.rose{background:var(--rose)}
    .split-chart{display:grid;gap:14px}
    .split-item{border:1px solid var(--line);border-radius:8px;padding:12px;background:#fbfcfe}
    .split-head{display:flex;justify-content:space-between;gap:12px;align-items:baseline;margin-bottom:8px}
    .mini-bars{display:grid;gap:7px}
    .mini-bar{display:grid;grid-template-columns:48px minmax(0,1fr) 88px;gap:8px;align-items:center;font-size:13px}
    .example{border:1px solid var(--line);border-radius:8px;padding:14px;background:#fbfcfe;margin:12px 0}
    .goal{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f2f6fb;border:1px solid #dce5ef;border-radius:6px;padding:10px;overflow-wrap:anywhere}
    .ranked{display:grid;gap:8px;margin-top:10px}
    .rank{display:grid;grid-template-columns:minmax(150px,1fr) minmax(0,1.2fr) 54px;gap:10px;align-items:center;font-size:13px}
    .demo-form{display:grid;gap:12px}
    .demo-form label{display:grid;gap:6px;font-size:13px;font-weight:700;color:#334155}
    .demo-form textarea,.demo-form input,.demo-form select{width:100%;border:1px solid var(--line);border-radius:6px;padding:9px;font:inherit;background:white;color:var(--ink)}
    .demo-form textarea{min-height:136px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;resize:vertical}
    .demo-controls{display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:10px}
    .demo-actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
    .button{border:1px solid #174ea6;background:var(--blue);color:white;border-radius:6px;padding:9px 13px;font-weight:800;cursor:pointer}
    .button.secondary{border-color:var(--line);background:white;color:#27364a}
    .demo-output{border:1px solid var(--line);border-radius:8px;background:#fbfcfe;padding:12px;min-height:120px;display:grid;gap:10px}
    .guidance-panel{min-height:360px}
    .guidance-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
    .guidance-block{border:1px solid var(--line);border-radius:8px;background:#fbfcfe;padding:12px;min-width:0}
    .relation-path{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f7fafc;border:1px solid var(--line);border-radius:8px;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere}
    .demo-error{color:var(--rose);font-weight:700}
    .graph-view{border:1px solid var(--line);border-radius:8px;background:#fbfcfe;min-height:360px;overflow:auto}
    .graph-svg{width:100%;min-width:760px;height:360px;display:block}
    .graph-edge{stroke:#9aa8b8;stroke-width:1.2;opacity:.72;cursor:pointer}
    .graph-edge.active{stroke:var(--rose);stroke-width:2.6;opacity:1}
    .graph-node{stroke:white;stroke-width:1.5;cursor:pointer}
    .graph-node.active{stroke:var(--ink);stroke-width:3}
    .graph-label{font-size:11px;fill:#263547;pointer-events:none}
    .legend{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
    .legend span{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}
    .swatch{width:10px;height:10px;border-radius:50%;display:inline-block}
    .edge-swatch{width:24px;height:3px;border-radius:999px;display:inline-block;background:#9aa8b8}
    .graph-detail{border:1px solid var(--line);border-radius:8px;background:#fff;padding:10px;margin-top:10px;min-height:76px}
    .timeline{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}
    .step{border-left:4px solid var(--teal);background:#f8fbfd;border-radius:8px;padding:12px;border-top:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
    .section-lead{color:var(--muted);max-width:850px;margin:0 0 16px}
    @media (max-width:900px){
      header h1{font-size:38px}
      .hero-grid,.two-col,.timeline,.workbench,.guidance-grid{grid-template-columns:1fr}
      .kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
    }
    @media (max-width:560px){
      main{padding:12px}
      section{padding:16px}
      .kpi-grid{grid-template-columns:1fr}
      .bar-row,.mini-bar,.rank{grid-template-columns:1fr}
      .demo-controls{grid-template-columns:1fr}
      .bar-track{height:14px}
    }
  </style>
</head>
<body>
<header>
  <div class="eyebrow">Research demo dashboard</div>
  <h1>LeanRank Proof Knowledge Graph</h1>
  <p>Visualizing Lean/mathlib proof relations and retrieving proof guidance for new theorems.</p>
  <div class="demo-form">
    <label>Lean theorem or proof goal
      <textarea id="api-theorem">theorem Nat.self_eq (x : Nat) : x = x := by
  simpa</textarea>
    </label>
    <div class="sample-row" id="sample-theorems"></div>
    <div class="demo-controls">
      <label>API URL
        <input id="api-url" value="http://127.0.0.1:8000/retrieve-theorem-guidance">
      </label>
      <label>Input type
        <select id="api-input-type">
          <option value="lean">lean</option>
          <option value="theorem">theorem</option>
          <option value="proof_state">proof_state</option>
          <option value="goal">goal</option>
        </select>
      </label>
      <label>Premises
        <input id="api-k-premises" type="number" min="1" max="50" value="10">
      </label>
      <label>Theorems
        <input id="api-k-theorems" type="number" min="1" max="20" value="5">
      </label>
    </div>
    <div class="demo-actions">
      <button class="button" id="api-submit" type="button">Get Proof Guidance</button>
      <button class="button secondary" id="api-metrics" type="button">Metrics</button>
      <button class="button secondary" id="api-clear" type="button">Clear</button>
      <label class="muted"><input id="api-validate-lean" type="checkbox"> Lean check</label>
      <span class="muted" id="api-status">Idle</span>
    </div>
  </div>
</header>
<main>
  <section>
    <h2>Interactive Proof Guidance Workbench</h2>
    <p class="section-lead">The graph visualization and guidance panel are backed by generated files in <code>homepage/assets/</code>. Sample theorem buttons load precomputed guidance; the main action can call a local API server for live retrieval.</p>
    <div class="workbench">
      <div>
        <h3>Knowledge Graph Overview</h3>
        <div class="graph-view">
          <svg class="graph-svg" id="kg-svg" viewBox="0 0 900 360" role="img" aria-label="LeanRank proof knowledge graph sample"></svg>
        </div>
        <div class="legend" id="kg-legend"></div>
        <h3 style="margin-top:12px">Edge Types</h3>
        <div class="legend" id="kg-edge-legend"></div>
        <div class="graph-detail" id="kg-detail">
          <span class="muted">Click a graph node or edge to inspect its proof relation.</span>
        </div>
        <div class="status-row" style="margin-top:12px">
          <div class="status"><div class="label">Theorems</div><div class="value">{{ overview.total_theorems }}</div><div class="note">train/val/test</div></div>
          <div class="status"><div class="label">Proof states</div><div class="value">{{ overview.train_proof_states }}</div><div class="note">train split</div></div>
          <div class="status"><div class="label">Premises</div><div class="value">{{ overview.train_premises }}</div><div class="note">train split</div></div>
          <div class="status"><div class="label">Edges</div><div class="value">{{ overview.total_edges }}</div><div class="note">typed graph relations</div></div>
        </div>
      </div>
      <div>
        <h3>Proof Guidance Panel</h3>
        <div class="demo-output guidance-panel" id="api-output">
          <span class="muted">Choose a sample theorem or run live retrieval.</span>
        </div>
      </div>
    </div>
  </section>

  <section>
    <h2>Executive Snapshot</h2>
    <div class="kpi-grid">
      <div class="kpi"><div class="label">Theorem nodes</div><div class="value">{{ overview.total_theorems }}</div><div class="note">{{ dataset.sample_rows }} LeanRank rows sampled</div></div>
      <div class="kpi"><div class="label">Train premises</div><div class="value">{{ overview.train_premises }}</div><div class="note">{{ overview.train_proof_states }} proof states</div></div>
      <div class="kpi"><div class="label">Graph scale</div><div class="value">{{ overview.total_edges }}</div><div class="note">{{ overview.total_nodes }} nodes across splits</div></div>
      <div class="kpi dark"><div class="label">Embedding model</div><div class="value">{{ embedding.model_name }}</div><div class="note">{{ embedding.backend }} on {{ embedding.device }}, batch {{ embedding.batch_size }}</div></div>
    </div>
  </section>

  <section>
    <h2>Data And Graph Scale</h2>
    <p class="section-lead">Split-level chart showing how much searchable proof structure is available to the demo and evaluation pipeline.</p>
    <div class="split-chart">
      {% for row in dataset.split_counts %}
      <div class="split-item">
        <div class="split-head"><h3>{{ row.split }}</h3><span class="muted">{{ row.theorems }} theorems, {{ row.proof_states }} proof states, {{ row.premises }} premises</span></div>
        <div class="mini-bars">
          <div class="mini-bar"><span>Nodes</span><div class="bar-track"><div class="bar teal" style="width:{{ (row.nodes / overview.max_split_nodes * 100)|round(1) }}%"></div></div><b>{{ row.nodes }}</b></div>
          <div class="mini-bar"><span>Edges</span><div class="bar-track"><div class="bar green" style="width:{{ (row.edges / overview.max_split_edges * 100)|round(1) }}%"></div></div><b>{{ row.edges }}</b></div>
        </div>
      </div>
      {% endfor %}
    </div>
  </section>

  <section>
    <h2>Coverage Charts</h2>
    <div class="two-col">
      <div>
        <h3>Top Mathlib Domains</h3>
        <div class="chart">
          {% for row in top_domains %}
          <div class="bar-row"><span>{{ row.domain }}</span><div class="bar-track"><div class="bar" style="width:{{ (row.count / overview.max_domain_count * 100)|round(1) }}%"></div></div><b>{{ row.count }}</b></div>
          {% endfor %}
        </div>
      </div>
      <div>
        <h3>Difficulty Buckets</h3>
        <div class="chart">
          {% for row in difficulty_distribution %}
          <div class="bar-row"><span>{{ row.split }} / {{ row.bucket }}</span><div class="bar-track"><div class="bar amber" style="width:{{ (row.count / overview.max_difficulty_count * 100)|round(1) }}%"></div></div><b>{{ row.count }}</b></div>
          {% endfor %}
        </div>
      </div>
    </div>
  </section>

  <section>
    <h2>Retrieval Quality Signals</h2>
    <div class="kpi-grid">
      <div class="kpi"><div class="label">AUC</div><div class="value">{{ "%.3f"|format(metrics.get("AUC") or 0) }}</div><div class="note">Ranker validation</div></div>
      <div class="kpi"><div class="label">Recall@10</div><div class="value">{{ "%.3f"|format(metrics.get("Recall@10", 0)) }}</div><div class="note">Gold premise in top 10</div></div>
      <div class="kpi"><div class="label">Theorem Recall@10</div><div class="value">{{ "%.3f"|format(metrics.get("theorem_retrieval_Recall@10", 0)) }}</div><div class="note">{{ metrics.get("theorem_retrieval_evaluated_theorems", 0) }} held-out theorems</div></div>
      <div class="kpi"><div class="label">Train gold premises</div><div class="value">{{ metrics.get("theorem_retrieval_gold_premises_in_train_index", 0) }}</div><div class="note">{{ metrics.get("theorem_retrieval_gold_premises_missing_from_train_index", 0) }} held-out gold premises missing</div></div>
    </div>
  </section>

  <section>
    <h2>Production Evidence</h2>
    {% set timing_delta = production_evidence.timing.evaluation_timing_delta|default({}) %}
    <p class="section-lead">The demo is backed by committed LeanRank-data reports, including full held-out retrieval metrics, premise supervision counts, and explicit timing freshness checks.</p>
    <div class="kpi-grid">
      <div class="kpi"><div class="label">Proof-state test coverage</div><div class="value">{{ "{:,}".format(production_evidence.heldout.proof_state_evaluated_queries|default(0) or 0) }}</div><div class="note">full held-out proof-state queries</div></div>
      <div class="kpi"><div class="label">Theorem test coverage</div><div class="value">{{ "{:,}".format(production_evidence.heldout.theorem_evaluated_queries|default(0) or 0) }}</div><div class="note">full held-out theorem queries</div></div>
      <div class="kpi"><div class="label">Theorem Recall@100</div><div class="value">{{ "%.3f"|format(production_evidence.heldout.theorem_recall_at_100 or 0) }}</div><div class="note">premise retrieval for theorem guidance</div></div>
      <div class="kpi"><div class="label">Proof-state Recall@100</div><div class="value">{{ "%.3f"|format(production_evidence.heldout.proof_state_recall_at_100 or 0) }}</div><div class="note">premise retrieval from proof states</div></div>
      <div class="kpi"><div class="label">Positive premise edges</div><div class="value">{{ "{:,}".format(production_evidence.supervision.total_positive_edges|default(0) or 0) }}</div><div class="note">LeanRank positive supervision</div></div>
      <div class="kpi"><div class="label">Negative candidates</div><div class="value">{{ "{:,}".format(production_evidence.supervision.total_negative_edges|default(0) or 0) }}</div><div class="note">hard-negative ranking pool</div></div>
      <div class="kpi"><div class="label">Pipeline timing</div><div class="value">{{ "%.1f"|format(production_evidence.timing.total_seconds or 0) }}s</div><div class="note">{{ production_evidence.timing.executed_stage_count|default(0) }} executed / {{ production_evidence.timing.skipped_stage_count|default(0) }} skipped stages</div></div>
      <div class="kpi"><div class="label">Scale estimate reliable</div><div class="value">{{ production_evidence.timing.scale_estimate_reliable }}</div><div class="note">{{ production_evidence.timing.throughput_basis|default("unknown") }} on {{ production_evidence.timing.embedding_device|default("unknown") }}</div></div>
      <div class="kpi"><div class="label">Current eval timing</div><div class="value">{{ "%.1f"|format(timing_delta.current_evaluation_seconds|default(0) or 0) }}s</div><div class="note">standalone held-out evaluation</div></div>
      <div class="kpi"><div class="label">Pipeline eval timing</div><div class="value">{{ "%.1f"|format(timing_delta.timed_pipeline_evaluate_seconds|default(0) or 0) }}s</div><div class="note">saved full-pipeline evaluate stage</div></div>
    </div>
    <div class="status-row" style="margin-top:12px">
      <div class="status ok"><div class="label">Train positive coverage</div><div class="value">{{ "%.3f"|format(production_evidence.supervision.train_positive_proof_state_coverage or 0) }}</div><div class="note">proof states with positive premise edges</div></div>
      <div class="status ok"><div class="label">Train negative coverage</div><div class="value">{{ "%.3f"|format(production_evidence.supervision.train_negative_proof_state_coverage or 0) }}</div><div class="note">proof states with negative candidates</div></div>
      <div class="status"><div class="label">Hardness mean</div><div class="value">{{ "%.3f"|format(production_evidence.supervision.train_negative_hardness_mean or 0) }}</div><div class="note">train negative candidate hardness</div></div>
      <div class="status"><div class="label">Embedding throughput</div><div class="value">{{ "%.1f"|format(production_evidence.timing.embedding_rows_per_second or 0) }}</div><div class="note">embedding rows per second</div></div>
      {% if timing_delta.current_faster_than_pipeline_timing %}
      <div class="status warn"><div class="label">Timing freshness</div><div class="value">{{ "%.1f"|format(timing_delta.timed_to_current_ratio|default(0) or 0) }}x</div><div class="note">rerun production timing before final bottleneck claims</div></div>
      {% else %}
      <div class="status ok"><div class="label">Timing freshness</div><div class="value">current</div><div class="note">pipeline and standalone evaluation timings align</div></div>
      {% endif %}
    </div>
    {% set bottleneck = production_evidence.timing.bottleneck_profile|default({}) %}
    {% set bottleneck_stages = bottleneck.top_stages|default([]) %}
    {% set evaluation_timing = production_evidence.timing.evaluation_timing|default({}) %}
    {% set evaluation_substages = evaluation_timing.slowest_substages|default([]) %}
    <div class="status-row" style="margin-top:12px">
      <div class="status warn"><div class="label">Pipeline bottleneck</div><div class="value">{{ bottleneck.primary_stage|default(production_evidence.timing.slowest_stage|default("unknown")) }}</div><div class="note">{{ "%.1f"|format((bottleneck.primary_stage_share_of_total|default(0) or 0) * 100) }}% of timed production run</div></div>
      <div class="status"><div class="label">Top-3 timed stages</div><div class="value">{{ "%.1f"|format((bottleneck.top3_stage_share_of_total|default(0) or 0) * 100) }}%</div><div class="note">combined share of pipeline wall time</div></div>
      {% if evaluation_substages %}
      <div class="status warn"><div class="label">Evaluation bottleneck</div><div class="value">{{ evaluation_substages[0].name }}</div><div class="note">{{ "%.1f"|format(evaluation_substages[0].seconds|default(0) or 0) }}s inside evaluation</div></div>
      <div class="status"><div class="label">Eval timed substages</div><div class="value">{{ evaluation_timing.substage_count|default(0) }}</div><div class="note">{{ "%.1f"|format(evaluation_timing.total_seconds|default(0) or 0) }}s measured evaluation total</div></div>
      {% endif %}
      {% for row in bottleneck_stages[:2] %}
      <div class="status"><div class="label">{{ row.name }}</div><div class="value">{{ "%.1f"|format(row.seconds|default(0) or 0) }}s</div><div class="note">{{ "%.1f"|format((row.share_of_total|default(0) or 0) * 100) }}% of timed run</div></div>
      {% endfor %}
    </div>
    {% set failure_profile = production_evidence.failure_profile|default({}) %}
    {% set proof_failure = failure_profile.proof_state|default({}) %}
    {% set theorem_failure = failure_profile.theorem|default({}) %}
    {% set reranked_failure = failure_profile.reranked_proof_state|default({}) %}
    <h3 style="margin-top:16px">Retrieval Failure Profile</h3>
    <div class="status-row">
      <div class="status warn"><div class="label">Proof-state miss top {{ proof_failure.max_k|default(100) }}</div><div class="value">{{ "{:,}".format(proof_failure.zero_recall_at_max_k|default(0) or 0) }}</div><div class="note">{{ "{:,}".format(proof_failure.retrievable_queries|default(0) or 0) }} retrievable proof-state queries</div></div>
      <div class="status"><div class="label">Proof-state no train gold</div><div class="value">{{ "{:,}".format(proof_failure.queries_without_train_gold|default(0) or 0) }}</div><div class="note">held-out queries with no train-side gold premise</div></div>
      <div class="status ok"><div class="label">Theorem miss top {{ theorem_failure.max_k|default(100) }}</div><div class="value">{{ "{:,}".format(theorem_failure.zero_recall_at_max_k|default(0) or 0) }}</div><div class="note">{{ "{:,}".format(theorem_failure.retrievable_queries|default(0) or 0) }} retrievable theorem queries</div></div>
      <div class="status"><div class="label">Rerank miss top {{ reranked_failure.max_k|default(10) }}</div><div class="value">{{ "{:,}".format(reranked_failure.zero_recall_at_max_k|default(0) or 0) }}</div><div class="note">homepage/API-style diagnostic queries</div></div>
    </div>
    {% set zero_domains = proof_failure.zero_recall_domains|default([]) %}
    {% if zero_domains %}
    <div class="chips" style="margin-top:12px">
      {% for row in zero_domains[:6] %}
      <span class="chip">{{ row.domain_tag }}: {{ row.zero_recall_queries }}</span>
      {% endfor %}
    </div>
    {% endif %}
    {% set rapid = production_evidence.rapid_convergence|default({}) %}
    {% set rapid_headroom = rapid.headroom|default({}) %}
    {% set rapid_steps = rapid.recommended_sequence|default([]) %}
    {% set refresh_reuse = production_evidence.refresh_reuse|default({}) %}
    {% set artifact_cache = refresh_reuse.artifact_cache|default({}) %}
    <h3 style="margin-top:16px">Refresh And Retraining Policy</h3>
    <div class="status-row">
      <div class="status ok"><div class="label">Reuse by default</div><div class="value">{{ refresh_reuse.reuse_by_default|default(false) }}</div><div class="note">report/homepage refreshes reuse existing artifacts</div></div>
      <div class="status"><div class="label">Cached embeddings</div><div class="value">{{ "{:,}".format(artifact_cache.embedding_rows|default(0) or 0) }}</div><div class="note">{{ artifact_cache.embedding_model|default("unknown") }}</div></div>
      <div class="status"><div class="label">Indexed manifests</div><div class="value">{{ artifact_cache.indexed_entity_count|default(0) }}</div><div class="note">{{ artifact_cache.index_backend|default("unknown") }} retrieval artifacts</div></div>
      <div class="status {{ 'ok' if artifact_cache.premise_ranker_exists else 'warn' }}"><div class="label">Premise ranker</div><div class="value">{{ artifact_cache.premise_ranker_exists|default(false) }}</div><div class="note">retrain only after feature, label, split, or config changes</div></div>
    </div>
    <p class="muted">{{ refresh_reuse.training_repeat_policy|default("Training policy unavailable.") }}</p>
    {% if rapid_steps %}
    <h3 style="margin-top:16px">Rapid Convergence Priorities</h3>
    <div class="status-row">
      <div class="status warn"><div class="label">Proof-state top-100 miss</div><div class="value">{{ "%.3f"|format(rapid_headroom.proof_state_missing_from_top100|default(0) or 0) }}</div><div class="note">candidate-generation headroom</div></div>
      <div class="status"><div class="label">Theorem top-10/100 gap</div><div class="value">{{ "%.3f"|format(rapid_headroom.theorem_top10_to_top100_gap|default(0) or 0) }}</div><div class="note">reranking headroom</div></div>
      {% for row in rapid_steps[:4] %}
      <div class="status"><div class="label">Priority {{ row.priority }}</div><div class="value">{{ row.area }}</div><div class="note">{{ row.target_metric }}: {{ "%.3f"|format(row.current_value|default(0) or 0) }}</div></div>
      {% endfor %}
    </div>
    {% endif %}
  </section>

  <section>
    <h2>Refresh Dashboard</h2>
    <p class="section-lead">A compact post-refresh health view combining artifact compatibility, KG scale, parsing coverage, retrieval quality, index benchmark, and difficulty calibration signals.</p>
    <div class="status-row">
      {% for name, passed in refresh_dashboard.quality_gates.items() %}
      <div class="status {{ 'ok' if passed else 'warn' }}"><div class="label">{{ name.replace('_', ' ') }}</div><div class="value">{{ 'pass' if passed else 'check' }}</div><div class="note">refresh quality gate</div></div>
      {% endfor %}
    </div>
    <div class="kpi-grid" style="margin-top:12px">
      <div class="kpi"><div class="label">Refresh ready</div><div class="value">{{ refresh_dashboard.ready_for_refresh_comparison }}</div><div class="note">all dashboard gates passing</div></div>
      <div class="kpi"><div class="label">Supervision</div><div class="value">{{ refresh_dashboard.corpus.data_supervision.kind|default("unknown") }}</div><div class="note">proof traces: {{ refresh_dashboard.corpus.data_supervision.has_tactic_states|default(false) }}</div></div>
      <div class="kpi"><div class="label">Min parse coverage</div><div class="value">{{ "%.3f"|format(refresh_dashboard.parsing.minimum_context_coverage or 0) }}</div><div class="note">context_parse_coverage</div></div>
      <div class="kpi"><div class="label">Index recall</div><div class="value">{{ "%.3f"|format(refresh_dashboard.index_benchmark.get("premise", {}).get("recall_vs_exact") or 0) }}</div><div class="note">premise index vs exact cosine</div></div>
      <div class="kpi"><div class="label">Difficulty MAE</div><div class="value">{{ "%.3f"|format(refresh_dashboard.difficulty.train_mae or 0) }}</div><div class="note">train estimator calibration</div></div>
      <div class="kpi"><div class="label">Trend baseline</div><div class="value">{{ 'yes' if refresh_trend.has_previous else 'no' }}</div><div class="note">previous refresh snapshot</div></div>
      <div class="kpi"><div class="label">Theorem delta</div><div class="value">{{ "%.0f"|format(refresh_trend.deltas.get("theorems", {}).get("absolute") or 0) }}</div><div class="note">change from previous dashboard</div></div>
      <div class="kpi"><div class="label">Recall delta</div><div class="value">{{ "%.3f"|format(refresh_trend.deltas.get("theorem_recall_at_10", {}).get("absolute") or 0) }}</div><div class="note">theorem Recall@10 change</div></div>
      <div class="kpi"><div class="label">Gate changes</div><div class="value">{{ refresh_trend.quality_gate_changes|length }}</div><div class="note">quality gates changed</div></div>
      <div class="kpi"><div class="label">History entries</div><div class="value">{{ refresh_history.entry_count }}</div><div class="note">bounded refresh trend history</div></div>
    </div>
    {% if refresh_dashboard.artifact_compatibility.warnings %}
    <div class="chips" style="margin-top:12px">
      {% for warning in refresh_dashboard.artifact_compatibility.warnings %}
      <span class="chip">{{ warning.replace('_', ' ') }}</span>
      {% endfor %}
    </div>
    {% endif %}
  </section>

  <section>
    <h2>New Theorem Proof Guidance</h2>
    <p class="section-lead">Precomputed held-out theorem case studies show the end-to-end guidance bundle: premises, similar theorems, likely techniques, related proof patterns, difficulty, and graph evidence.</p>
    {% for case in theorem_case_studies[:3] %}
    <div class="example">
      <h3>{{ case.full_name }}</h3>
      <p class="muted">Gold premises in train index: <b>{{ case.gold_premises_in_train_index }}</b> / {{ case.gold_positive_premise_count }} | missing from train: <b>{{ case.gold_premises_missing_from_train_index }}</b> | coverage: <b>{{ "%.3f"|format(case.gold_premise_train_coverage) }}</b></p>
      <div class="goal">{{ case.guidance.query.theorem_text[:420] }}</div>
      <h3>Relevant Premises</h3>
      <div class="ranked">
        {% for r in case.guidance.ranked_premises[:5] %}
        <div class="rank">
          <code>{{ r.full_name }}</code>
          <div class="bar-track"><div class="bar green" style="width:{{ (r.score * 100)|round(1) }}%"></div></div>
          <b>{{ "%.3f"|format(r.score) }}</b>
        </div>
        {% if r.ranking_reasons %}
        <p class="muted">{{ r.ranking_reasons[:2]|join("; ") }}</p>
        {% endif %}
        {% endfor %}
      </div>
      <h3>Similar Theorems</h3>
      <div class="ranked">
        {% for r in case.guidance.similar_theorems[:3] %}
        <div class="rank">
          <code>{{ r.full_name }}</code>
          <div class="bar-track"><div class="bar teal" style="width:{{ (r.score * 100)|round(1) }}%"></div></div>
          <b>{{ "%.3f"|format(r.score) }}</b>
        </div>
        {% endfor %}
      </div>
      <p class="muted">Difficulty: <b>{{ case.guidance.difficulty_profile.difficulty_bucket }}</b> / {{ "%.3f"|format(case.guidance.difficulty_profile.difficulty_score) }} | {{ case.guidance.difficulty_profile.signals.calibrated_by|default("query_heuristic") }}</p>
    </div>
    {% endfor %}
  </section>

  <section>
    <h2>Evaluation And Examples</h2>
    <p class="section-lead">Compact held-out guidance examples summarize whether the pipeline returns a plausible premise, a proof-template theorem, and a proof-technique hint for each query theorem.</p>
    <div class="status-row" style="margin-bottom:12px">
      <div class="status"><div class="label">Case studies</div><div class="value">{{ theorem_case_studies|length }}</div><div class="note">precomputed theorem guidance examples</div></div>
      <div class="status"><div class="label">Retrieval examples</div><div class="value">{{ retrieval_examples|length }}</div><div class="note">proof-state premise queries</div></div>
      <div class="status"><div class="label">Theorem Recall@10</div><div class="value">{{ "%.3f"|format(metrics.get("theorem_retrieval_Recall@10", 0)) }}</div><div class="note">held-out theorem retrieval</div></div>
      <div class="status"><div class="label">Premise Recall@10</div><div class="value">{{ "%.3f"|format(metrics.get("Recall@10", 0)) }}</div><div class="note">premise ranking metric</div></div>
    </div>
    <table>
      <tr><th>Query theorem</th><th>Top premise</th><th>Similar theorem</th><th>Suggested technique</th></tr>
      {% for case in theorem_case_studies[:8] %}
      {% set top_premise = case.guidance.ranked_premises[0] if case.guidance.ranked_premises else {} %}
      {% set top_theorem = case.guidance.similar_theorems[0] if case.guidance.similar_theorems else {} %}
      {% set top_technique = case.guidance.likely_proof_techniques[0] if case.guidance.likely_proof_techniques else {} %}
      <tr>
        <td><code>{{ case.full_name }}</code></td>
        <td><code>{{ top_premise.full_name|default("none") }}</code><br><span class="muted">{{ "%.3f"|format(top_premise.score|default(0)) }}</span></td>
        <td><code>{{ top_theorem.full_name|default("none") }}</code><br><span class="muted">{{ "%.3f"|format(top_theorem.score|default(0)) }}</span></td>
        <td><code>{{ top_technique.label|default("unknown") }}</code></td>
      </tr>
      {% endfor %}
    </table>
  </section>

  <section>
    <h2>Why Were These Premises Recommended?</h2>
    <p class="section-lead">The demo exposes graph and retrieval evidence instead of only showing lemma names. A typical explanation path is:</p>
    <div class="relation-path">New Theorem
  -> similar proof state
  -> previously used premise
  -> recommended lemma</div>
    <div class="status-row" style="margin-top:12px">
      <div class="status"><div class="label">Text retrieval</div><div class="value">goal</div><div class="note">statement and proof-state similarity</div></div>
      <div class="status"><div class="label">Graph evidence</div><div class="value">paths</div><div class="note">theorem, proof-state, premise relations</div></div>
      <div class="status"><div class="label">Ranking reasons</div><div class="value">signals</div><div class="note">namespace, symbols, frequency, learned score</div></div>
      <div class="status"><div class="label">Difficulty</div><div class="value">profile</div><div class="note">complexity and historical prior</div></div>
    </div>
  </section>

  <section>
    <h2>Live Retrieval Examples</h2>
    <p class="section-lead">Each example starts from a Lean proof goal and retrieves likely useful premises from the train index using BGE cosine similarity.</p>
    {% for ex in retrieval_examples[:4] %}
    <div class="example">
      <h3>{{ ex.proof_state_id }}</h3>
      <div class="goal">{{ ex.proof_state }}</div>
      <p class="muted">Gold premise: <code>{{ ex.gold_positive_premise }}</code> | in train index: <b>{{ ex.gold_in_train_index }}</b></p>
      <div class="ranked">
        {% for r in ex.top_retrieved_premises %}
        <div class="rank">
          <code>{{ r.full_name }}</code>
          <div class="bar-track"><div class="bar green" style="width:{{ (r.score * 100)|round(1) }}%"></div></div>
          <b>{{ "%.3f"|format(r.score) }}</b>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endfor %}
  </section>

  <section>
    <h2>Proof Technique Labels</h2>
    <p class="section-lead">Weak labels make the graph browsable by proof style, not just by theorem names and embeddings.</p>
    <div class="chart">
      {% for row in proof_techniques[:16] %}
      <div class="bar-row"><span>{{ row.split }} / {{ row.label }}</span><div class="bar-track"><div class="bar rose" style="width:{{ (row.count / overview.max_technique_count * 100)|round(1) }}%"></div></div><b>{{ row.count }}</b></div>
      {% endfor %}
    </div>
  </section>

  <section>
    <h2>Pipeline Summary</h2>
    <p class="section-lead">LeanRank Data -> Normalize Records -> Build Proof KG -> Add Weak Labels + Difficulty Features -> Generate Embeddings -> Train Premise Ranker -> Retrieve Proof Guidance -> Homepage Demo.</p>
    <div class="timeline">
      {% for fn in functions %}
      <div class="step"><h3><code>{{ fn }}</code></h3><p class="muted">Ready in the Python package and represented by homepage artifacts.</p></div>
      {% endfor %}
    </div>
  </section>

  <section>
    <h2>Validation Matrix</h2>
    <div class="status-row">
      <div class="status ok"><div class="label">Schema</div><div class="value">{{ overview.schema_errors }}</div><div class="note">validation errors</div></div>
      <div class="status ok"><div class="label">Split leakage</div><div class="value">{{ overview.split_leakage }}</div><div class="note">false means clean split boundaries</div></div>
      <div class="status ok"><div class="label">Graph endpoints</div><div class="value">{{ overview.missing_graph_endpoints }}</div><div class="note">missing source/target ids</div></div>
      <div class="status ok"><div class="label">NetworkX loadable</div><div class="value">{{ overview.networkx_ok }}</div><div class="note">all split graphs parse</div></div>
    </div>
    <table>
      <tr><th>Split</th><th>NetworkX loadable</th><th>Missing endpoints</th><th>Nodes</th><th>Edges</th></tr>
      {% for split, row in validation.graph.items() %}
      <tr><td>{{ split }}</td><td>{{ row.networkx_loadable }}</td><td>{{ row.missing_endpoint_count }}</td><td>{{ row.node_count }}</td><td>{{ row.edge_count }}</td></tr>
      {% endfor %}
    </table>
  </section>

  <section>
    <h2>Reproducibility</h2>
    <p class="section-lead">The public repo keeps source code, configs, homepage assets, and tests. Large raw data, generated embeddings, and graph artifacts are intentionally reproducible rather than checked into git.</p>
    <div class="status-row">
      <div class="status"><div class="label">Corpus source</div><div class="value">{{ corpus_manifest.source_kind|default("unknown") }}</div><div class="note">{{ corpus_manifest.dataset_name|default(dataset.source) }}</div></div>
      <div class="status {{ 'ok' if corpus_manifest.data_supervision.has_tactic_states|default(false) and corpus_manifest.data_supervision.has_true_positive_premises|default(false) else 'warn' }}"><div class="label">Data supervision</div><div class="value">{{ corpus_manifest.data_supervision.kind|default("unknown") }}</div><div class="note">premise labels: {{ corpus_manifest.data_supervision.has_true_positive_premises|default(false) }}</div></div>
      <div class="status"><div class="label">Config hash</div><div class="value">{{ corpus_manifest.config_hash|default("unknown") }}</div><div class="note">{{ corpus_manifest.config_path|default("") }}</div></div>
      <div class="status"><div class="label">Mathlib commit</div><div class="value">{{ corpus_manifest.corpus.mathlib_commit|default("unknown") }}</div><div class="note">Lean {{ corpus_manifest.corpus.lean_version|default("unknown") }}</div></div>
      <div class="status"><div class="label">Sampled theorems</div><div class="value">{{ corpus_manifest.sampled_theorems|default(overview.total_theorems) }}</div><div class="note">{{ corpus_manifest.sampled_rows|default(dataset.sample_rows) }} rows</div></div>
    </div>
    <div class="chips">{% for cmd in commands %}<span class="chip"><code>{{ cmd }}</code></span>{% endfor %}</div>
  </section>
</main>
<script>
const output = document.getElementById('api-output');
const statusEl = document.getElementById('api-status');
const graphData = {{ graph_visualization|tojson }};
const caseStudies = {{ theorem_case_studies|tojson }};
const graphColors = {Theorem:'#2563eb',ProofState:'#0e7490',Premise:'#0f8a5f',ProofTechnique:'#be123c',TacticStep:'#b7791f',FileModule:'#64748b'};
const edgeDescriptions = {
  has_proof_state:'theorem has proof state',
  appears_in_file:'theorem belongs to module',
  defined_in_file:'premise is defined in module',
  positive_uses:'proof state previously used premise',
  negative_candidate:'proof state contrasted with negative candidate',
  invokes_premise:'theorem uses premise',
  at_tactic_step:'proof state occurs at tactic step',
  uses_proof_technique:'proof state suggests proof technique',
  similar_to_theorem:'theorem is similar to theorem',
  co_occurs_with:'premises co-occur in proof state'
};
function escapeHtml(value){
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
}
function scoreBar(value, cls='green'){
  const pct = Math.max(0, Math.min(100, Number(value || 0) * 100)).toFixed(1);
  return `<div class="bar-track"><div class="bar ${cls}" style="width:${pct}%"></div></div>`;
}
function signalSummary(signals){
  const parts = [];
  if (!signals) return parts;
  if (Number(signals.same_namespace_score || 0) > 0) parts.push('shared namespace');
  if (Number(signals.same_domain_score || 0) > 0) parts.push('shared domain');
  if (Number(signals.graph_neighbor_score || 0) > 0) parts.push('graph-neighbor evidence');
  if (Number(signals.proof_technique_overlap_score || 0) > 0) parts.push('proof-technique overlap');
  if (Number(signals.premise_frequency_score || 0) > 0) parts.push('historical premise frequency');
  if (Number(signals.learned_ranker_score || 0) > 0) parts.push('learned ranker signal');
  if (Number(signals.parsed_symbol_overlap_score || 0) > 0) parts.push('shared parsed symbols');
  return parts;
}
function renderExplanationPaths(data){
  const premisePaths = (data.ranked_premises || []).slice(0, 4).map(row => {
    const reasons = (row.ranking_reasons || []).slice(0, 3);
    const signals = signalSummary(row.signals || {});
    const bullets = [...reasons, ...signals].slice(0, 5).map(reason => `<li>${escapeHtml(reason)}</li>`).join('');
    return `<div class="guidance-block"><h3><code>${escapeHtml(row.full_name || row.premise_id || 'premise')}</code></h3><div class="relation-path">New theorem -> retrieved proof context -> ${escapeHtml(row.full_name || 'recommended premise')}</div><ul>${bullets || '<li>Ranked by embedding similarity and available retrieval signals.</li>'}</ul></div>`;
  }).join('');
  const graphPaths = (data.graph_evidence || []).slice(0, 3).map(item => {
    const edges = (item.edges || []).slice(0, 3).map(edge => `<li><code>${escapeHtml(edge.edge_type)}</code>: ${escapeHtml(edge.source)} -> ${escapeHtml(edge.target)}</li>`).join('');
    return `<div class="guidance-block"><h3>Graph evidence for <code>${escapeHtml(item.entity_id || 'entity')}</code></h3><ul>${edges || '<li>No sampled graph edges for this entity.</li>'}</ul></div>`;
  }).join('');
  return premisePaths || graphPaths
    ? `<div class="guidance-grid">${premisePaths}${graphPaths}</div>`
    : '<div class="relation-path">New theorem -> similar proof state -> historical premise usage -> ranked recommendation</div>';
}
function renderGraph(){
  const svg = document.getElementById('kg-svg');
  const legend = document.getElementById('kg-legend');
  const edgeLegend = document.getElementById('kg-edge-legend');
  const detail = document.getElementById('kg-detail');
  if (!svg || !graphData || !graphData.nodes || graphData.nodes.length === 0) return;
  const nodes = new Map(graphData.nodes.map(node => [node.id, node]));
  const setDetail = html => { if (detail) detail.innerHTML = html; };
  const clearActive = () => svg.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
  const maxY = Math.max(340, ...graphData.nodes.map(node => Number(node.y || 0) + 38));
  svg.setAttribute('viewBox', `0 0 900 ${maxY}`);
  svg.innerHTML = '';
  for (const edge of graphData.edges || []) {
    const source = nodes.get(edge.source);
    const target = nodes.get(edge.target);
    if (!source || !target) continue;
    const line = document.createElementNS('http://www.w3.org/2000/svg','line');
    line.setAttribute('x1', source.x); line.setAttribute('y1', source.y);
    line.setAttribute('x2', target.x); line.setAttribute('y2', target.y);
    line.setAttribute('class', 'graph-edge');
    line.setAttribute('tabindex', '0');
    const title = document.createElementNS('http://www.w3.org/2000/svg','title');
    title.textContent = edge.edge_type;
    line.appendChild(title);
    const showEdge = () => {
      clearActive();
      line.classList.add('active');
      setDetail(`<h3>${escapeHtml(edgeDescriptions[edge.edge_type] || edge.edge_type)}</h3><p><code>${escapeHtml(source.label || source.id)}</code> -> <code>${escapeHtml(target.label || target.id)}</code></p><p class="muted">Edge type: <b>${escapeHtml(edge.edge_type)}</b></p>`);
    };
    line.addEventListener('click', showEdge);
    line.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') showEdge(); });
    svg.appendChild(line);
  }
  for (const node of graphData.nodes) {
    const circle = document.createElementNS('http://www.w3.org/2000/svg','circle');
    circle.setAttribute('cx', node.x); circle.setAttribute('cy', node.y); circle.setAttribute('r', node.node_type === 'Theorem' ? 10 : 8);
    circle.setAttribute('fill', graphColors[node.node_type] || '#475569');
    circle.setAttribute('class', 'graph-node');
    circle.setAttribute('tabindex', '0');
    const title = document.createElementNS('http://www.w3.org/2000/svg','title');
    title.textContent = `${node.node_type}: ${node.id}`;
    circle.appendChild(title);
    const showNode = () => {
      clearActive();
      circle.classList.add('active');
      const incident = (graphData.edges || []).filter(edge => edge.source === node.id || edge.target === node.id).slice(0, 8);
      const relations = incident.map(edge => {
        const otherId = edge.source === node.id ? edge.target : edge.source;
        const other = nodes.get(otherId) || {label: otherId, id: otherId};
        const arrow = edge.source === node.id ? '->' : '<-';
        return `<li><code>${escapeHtml(edge.edge_type)}</code> ${arrow} ${escapeHtml(other.label || other.id)}</li>`;
      }).join('');
      setDetail(`<h3>${escapeHtml(node.node_type)}: ${escapeHtml(node.label || node.id)}</h3><p class="muted"><code>${escapeHtml(node.id)}</code></p><ul>${relations || '<li>No sampled incident relations.</li>'}</ul>`);
    };
    circle.addEventListener('click', showNode);
    circle.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') showNode(); });
    svg.appendChild(circle);
    const label = document.createElementNS('http://www.w3.org/2000/svg','text');
    label.setAttribute('x', Number(node.x) + 12); label.setAttribute('y', Number(node.y) + 4);
    label.setAttribute('class', 'graph-label');
    label.textContent = node.label;
    svg.appendChild(label);
  }
  const types = [...new Set(graphData.nodes.map(node => node.node_type))];
  legend.innerHTML = types.map(type => `<span><i class="swatch" style="background:${graphColors[type] || '#475569'}"></i>${escapeHtml(type)}</span>`).join('');
  const edgeTypes = [...new Set((graphData.edges || []).map(edge => edge.edge_type))];
  if (edgeLegend) edgeLegend.innerHTML = edgeTypes.map(type => `<span><i class="edge-swatch"></i>${escapeHtml(edgeDescriptions[type] || type)}</span>`).join('');
}
function renderGuidance(data){
  const premises = (data.ranked_premises || []).slice(0, 6).map(row => {
    const reasons = (row.ranking_reasons || []).slice(0, 2).map(escapeHtml).join('; ');
    const statement = row.code || row.statement || row.file_path || '';
    return `<div><div class="rank"><code>${escapeHtml(row.full_name)}</code>${scoreBar(row.score,'green')}<b>${Number(row.score || 0).toFixed(3)}</b></div>${statement ? `<p class="muted">${escapeHtml(String(statement).slice(0, 180))}</p>` : ''}${reasons ? `<p class="muted">${reasons}</p>` : ''}</div>`;
  }).join('');
  const theorems = (data.similar_theorems || []).slice(0, 4).map(row => {
    const statement = row.statement || row.theorem_text || row.file_path || '';
    return `<div><div class="rank"><code>${escapeHtml(row.full_name)}</code>${scoreBar(row.score,'teal')}<b>${Number(row.score || 0).toFixed(3)}</b></div>${statement ? `<p class="muted">${escapeHtml(String(statement).slice(0, 160))}</p>` : ''}</div>`;
  }).join('');
  const techniques = (data.likely_proof_techniques || []).map(row => `<span class="chip">${escapeHtml(row.label)}</span>`).join('');
  const proofStates = (data.similar_proof_states || data.related_proof_states || []).slice(0, 3).map(row => {
    const goal = row.goal_text || row.context || row.proof_state || '';
    const source = row.full_name || row.theorem_id || row.proof_state_id || '';
    return `<div><code>${escapeHtml(source)}</code><div class="goal">${escapeHtml(String(goal).slice(0, 260))}</div></div>`;
  }).join('');
  const difficulty = data.difficulty_profile || {};
  const query = data.query || {};
  const lean = data.lean_diagnostics || {};
  const leanSummary = lean.summary || {};
  const source = query.retrieval_query_source || 'parsed_theorem_text';
  const extracted = query.lean_extracted_proof_state_count || 0;
  const diffSignals = difficulty.signals || {};
  const service = data.service || {};
  const serviceLine = service.request_id ? `<p class="muted">Request: <code>${escapeHtml(service.request_id)}</code>; duration: <b>${Number(service.duration_ms || 0).toFixed(1)} ms</b></p>` : '';
  const difficultyScore = typeof difficulty.difficulty_score === 'number' ? Number(difficulty.difficulty_score) : 0;
  const explanations = renderExplanationPaths(data);
  output.innerHTML = `<div class="guidance-grid"><div class="guidance-block"><h3>Relevant Lemmas / Premises</h3><div class="ranked">${premises || '<span class="muted">No premises returned.</span>'}</div></div><div class="guidance-block"><h3>Similar Theorems</h3><div class="ranked">${theorems || '<span class="muted">No similar theorems returned.</span>'}</div></div><div class="guidance-block"><h3>Likely Proof Techniques</h3><div class="chips">${techniques || '<span class="muted">No labels returned.</span>'}</div><h3 style="margin-top:14px">Difficulty Profile</h3><div class="rank"><code>${escapeHtml(difficulty.difficulty_bucket || 'unknown')}</code>${scoreBar(difficultyScore,'amber')}<b>${difficultyScore.toFixed(3)}</b></div><p class="muted">${escapeHtml(diffSignals.calibrated_by || 'query_heuristic')}</p></div><div class="guidance-block"><h3>Related Proof States / Patterns</h3><div class="ranked">${proofStates || '<span class="muted">No proof states returned.</span>'}</div></div></div><h3>Premise Ranking And Explanation</h3><p class="muted">Query source: <b>${escapeHtml(source)}</b>; Lean proof states: <b>${Number(extracted)}</b>${lean.checked ? `; unsolved goals: <b>${leanSummary.has_unsolved_goals ? 'yes' : 'no'}</b>` : ''}</p>${explanations}${serviceLine}`;
}
function renderCaseStudy(index){
  const item = caseStudies[index];
  if (!item) return;
  const guidance = item.guidance || {};
  const text = guidance.query?.theorem_text || item.theorem_text || item.full_name || '';
  document.getElementById('api-theorem').value = text;
  renderGuidance(guidance);
  statusEl.textContent = `Loaded sample: ${item.full_name || index + 1}`;
}
function tokenSet(text){
  return new Set(String(text || '').toLowerCase().split(/[^a-z0-9_'.]+/).filter(token => token.length > 2));
}
function localCaseScore(queryText, item){
  const queryTokens = tokenSet(queryText);
  const guidance = item.guidance || {};
  const candidateText = [
    item.full_name,
    item.theorem_text,
    guidance.query && guidance.query.theorem_text,
    ...(guidance.ranked_premises || []).slice(0, 5).map(row => row.full_name),
    ...(guidance.similar_theorems || []).slice(0, 5).map(row => row.full_name),
  ].join(' ');
  const candidateTokens = tokenSet(candidateText);
  let overlap = 0;
  queryTokens.forEach(token => { if (candidateTokens.has(token)) overlap += 1; });
  return overlap / Math.max(1, queryTokens.size);
}
function renderLocalFallback(queryText, reason){
  if (!caseStudies || caseStudies.length === 0) {
    output.innerHTML = `<span class="demo-error">${escapeHtml(reason || 'API unavailable and no local case studies exist.')}</span>`;
    statusEl.textContent = 'Error';
    return;
  }
  let bestIndex = 0;
  let bestScore = -1;
  caseStudies.forEach((item, index) => {
    const score = localCaseScore(queryText, item);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  });
  const item = caseStudies[bestIndex];
  renderGuidance(item.guidance || {});
  output.insertAdjacentHTML('afterbegin', `<div class="status warn"><div class="label">Local asset fallback</div><div class="value">sample match</div><div class="note">API unavailable; showing nearest precomputed case study <code>${escapeHtml(item.full_name || `Sample ${bestIndex + 1}`)}</code> with token-overlap score ${bestScore.toFixed(3)}.</div></div>`);
  statusEl.textContent = 'Local asset fallback';
}
function renderSampleButtons(){
  const container = document.getElementById('sample-theorems');
  if (!container) return;
  const samples = (caseStudies || []).slice(0, 4);
  container.innerHTML = samples.map((item, index) => `<button class="button secondary sample-button" type="button" data-sample-index="${index}">${escapeHtml((item.full_name || `Sample ${index + 1}`).split('.').slice(-2).join('.'))}</button>`).join('');
  container.querySelectorAll('[data-sample-index]').forEach(button => {
    button.addEventListener('click', () => renderCaseStudy(Number(button.dataset.sampleIndex)));
  });
}
document.getElementById('api-submit').addEventListener('click', async () => {
  statusEl.textContent = 'Retrieving';
  output.innerHTML = '<span class="muted">Retrieving guidance...</span>';
  const slowTimer = window.setTimeout(() => { statusEl.textContent = 'Still running'; }, 2500);
  const payload = {
    theorem_text: document.getElementById('api-theorem').value,
    input_type: document.getElementById('api-input-type').value,
    k_premises: Number(document.getElementById('api-k-premises').value || 10),
    k_theorems: Number(document.getElementById('api-k-theorems').value || 5),
    index_split: 'train',
    validate_lean: document.getElementById('api-validate-lean').checked
  };
  try {
    const response = await fetch(document.getElementById('api-url').value, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    if (!response.ok) throw new Error(await response.text());
    renderGuidance(await response.json());
    statusEl.textContent = 'Done';
  } catch (error) {
    renderLocalFallback(payload.theorem_text, error.message);
  } finally {
    window.clearTimeout(slowTimer);
  }
});
document.getElementById('api-metrics').addEventListener('click', async () => {
  statusEl.textContent = 'Loading metrics';
  const url = new URL(document.getElementById('api-url').value);
  url.pathname = '/metrics';
  try {
    const response = await fetch(url.toString());
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    output.innerHTML = `<h3>API Metrics</h3><table><tr><th>Total</th><th>Success</th><th>Failed</th><th>Avg ms</th></tr><tr><td>${Number(data.total_requests || 0)}</td><td>${Number(data.successful_requests || 0)}</td><td>${Number(data.failed_requests || 0)}</td><td>${Number(data.average_duration_ms || 0).toFixed(1)}</td></tr></table><pre>${escapeHtml(JSON.stringify(data.last_request || {}, null, 2))}</pre>`;
    statusEl.textContent = 'Metrics loaded';
  } catch (error) {
    output.innerHTML = `<span class="demo-error">${escapeHtml(error.message)}</span>`;
    statusEl.textContent = 'Error';
  }
});
document.getElementById('api-clear').addEventListener('click', () => {
  output.innerHTML = '<span class="muted">Choose a sample theorem or run live retrieval.</span>';
  statusEl.textContent = 'Idle';
});
renderGraph();
renderSampleButtons();
renderCaseStudy(0);
</script>
</body>
</html>"""


def _strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def run(config_path: str) -> None:
    Path("homepage/assets").mkdir(parents=True, exist_ok=True)
    summary = build_summary(config_path)
    for name, data in [
        ("metrics", summary["metrics"]),
        ("retrieval_examples", summary["retrieval_examples"]),
        ("theorem_retrieval_case_studies", summary["theorem_case_studies"]),
        ("graph_visualization", summary["graph_visualization"]),
        ("corpus_manifest", summary["corpus_manifest"]),
        ("refresh_dashboard", summary["refresh_dashboard"]),
        ("refresh_trend", summary["refresh_trend"]),
        ("refresh_history", summary["refresh_history"]),
        ("domain_coverage", summary["domain_coverage"]),
        ("graph_stats", summary["graph_stats"]),
        ("homepage_summary", summary),
    ]:
        write_json(f"homepage/assets/{name}.json", data)
    html = _strip_trailing_whitespace(Template(HTML).render(**summary))
    Path("homepage/index.html").write_text(html, encoding="utf-8")
