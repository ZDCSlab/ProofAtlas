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
    :root{color-scheme:light;--ink:#202124;--muted:#5f6368;--line:#d9dde3;--bg:#f7f8fa;--panel:#fff;--accent:#1a73e8}
    body{font-family:Inter,Arial,sans-serif;margin:0;color:var(--ink);background:var(--bg);line-height:1.45}
    header{background:#233142;color:white;padding:44px 8vw 38px}
    header p{max-width:820px;color:#d7dde6}
    main{max-width:1160px;margin:auto;padding:24px}
    section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:20px;margin:16px 0}
    h1,h2,h3{margin:0 0 10px}
    h2{font-size:20px}
    h3{font-size:16px;color:#2b3645}
    table{border-collapse:collapse;width:100%;font-size:14px}
    td,th{border-bottom:1px solid #e5e7eb;padding:8px;text-align:left;vertical-align:top}
    th{color:#3c4043;background:#fafafa}
    code{background:#eef1f4;padding:2px 4px;border-radius:4px}
    pre{white-space:pre-wrap;background:#f5f7f9;border:1px solid #e2e6ea;border-radius:6px;padding:10px;overflow:auto}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
    .stat{font-size:28px;font-weight:700;overflow-wrap:anywhere;line-height:1.2}
    .muted{color:var(--muted)}
    .chips{display:flex;flex-wrap:wrap;gap:8px}
    .chip{border:1px solid var(--line);border-radius:999px;padding:4px 9px;background:#fafafa}
  </style>
</head>
<body>
<header>
  <h1>{{ dataset.project_name }}</h1>
  <p>A retrieval-ready atlas of LeanRank proof states, theorems, premises, proof techniques, and difficulty signals built from real Lean/mathlib data with GPU-backed semantic embeddings.</p>
</header>
<main>
  <section>
    <h2>Dataset</h2>
    <div class="grid">
      <div><div class="muted">Source</div><div class="stat">{{ dataset.source }}</div></div>
      <div><div class="muted">Real rows loaded</div><div class="stat">{{ dataset.sample_rows }}</div></div>
      <div><div class="muted">Original plan</div><div class="stat">{{ dataset.configured_rows }}</div></div>
      <div><div class="muted">Hugging Face data</div><div class="stat">{{ dataset.use_huggingface }}</div></div>
      <div><div class="muted">Demo processed files</div><div class="stat">{{ dataset.processed_files|length }}</div></div>
      <div><div class="muted">Premise object shape</div><code>{{ dataset.premise_shape }}</code></div>
    </div>
    <table>
      <tr><th>Split</th><th>Theorems</th><th>Proof states</th><th>Premises</th><th>Techniques</th><th>Nodes</th><th>Edges</th></tr>
      {% for row in dataset.split_counts %}
      <tr><td>{{ row.split }}</td><td>{{ row.theorems }}</td><td>{{ row.proof_states }}</td><td>{{ row.premises }}</td><td>{{ row.proof_techniques }}</td><td>{{ row.nodes }}</td><td>{{ row.edges }}</td></tr>
      {% endfor %}
    </table>
  </section>

  <section>
    <h2>Embedding Model</h2>
    <div class="grid">
      <div><div class="muted">Backend</div><div class="stat">{{ embedding.backend }}</div></div>
      <div><div class="muted">Model</div><div class="stat">{{ embedding.model_name }}</div></div>
      <div><div class="muted">Device</div><div class="stat">{{ embedding.device }}</div></div>
      <div><div class="muted">Batch size</div><div class="stat">{{ embedding.batch_size }}</div></div>
    </div>
    <p class="muted">Proof states are encoded as queries; Lean theorems and premises are encoded as retrieval passages.</p>
  </section>

  <section>
    <h2>KG Overview</h2>
    {% for split, stats in graph_stats.items() %}
      <h3>{{ split }}</h3>
      <div class="grid">
        <div><div class="muted">Nodes</div><div class="stat">{{ stats.node_count }}</div></div>
        <div><div class="muted">Edges</div><div class="stat">{{ stats.edge_count }}</div></div>
      </div>
      <table>
        <tr><th>Node type</th><th>Count</th><th>Edge type</th><th>Count</th></tr>
        {% set node_items = stats.node_counts_by_type.items()|list %}
        {% set edge_items = stats.edge_counts_by_type.items()|list %}
        {% for i in range([node_items|length, edge_items|length]|max) %}
        <tr>
          <td>{{ node_items[i][0] if i < node_items|length else "" }}</td>
          <td>{{ node_items[i][1] if i < node_items|length else "" }}</td>
          <td>{{ edge_items[i][0] if i < edge_items|length else "" }}</td>
          <td>{{ edge_items[i][1] if i < edge_items|length else "" }}</td>
        </tr>
        {% endfor %}
      </table>
    {% endfor %}
  </section>

  <section>
    <h2>Domain Coverage</h2>
    <table>
      <tr><th>Split</th><th>Domain counts</th></tr>
      {% for split, counts in domain_coverage.items() %}
      <tr><td>{{ split }}</td><td>{{ counts }}</td></tr>
      {% endfor %}
    </table>
  </section>

  <section>
    <h2>Function Showcase</h2>
    <div class="chips">{% for fn in functions %}<span class="chip"><code>{{ fn }}</code></span>{% endfor %}</div>
  </section>

  <section>
    <h2>Retrieval Examples</h2>
    {% for ex in retrieval_examples[:5] %}
      <h3>{{ ex.proof_state_id }}</h3>
      <p>{{ ex.proof_state }}</p>
      <p>Gold premise: <code>{{ ex.gold_positive_premise }}</code> | in train index: <b>{{ ex.gold_in_train_index }}</b></p>
      <table>
        <tr><th>Premise</th><th>Score</th><th>Index</th></tr>
        {% for r in ex.top_retrieved_premises %}
        <tr><td>{{ r.full_name }}</td><td>{{ "%.3f"|format(r.score) }}</td><td>{{ r.index_split }}</td></tr>
        {% endfor %}
      </table>
    {% endfor %}
  </section>

  <section>
    <h2>Proof-Technique Labels</h2>
    <table>
      <tr><th>Split</th><th>Label</th><th>Count</th></tr>
      {% for row in proof_techniques %}
      <tr><td>{{ row.split }}</td><td>{{ row.label }}</td><td>{{ row.count }}</td></tr>
      {% endfor %}
    </table>
  </section>

  <section>
    <h2>Difficulty</h2>
    <table>
      <tr><th>Split</th><th>Bucket</th><th>Count</th></tr>
      {% for row in difficulty_distribution %}
      <tr><td>{{ row.split }}</td><td>{{ row.bucket }}</td><td>{{ row.count }}</td></tr>
      {% endfor %}
    </table>
  </section>

  <section>
    <h2>Evaluation</h2>
    <div class="grid">
      <div><div class="muted">Recall@1</div><div class="stat">{{ "%.3f"|format(metrics.get("Recall@1", 0)) }}</div></div>
      <div><div class="muted">Recall@5</div><div class="stat">{{ "%.3f"|format(metrics.get("Recall@5", 0)) }}</div></div>
      <div><div class="muted">Recall@10</div><div class="stat">{{ "%.3f"|format(metrics.get("Recall@10", 0)) }}</div></div>
      <div><div class="muted">MRR</div><div class="stat">{{ "%.3f"|format(metrics.get("MRR", 0)) }}</div></div>
      <div><div class="muted">Gold premise coverage</div><div class="stat">{{ "%.3f"|format(metrics.get("gold_premise_coverage", 0)) }}</div></div>
      <div><div class="muted">Technique coverage</div><div class="stat">{{ "%.3f"|format(metrics.get("proof_technique_label_coverage", 0)) }}</div></div>
    </div>
  </section>

  <section>
    <h2>Validation</h2>
    <div class="grid">
      <div><div class="muted">Schema errors</div><div class="stat">{{ validation.schema.error_count }}</div></div>
      <div><div class="muted">Split leakage</div><div class="stat">{{ validation.split_leakage.has_leakage }}</div></div>
    </div>
    <h3>Graph validation</h3>
    <table>
      <tr><th>Split</th><th>NetworkX loadable</th><th>Missing endpoints</th></tr>
      {% for split, row in validation.graph.items() %}
      <tr><td>{{ split }}</td><td>{{ row.networkx_loadable }}</td><td>{{ row.missing_endpoint_count }}</td></tr>
      {% endfor %}
    </table>
  </section>

  <section>
    <h2>Reproducibility</h2>
    {% for cmd in commands %}<p><code>{{ cmd }}</code></p>{% endfor %}
  </section>
</main>
</body>
</html>"""


def run(config_path: str) -> None:
    Path("homepage/assets").mkdir(parents=True, exist_ok=True)
    summary = build_summary(config_path)
    for name, data in [
        ("metrics", summary["metrics"]),
        ("retrieval_examples", summary["retrieval_examples"]),
        ("domain_coverage", summary["domain_coverage"]),
        ("graph_stats", summary["graph_stats"]),
        ("homepage_summary", summary),
    ]:
        write_json(f"homepage/assets/{name}.json", data)
    html = Template(HTML).render(**summary)
    Path("homepage/index.html").write_text(html, encoding="utf-8")
