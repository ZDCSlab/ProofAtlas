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
    .timeline{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}
    .step{border-left:4px solid var(--teal);background:#f8fbfd;border-radius:8px;padding:12px;border-top:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
    .section-lead{color:var(--muted);max-width:850px;margin:0 0 16px}
    @media (max-width:900px){
      header h1{font-size:38px}
      .hero-grid,.two-col,.timeline{grid-template-columns:1fr}
      .kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
    }
    @media (max-width:560px){
      main{padding:12px}
      section{padding:16px}
      .kpi-grid{grid-template-columns:1fr}
      .bar-row,.mini-bar,.rank{grid-template-columns:1fr}
      .bar-track{height:14px}
    }
  </style>
</head>
<body>
<header>
  <div class="eyebrow">Real LeanRank data + GPU semantic retrieval</div>
  <h1>{{ dataset.project_name }}</h1>
  <p>A proof knowledge atlas that turns Lean/mathlib proof states into a searchable graph of theorems, premises, techniques, difficulty signals, and retrieval evidence.</p>
  <div class="hero-grid">
    <div class="hero-panel">
      <h2>What the reviewer should see first</h2>
      <div class="chips">
        <span class="pill ok">Real dataset: {{ dataset.sample_rows }} rows</span>
        <span class="pill info">{{ embedding.model_name }}</span>
        <span class="pill info">GPU: {{ embedding.device }}</span>
        <span class="pill ok">Schema errors: {{ overview.schema_errors }}</span>
        <span class="pill ok">Missing graph endpoints: {{ overview.missing_graph_endpoints }}</span>
      </div>
    </div>
    <div class="hero-panel">
      <h2>Pipeline</h2>
      <div class="chips">
        <span class="chip">LeanRank rows</span>
        <span class="chip">normalized entities</span>
        <span class="chip">knowledge graph</span>
        <span class="chip">BGE embeddings</span>
        <span class="chip">retrieval demo</span>
      </div>
    </div>
  </div>
</header>
<main>
  <section>
    <h2>Executive Snapshot</h2>
    <div class="kpi-grid">
      <div class="kpi"><div class="label">Rows loaded</div><div class="value">{{ dataset.sample_rows }}</div><div class="note">Original plan: {{ dataset.configured_rows }}</div></div>
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
      <div class="kpi"><div class="label">AUC</div><div class="value">{{ "%.3f"|format(metrics.get("AUC", 0)) }}</div><div class="note">Ranker validation</div></div>
      <div class="kpi"><div class="label">Recall@10</div><div class="value">{{ "%.3f"|format(metrics.get("Recall@10", 0)) }}</div><div class="note">Gold premise in top 10</div></div>
      <div class="kpi"><div class="label">MRR</div><div class="value">{{ "%.3f"|format(metrics.get("MRR", 0)) }}</div><div class="note">Mean reciprocal rank</div></div>
      <div class="kpi"><div class="label">Technique coverage</div><div class="value">{{ "%.3f"|format(metrics.get("proof_technique_label_coverage", 0)) }}</div><div class="note">Weak labels over proof states</div></div>
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
    <h2>System Capabilities</h2>
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
    <div class="chips">{% for cmd in commands %}<span class="chip"><code>{{ cmd }}</code></span>{% endfor %}</div>
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
