from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Template

from .report import build_summary
from .utils import read_json, write_json


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ProofAtlas Research Report</title>
  <style>
    :root{
      color-scheme:light;
      --ink:#17202a;--muted:#657180;--line:#d9e0e8;--bg:#f6f8fb;--panel:#ffffff;
      --blue:#1d4ed8;--green:#0f766e;--amber:#b7791f;--rose:#be123c;--violet:#6d28d9;
      --soft-blue:#e8f0ff;--soft-green:#e6f6f2;--soft-amber:#fff4db;--soft-rose:#fde8ee;
    }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Arial,sans-serif;line-height:1.45}
    header{background:#142033;color:white;padding:42px 7vw 34px;border-bottom:5px solid #14b8a6}
    header h1{font-size:46px;line-height:1.04;margin:0 0 10px;letter-spacing:0}
    header p{max-width:940px;margin:0;color:#dce7f5;font-size:18px}
    main{max-width:1220px;margin:0 auto;padding:22px}
    section{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:18px 0;padding:22px}
    h2,h3{margin:0 0 12px;letter-spacing:0}
    h2{font-size:22px}
    h3{font-size:16px;color:#263547}
    table{width:100%;border-collapse:collapse;font-size:14px}
    th,td{padding:8px;border-bottom:1px solid #e7edf4;text-align:left;vertical-align:top}
    th{background:#f8fafc;color:#334155}
    code{background:#edf2f7;border-radius:4px;padding:2px 5px;overflow-wrap:anywhere}
    .eyebrow{text-transform:uppercase;letter-spacing:.08em;font-size:12px;color:#9fc5ee;font-weight:800;margin-bottom:8px}
    .lead{color:var(--muted);max-width:900px;margin:0 0 16px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}
    .two-col{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}
    .panel{border:1px solid var(--line);border-radius:8px;background:#fbfcfe;padding:14px}
    .kpi{border:1px solid var(--line);border-radius:8px;background:#fbfcfe;padding:14px;min-height:104px}
    .label{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:800}
    .value{font-size:29px;line-height:1.12;font-weight:850;margin-top:7px;overflow-wrap:anywhere}
    .note{font-size:13px;color:var(--muted);margin-top:6px}
    .task{border-left:5px solid var(--blue)}
    .task:nth-child(2){border-left-color:var(--green)}
    .task:nth-child(3){border-left-color:var(--amber)}
    .task:nth-child(4){border-left-color:var(--violet)}
    .bar-row{display:grid;grid-template-columns:140px minmax(0,1fr) 70px;gap:10px;align-items:center;font-size:14px;margin:9px 0}
    .bar-track{height:12px;background:#edf2f7;border-radius:999px;overflow:hidden}
    .bar{height:100%;border-radius:999px;background:var(--blue)}
    .bar.green{background:var(--green)}
    .bar.amber{background:var(--amber)}
    .bar.rose{background:var(--rose)}
    .bar.violet{background:var(--violet)}
    .goal{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f3f6fb;border:1px solid #dce5ef;border-radius:6px;padding:10px;overflow-wrap:anywhere;white-space:pre-wrap}
    .case{display:grid;gap:14px}
    .rank{display:grid;grid-template-columns:minmax(210px,1fr) minmax(0,1fr) 58px;gap:10px;align-items:center;font-size:13px;margin:8px 0}
    .chips{display:flex;flex-wrap:wrap;gap:8px}
    .chip{display:inline-flex;border:1px solid var(--line);border-radius:999px;background:#fafcff;padding:5px 10px;font-size:13px}
    .callout{border:1px solid #badfcb;background:var(--soft-green);border-radius:8px;padding:13px}
    .metric-table td:nth-child(n+3),.metric-table th:nth-child(n+3){text-align:right}
    .small{font-size:13px;color:var(--muted)}
    @media (max-width:820px){
      header h1{font-size:34px}
      main{padding:14px}
      section{padding:16px}
      .two-col{grid-template-columns:1fr}
      .bar-row,.rank{grid-template-columns:1fr}
    }
  </style>
</head>
<body>
<header>
  <div class="eyebrow">Static Visual Research Report</div>
  <h1>ProofAtlas Research Report</h1>
  <p>Retrieval-centered LeanRank proof guidance: processed theorem/proof-state/premise data, theorem-disjoint evaluation, strategy-facet recovery, difficulty-profile recovery, and math-facing case studies.</p>
</header>
<main>
  <section>
    <h2>Retrieval-Centered LeanRank Proof Guidance</h2>
    <p class="lead">This page is a visual companion to <code>outputs/reports/research_report.md</code>. It is not an API demo; it summarizes the dataset, split statistics, retrieval tasks, metrics, and case studies produced by the local ProofAtlas pipeline.</p>
    <div class="grid">
      <div class="kpi"><div class="label">Sampled theorems</div><div class="value">{{ corpus.sampled_theorems }}</div><div class="note">{{ corpus.sampled_rows }} LeanRank rows from {{ corpus.dataset_name }}</div></div>
      <div class="kpi"><div class="label">Test theorems</div><div class="value">{{ test_counts.theorems|default(0) }}</div><div class="note">theorem-disjoint held-out split</div></div>
      <div class="kpi"><div class="label">Theorem Recall@10</div><div class="value">{{ fmt(premise.theorem["Recall@10"]) }}</div><div class="note">headline premise retrieval result</div></div>
      <div class="kpi"><div class="label">Strategy Hit@3</div><div class="value">{{ fmt(strategy_eval["any_label_hit@3"]) }}</div><div class="note">proof-state neighbor facet recovery</div></div>
    </div>
  </section>

  <section>
    <h2>Four Research Tasks</h2>
    <div class="grid">
      <div class="panel task"><h3>1. Premise Prediction</h3><p>Theorem-level premise retrieval and reranking for held-out theorems.</p><p class="small">Evidence: Recall@10 {{ fmt(premise.theorem["Recall@10"]) }}, Recall@100 {{ fmt(premise.theorem["Recall@100"]) }}, MRR {{ fmt(premise.theorem["MRR"]) }}.</p></div>
      <div class="panel task"><h3>2. Proof Pattern Retrieval</h3><p>Similar theorem and similar proof-state retrieval used as proof-pattern evidence.</p><p class="small">Neighbors are not evaluated directly; utility is measured through strategy-facet and difficulty-profile recovery.</p></div>
      <div class="panel task"><h3>3. Strategy Retrieval</h3><p>Retrieve strategy facets from similar train proof states.</p><p class="small">Evidence: Label Recall@5 {{ fmt(strategy_eval["label_recall@5"]) }}, Any-label Hit@3 {{ fmt(strategy_eval["any_label_hit@3"]) }}.</p></div>
      <div class="panel task"><h3>4. Difficulty Retrieval</h3><p>Retrieve historical difficulty profiles and calibrate them into score/bucket estimates.</p><p class="small">Evidence: MAE {{ fmt(difficulty_eval["retrieved_profile_mae"]) }}, bucket accuracy {{ fmt(difficulty_eval["bucket_accuracy"]) }}.</p></div>
    </div>
  </section>

  <section>
    <h2>Dataset And Split Statistics</h2>
    <div class="two-col">
      <div>
        <h3>Processed Splits</h3>
        <table>
          <tr><th>Split</th><th>Theorems</th><th>Proof states</th><th>Premises</th><th>Positive edges</th></tr>
          {% for split, row in split_counts.items() %}
          <tr><td>{{ split }}</td><td>{{ row.theorems }}</td><td>{{ row.proof_states }}</td><td>{{ row.premises }}</td><td>{{ row.positive_edges }}</td></tr>
          {% endfor %}
        </table>
      </div>
      <div>
        <h3>Top Test Domains</h3>
        {% for row in test_domains[:10] %}
        <div class="bar-row"><span>{{ row.domain }}</span><div class="bar-track"><div class="bar" style="width:{{ pct(row.share) }}%"></div></div><b>{{ fmt(row.share) }}</b></div>
        {% endfor %}
      </div>
    </div>
  </section>

  <section>
    <h2>Evaluation Results</h2>
    <div class="two-col">
      <div>
        <h3>Theorem-Level Premise Retrieval</h3>
        <table class="metric-table">
          <tr><th>Queries</th><th>Recall@1</th><th>Recall@5</th><th>Recall@10</th><th>Recall@100</th><th>MRR</th><th>MAP</th></tr>
          <tr><td>{{ premise.theorem.evaluated_theorems }}</td><td>{{ fmt(premise.theorem["Recall@1"]) }}</td><td>{{ fmt(premise.theorem["Recall@5"]) }}</td><td>{{ fmt(premise.theorem["Recall@10"]) }}</td><td>{{ fmt(premise.theorem["Recall@100"]) }}</td><td>{{ fmt(premise.theorem.MRR) }}</td><td>{{ fmt(premise.theorem.MAP) }}</td></tr>
        </table>
      </div>
      <div>
        <h3>Retrieval-Grounded Auxiliary Tasks</h3>
        <table class="metric-table">
          <tr><th>Task</th><th>Queries</th><th>Metric</th><th>Value</th></tr>
          <tr><td>Strategy retrieval</td><td>{{ strategy_eval.evaluated_queries }}</td><td>Label Recall@5</td><td>{{ fmt(strategy_eval["label_recall@5"]) }}</td></tr>
          <tr><td>Strategy retrieval</td><td>{{ strategy_eval.evaluated_queries }}</td><td>Any-label Hit@3</td><td>{{ fmt(strategy_eval["any_label_hit@3"]) }}</td></tr>
          <tr><td>Difficulty retrieval</td><td>{{ difficulty_eval.evaluated_queries }}</td><td>Retrieved-profile MAE</td><td>{{ fmt(difficulty_eval.retrieved_profile_mae) }}</td></tr>
          <tr><td>Difficulty retrieval</td><td>{{ difficulty_eval.evaluated_queries }}</td><td>Bucket accuracy</td><td>{{ fmt(difficulty_eval.bucket_accuracy) }}</td></tr>
        </table>
      </div>
    </div>
    <div class="grid" style="margin-top:14px">
      <div class="panel">
        <h3>Strategy Facet Distribution</h3>
        {% for row in strategy_distribution[:8] %}
        <div class="bar-row"><span>{{ row.label }}</span><div class="bar-track"><div class="bar green" style="width:{{ pct(row.share) }}%"></div></div><b>{{ row.count }}</b></div>
        {% endfor %}
      </div>
      <div class="panel">
        <h3>Difficulty Bucket Distribution</h3>
        {% for row in difficulty_distribution %}
        <div class="bar-row"><span>{{ row.split }} / {{ row.bucket }}</span><div class="bar-track"><div class="bar amber" style="width:{{ pct(row.share) }}%"></div></div><b>{{ row.count }}</b></div>
        {% endfor %}
      </div>
    </div>
  </section>

  <section>
    <h2>Case Studies</h2>
    <p class="lead">These held-out examples show how the retrieval bundle is meant to be read: inspect candidate premises, compare nearby historical theorem/proof-state neighbors, read strategy facets, and use difficulty as calibration.</p>
    {% for case in cases[:2] %}
    <div class="case panel">
      <div>
        <h3>{{ loop.index }}. <code>{{ case.theorem }}</code></h3>
        <p>{{ case.meaning }}</p>
        <p class="small">Domain: <b>{{ case.domain }} / {{ case.subdomain }}</b> · split: <b>{{ case.split }}</b> · gold premise train coverage: <b>{{ fmt(case.gold_premise_train_coverage) }}</b> · difficulty: <b>{{ case.difficulty.difficulty_bucket }} / {{ fmt(case.difficulty.difficulty_score) }}</b></p>
      </div>
      <div class="goal">{{ case.goal_text }}</div>
      <div class="two-col">
        <div>
          <h3>Retrieved Premises</h3>
          {% for row in case.top_premises[:5] %}
          <div class="rank"><code>{{ row.full_name }}</code><div class="bar-track"><div class="bar green" style="width:{{ pct(row.score) }}%"></div></div><b>{{ fmt(row.score) }}</b></div>
          <p class="small">{{ row.reason }}</p>
          {% endfor %}
        </div>
        <div>
          <h3>Similar Theorems</h3>
          {% for row in case.similar_theorems[:4] %}
          <div class="rank"><code>{{ row.full_name }}</code><div class="bar-track"><div class="bar" style="width:{{ pct(row.score) }}%"></div></div><b>{{ fmt(row.score) }}</b></div>
          {% endfor %}
          <h3 style="margin-top:16px">Similar Proof States</h3>
          {% for row in case.similar_proof_states[:3] %}
          <p><code>{{ row.full_name }}</code> <span class="small">({{ fmt(row.score) }})</span></p>
          <div class="goal">{{ row.goal_text }}</div>
          {% endfor %}
        </div>
      </div>
      <div>
        <h3>Strategy Facets</h3>
        <div class="chips">
          {% for row in case.techniques[:5] %}
          <span class="chip">{{ row.label }} · {{ fmt(row.confidence) }}</span>
          {% endfor %}
        </div>
      </div>
      <div class="callout">{{ case.takeaway }}</div>
    </div>
    {% endfor %}
  </section>

  <section>
    <h2>Reproducibility Notes</h2>
    <div class="grid">
      <div class="panel"><h3>Processed data</h3><p><code>data/processed/{train,val,test,demo}</code></p></div>
      <div class="panel"><h3>Embeddings and indexes</h3><p><code>outputs/embeddings</code>, <code>outputs/indexes</code></p></div>
      <div class="panel"><h3>Prediction results</h3><p><code>outputs/predictions/research_prediction_results.json</code></p></div>
      <div class="panel"><h3>Markdown report</h3><p><code>outputs/reports/research_report.md</code></p></div>
    </div>
  </section>
</main>
</body>
</html>"""


CASE_MEANINGS = {
    "Action.full_res": (
        "This is a category/action morphism equality. The mathematical content is about compatibility of an action map with a morphism, so useful neighbors should emphasize categorical composition, hom/extensionality structure, and commutative-diagram style goals.",
        "The retrieved premises point to category and monoidal-category structure, while the nearest theorem/proof-state neighbors are morphism-composition identities. The strategy facets summarize the intended proof mode as category_morphism_reasoning plus rewriting.",
    ),
    "Affine.Simplex.affineCombination_mem_interior_iff": (
        "This is an affine-geometry statement: an affine combination of simplex vertices lies in the interior exactly when every barycentric coordinate is strictly between 0 and 1.",
        "The retrieved premises and neighbors concentrate around affine combinations, affine spans, simplex centers, and membership conditions. The strategy facets indicate rewriting across an iff statement, set-membership reasoning, and algebraic manipulation of coordinates.",
    ),
}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    try:
        return f"{max(0.0, min(100.0, float(value) * 100.0)):.1f}"
    except (TypeError, ValueError):
        return "0.0"


def _strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def _research_context(summary: dict[str, Any]) -> dict[str, Any]:
    research = read_json("outputs/predictions/research_prediction_results.json", {})
    metrics = summary.get("metrics", {})
    premise = research.get("premise_prediction") or {
        "theorem": {
            "evaluated_theorems": metrics.get("theorem_retrieval_evaluated_theorems", 0),
            "Recall@1": metrics.get("theorem_retrieval_Recall@1", 0.0),
            "Recall@5": metrics.get("theorem_retrieval_Recall@5", 0.0),
            "Recall@10": metrics.get("theorem_retrieval_Recall@10", 0.0),
            "Recall@100": metrics.get("theorem_retrieval_Recall@100", 0.0),
            "MRR": metrics.get("theorem_retrieval_MRR", 0.0),
            "MAP": metrics.get("theorem_retrieval_MAP", 0.0),
        }
    }
    strategy = research.get("proof_strategy_hinting", {})
    difficulty = research.get("difficulty_prediction", {})
    processed = research.get("processed_data", {})
    split_counts = processed.get("splits") or {
        str(row.get("split")): {
            "theorems": row.get("theorems", 0),
            "proof_states": row.get("proof_states", 0),
            "premises": row.get("premises", 0),
            "positive_edges": row.get("positive_edges", 0),
        }
        for row in (summary.get("dataset", {}).get("split_counts") or [])
    }
    test_counts = split_counts.get("test") or next(iter(split_counts.values()), {})
    test_domains = (processed.get("domain_counts") or {}).get("test") or [
        {"domain": row.get("domain"), "share": row.get("share", 0.0), "theorems": row.get("count", 0)}
        for row in summary.get("top_domains", [])
    ]

    strategy_rows = [row for row in strategy.get("label_distribution", []) if row.get("split") == "test"] or [
        row for row in summary.get("proof_techniques", []) if row.get("split") in {"test", "train"}
    ]
    max_strategy_count = max([int(row.get("count", 0) or 0) for row in strategy_rows] or [1])
    strategy_distribution = [
        {**row, "share": (float(row.get("count", 0) or 0) / max_strategy_count)}
        for row in sorted(strategy_rows, key=lambda row: int(row.get("count", 0) or 0), reverse=True)
    ]

    difficulty_rows = difficulty.get("distribution") or summary.get("difficulty_distribution", [])
    max_difficulty_count = max([int(row.get("count", 0) or 0) for row in difficulty_rows] or [1])
    difficulty_distribution = [
        {**row, "share": (float(row.get("count", 0) or 0) / max_difficulty_count)}
        for row in difficulty_rows
        if row.get("split") in {"train", "test"}
    ]

    cases = []
    for case in research.get("sample_prediction_cases", []):
        theorem = str(case.get("theorem", ""))
        if theorem not in CASE_MEANINGS:
            continue
        meaning, takeaway = CASE_MEANINGS[theorem]
        cases.append({**case, "meaning": meaning, "takeaway": takeaway})

    if not cases:
        for case in summary.get("theorem_case_studies", [])[:2]:
            guidance = case.get("guidance", {})
            difficulty_profile = guidance.get("difficulty_profile", {})
            cases.append(
                {
                    "theorem": case.get("full_name", "sample theorem"),
                    "meaning": "This sample theorem illustrates how the generated visual report presents retrieved premises, theorem neighbors, strategy facets, and a difficulty profile.",
                    "takeaway": "The case study is generated from local retrieval artifacts and is intended for report inspection rather than live API interaction.",
                    "domain": (guidance.get("query", {}) or {}).get("domain_hint") or "sample",
                    "subdomain": "sample",
                    "split": case.get("split", "test"),
                    "gold_premise_train_coverage": case.get("gold_premise_train_coverage", 0.0),
                    "goal_text": (guidance.get("query", {}) or {}).get("goal_text") or (guidance.get("query", {}) or {}).get("theorem_text", ""),
                    "top_premises": [
                        {"full_name": row.get("full_name", ""), "score": row.get("score", 0.0), "reason": "; ".join(row.get("ranking_reasons", [])[:2])}
                        for row in guidance.get("ranked_premises", [])[:5]
                    ],
                    "similar_theorems": guidance.get("similar_theorems", [])[:4],
                    "similar_proof_states": guidance.get("similar_proof_states", [])[:3],
                    "techniques": guidance.get("likely_proof_techniques", [])[:5],
                    "difficulty": {
                        "difficulty_bucket": difficulty_profile.get("difficulty_bucket", "unknown"),
                        "difficulty_score": difficulty_profile.get("difficulty_score", 0.0),
                    },
                }
            )

    return {
        **summary,
        "corpus": research.get("corpus", summary.get("corpus_manifest", {})),
        "premise": premise,
        "strategy_eval": strategy.get("retrieval_evaluation", {"evaluated_queries": 0, "label_recall@5": 0.0, "any_label_hit@3": 0.0}),
        "difficulty_eval": difficulty.get("retrieval_evaluation", {"evaluated_queries": 0, "retrieved_profile_mae": 0.0, "bucket_accuracy": 0.0}),
        "split_counts": split_counts,
        "test_counts": test_counts,
        "test_domains": test_domains,
        "strategy_distribution": strategy_distribution,
        "difficulty_distribution": difficulty_distribution,
        "cases": cases[:2],
        "fmt": _fmt,
        "pct": _pct,
    }


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
    html = _strip_trailing_whitespace(Template(HTML).render(**_research_context(summary)))
    Path("homepage/index.html").write_text(html, encoding="utf-8")
