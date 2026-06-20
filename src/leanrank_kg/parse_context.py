from __future__ import annotations

import re


SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_'.]*|[∀∃→↔≤≥=+*/<>]")


def parse_context(context: str | None) -> dict:
    text = (context or "").strip()
    if not text:
        return {
            "goal_text": "",
            "local_hypotheses": [],
            "symbols": [],
            "namespace_hints": [],
            "typeclass_hints": [],
        }
    if "⊢" in text:
        before, goal = text.split("⊢", 1)
    else:
        before, goal = "", text
    hypotheses = [line.strip() for line in before.splitlines() if line.strip()]
    symbols = sorted(set(SYMBOL_RE.findall(text)))
    namespace_hints = sorted({s.split(".")[0] for s in symbols if "." in s})
    typeclass_hints = sorted({h for h in hypotheses if "[" in h or "inst" in h.lower()})
    return {
        "goal_text": goal.strip(),
        "local_hypotheses": hypotheses,
        "symbols": symbols,
        "namespace_hints": namespace_hints,
        "typeclass_hints": typeclass_hints,
    }
