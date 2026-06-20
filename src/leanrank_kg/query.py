from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Literal

from .parse_context import parse_context

InputType = Literal["theorem", "proof_state", "goal", "lean"]

DECL_RE = re.compile(r"\b(theorem|lemma|example)\s+([A-Za-z_][A-Za-z0-9_'.]*)?")
IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_'.]+)", re.MULTILINE)
NAMESPACE_RE = re.compile(r"^\s*namespace\s+([A-Za-z0-9_'.]+)", re.MULTILINE)
OPEN_RE = re.compile(r"^\s*open\s+([A-Za-z0-9_'. ]+)", re.MULTILINE)
TOKEN_RE = re.compile(r"[A-Za-z_\u0370-\u03FF][A-Za-z0-9_\u0370-\u03FF'.]*")
OPERATOR_RE = re.compile(r"∀|∃|→|↔|≤|≥|=|≠|<|>|\+|-|\*|/|∧|∨|¬")
SORT_RE = re.compile(r"\b(?:Prop|Sort|Type(?:\s+[A-Za-z0-9_]+)?)\b")
LEAN_KEYWORDS = {
    "Prop",
    "Sort",
    "Type",
    "by",
    "fun",
    "let",
    "match",
    "if",
    "then",
    "else",
    "forall",
    "exists",
    "theorem",
    "lemma",
    "example",
}

BRACKET_PAIRS = {"(": ")", "{": "}", "[": "]", "⦃": "⦄"}
BRACKET_KINDS = {"(": "explicit", "{": "implicit", "[": "typeclass", "⦃": "strict_implicit"}


def _strip_proof_body(text: str) -> str:
    for marker in [" := by", ":= by", " :=\n", ":=\n"]:
        if marker in text:
            return text.split(marker, 1)[0]
    return text


def _declaration_header(text: str) -> str:
    head = _strip_proof_body(text)
    for marker in [" where", "\nwhere"]:
        if marker in head:
            head = head.split(marker, 1)[0]
    return head.strip()


def _extract_decl_name(text: str) -> str | None:
    match = DECL_RE.search(text)
    if not match:
        return None
    name = match.group(2)
    return name if name and name != ":" else None


def _find_top_level_colon(text: str) -> int:
    stack: list[str] = []
    for idx, char in enumerate(text):
        if char in BRACKET_PAIRS:
            stack.append(BRACKET_PAIRS[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char == ":" and not stack:
            return idx
    return -1


def _split_top_level_colon(text: str) -> tuple[str, str | None]:
    idx = _find_top_level_colon(text)
    if idx < 0:
        return text.strip(), None
    return text[:idx].strip(), text[idx + 1 :].strip()


def _extract_goal_from_decl(text: str) -> str:
    head = _declaration_header(text)
    colon_idx = _find_top_level_colon(head)
    if colon_idx < 0:
        return head.strip()
    return head[colon_idx + 1 :].strip()


def _extract_binder_groups(text: str) -> list[dict[str, str]]:
    head = _declaration_header(text)
    groups = []
    idx = 0
    while idx < len(head):
        char = head[idx]
        close = BRACKET_PAIRS.get(char)
        if not close:
            idx += 1
            continue
        depth = 1
        end = idx + 1
        while end < len(head) and depth:
            if head[end] == char:
                depth += 1
            elif head[end] == close:
                depth -= 1
            end += 1
        if depth == 0:
            raw = head[idx:end]
            content = raw[1:-1].strip()
            if content:
                groups.append({"raw": raw, "content": content, "kind": BRACKET_KINDS[char]})
            idx = end
        else:
            idx += 1
    return groups


def _binder_names(name_part: str) -> list[str]:
    names = []
    for token in TOKEN_RE.findall(name_part):
        if token not in LEAN_KEYWORDS and not token[0].isupper():
            names.append(token)
    return names


def _structured_binder_groups(text: str) -> list[dict]:
    structured = []
    for group in _extract_binder_groups(text):
        name_part, type_part = _split_top_level_colon(group["content"])
        names = _binder_names(name_part) if type_part else []
        binder_type = type_part or group["content"]
        structured.append(
            {
                "raw": group["raw"],
                "content": group["content"],
                "kind": group["kind"],
                "names": names,
                "type": binder_type,
            }
        )
    return structured


def _extract_hypotheses_from_decl(text: str) -> list[str]:
    return [group["content"] for group in _extract_binder_groups(text)]


def _namespace_hints(text: str, full_name: str | None) -> list[str]:
    open_namespaces = {item for match in OPEN_RE.findall(text) for item in match.split() if item}
    hints = set(IMPORT_RE.findall(text)) | set(NAMESPACE_RE.findall(text)) | open_namespaces
    if full_name and "." in full_name:
        hints.add(full_name.rsplit(".", 1)[0])
    return sorted(hints)


def _typeclass_hints_from_decl(text: str) -> list[str]:
    return sorted({group["content"] for group in _extract_binder_groups(text) if group["kind"] == "typeclass"})


def _typeclass_symbols(typeclass_hints: list[str]) -> list[str]:
    return sorted({token for hint in typeclass_hints for token in TOKEN_RE.findall(hint) if token not in LEAN_KEYWORDS})


def _operator_symbols(text: str) -> list[str]:
    return sorted(set(OPERATOR_RE.findall(text)))


def _sort_symbols(text: str) -> list[str]:
    return sorted(set(SORT_RE.findall(text)))


def _constant_symbols(text: str, binder_names: set[str] | None = None) -> list[str]:
    binder_names = binder_names or set()
    constants = []
    for token in TOKEN_RE.findall(text):
        if token in binder_names or token in LEAN_KEYWORDS:
            continue
        constants.append(token)
    return sorted(set(constants))


def _normalize_with_binders(text: str, binder_groups: list[dict]) -> str:
    replacements: dict[str, str] = {}
    counter = 0
    for group in binder_groups:
        for name in group.get("names", []):
            if name not in replacements:
                replacements[name] = f"v{counter}"
                counter += 1
    normalized = text
    for name, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = re.sub(rf"\b{re.escape(name)}\b", replacement, normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _parsed_feature_summary(
    *,
    binder_groups: list[dict],
    conclusion_symbols: list[str],
    operator_symbols: list[str],
    sort_symbols: list[str],
    typeclass_hints: list[str],
    normalized_goal_text: str,
) -> dict:
    return {
        "binder_count": len(binder_groups),
        "explicit_binder_count": sum(1 for group in binder_groups if group.get("kind") == "explicit"),
        "implicit_binder_count": sum(1 for group in binder_groups if "implicit" in str(group.get("kind", ""))),
        "typeclass_binder_count": sum(1 for group in binder_groups if group.get("kind") == "typeclass"),
        "conclusion_symbol_count": len(conclusion_symbols),
        "operator_symbol_count": len(operator_symbols),
        "sort_symbol_count": len(sort_symbols),
        "typeclass_hint_count": len(typeclass_hints),
        "normalized_goal_text": normalized_goal_text,
    }


def _query_id(input_type: str, full_name: str | None, text: str) -> str:
    from .utils import stable_hash

    stem = full_name or stable_hash(text, 16)
    return f"query:{input_type}:{stem}"


@dataclass(frozen=True)
class NewProofStateQuery:
    raw_text: str
    input_type: InputType = "proof_state"
    full_name: str | None = None
    domain_hint: str | None = None
    file_path: str | None = None
    goal_text: str = ""
    local_hypotheses: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    namespace_hints: list[str] = field(default_factory=list)
    typeclass_hints: list[str] = field(default_factory=list)
    typeclass_symbols: list[str] = field(default_factory=list)
    binder_groups: list[dict] = field(default_factory=list)
    conclusion_symbols: list[str] = field(default_factory=list)
    operator_symbols: list[str] = field(default_factory=list)
    sort_symbols: list[str] = field(default_factory=list)
    normalized_goal_text: str = ""
    parsed_feature_summary: dict = field(default_factory=dict)

    @property
    def query_id(self) -> str:
        return _query_id(self.input_type, self.full_name, self.raw_text)

    @property
    def retrieval_text(self) -> str:
        goal = self.normalized_goal_text or self.goal_text or self.raw_text
        parts = [self.full_name or "", self.domain_hint or "", goal]
        return "\n".join(part for part in parts if part).strip()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["query_id"] = self.query_id
        data["retrieval_text"] = self.retrieval_text
        return data

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        input_type: InputType = "proof_state",
        full_name: str | None = None,
        domain_hint: str | None = None,
        file_path: str | None = None,
    ) -> "NewProofStateQuery":
        parsed = parse_context(text)
        conclusion_symbols = _constant_symbols(parsed["goal_text"])
        operator_symbols = _operator_symbols(parsed["goal_text"])
        sort_symbols = _sort_symbols(text)
        normalized_goal_text = re.sub(r"\s+", " ", parsed["goal_text"]).strip()
        return cls(
            raw_text=text,
            input_type=input_type,
            full_name=full_name,
            domain_hint=domain_hint,
            file_path=file_path,
            goal_text=parsed["goal_text"],
            local_hypotheses=parsed["local_hypotheses"],
            symbols=parsed["symbols"],
            namespace_hints=parsed["namespace_hints"],
            typeclass_hints=parsed["typeclass_hints"],
            typeclass_symbols=_typeclass_symbols(parsed["typeclass_hints"]),
            conclusion_symbols=conclusion_symbols,
            operator_symbols=operator_symbols,
            sort_symbols=sort_symbols,
            normalized_goal_text=normalized_goal_text,
            parsed_feature_summary=_parsed_feature_summary(
                binder_groups=[],
                conclusion_symbols=conclusion_symbols,
                operator_symbols=operator_symbols,
                sort_symbols=sort_symbols,
                typeclass_hints=parsed["typeclass_hints"],
                normalized_goal_text=normalized_goal_text,
            ),
        )


@dataclass(frozen=True)
class NewTheoremQuery(NewProofStateQuery):
    input_type: InputType = "theorem"

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        full_name: str | None = None,
        domain_hint: str | None = None,
        file_path: str | None = None,
        input_type: InputType = "theorem",
    ) -> "NewTheoremQuery":
        detected_name = full_name or _extract_decl_name(text)
        goal_text = _extract_goal_from_decl(text)
        hypotheses = _extract_hypotheses_from_decl(text)
        binder_groups = _structured_binder_groups(text)
        binder_names = {name for group in binder_groups for name in group.get("names", [])}
        context_like = "\n".join([*hypotheses, f"⊢ {goal_text}" if goal_text else text])
        parsed = parse_context(context_like)
        namespaces = sorted(set(parsed["namespace_hints"]) | set(_namespace_hints(text, detected_name)))
        typeclasses = sorted(set(parsed["typeclass_hints"]) | set(_typeclass_hints_from_decl(text)))
        conclusion_symbols = _constant_symbols(parsed["goal_text"], binder_names)
        operator_symbols = _operator_symbols(parsed["goal_text"])
        sort_symbols = _sort_symbols(_declaration_header(text))
        normalized_goal_text = _normalize_with_binders(parsed["goal_text"], binder_groups)
        return cls(
            raw_text=text,
            input_type=input_type,
            full_name=detected_name,
            domain_hint=domain_hint,
            file_path=file_path,
            goal_text=parsed["goal_text"],
            local_hypotheses=parsed["local_hypotheses"],
            symbols=parsed["symbols"],
            namespace_hints=namespaces,
            typeclass_hints=typeclasses,
            typeclass_symbols=_typeclass_symbols(typeclasses),
            binder_groups=binder_groups,
            conclusion_symbols=conclusion_symbols,
            operator_symbols=operator_symbols,
            sort_symbols=sort_symbols,
            normalized_goal_text=normalized_goal_text,
            parsed_feature_summary=_parsed_feature_summary(
                binder_groups=binder_groups,
                conclusion_symbols=conclusion_symbols,
                operator_symbols=operator_symbols,
                sort_symbols=sort_symbols,
                typeclass_hints=typeclasses,
                normalized_goal_text=normalized_goal_text,
            ),
        )


def build_query(
    text: str,
    *,
    input_type: InputType = "lean",
    full_name: str | None = None,
    domain_hint: str | None = None,
    file_path: str | None = None,
) -> NewProofStateQuery:
    if input_type not in {"theorem", "proof_state", "goal", "lean"}:
        raise ValueError(f"Unknown query input_type: {input_type}")
    if input_type in {"theorem", "lean"} and DECL_RE.search(text):
        return NewTheoremQuery.from_text(text, full_name=full_name, domain_hint=domain_hint, file_path=file_path, input_type="theorem")
    if input_type == "goal":
        text = text if "⊢" in text else f"⊢ {text}"
    return NewProofStateQuery.from_text(text, input_type=input_type if input_type != "lean" else "proof_state", full_name=full_name, domain_hint=domain_hint, file_path=file_path)
