"""Domain-specific document filters (Data Filter v1).

Every domain has its own rule list because the text statistics differ
(goal.md section 12):

    English : long explanatory documents, low noise      -> Gopher-style rules
    Korean  : much shorter documents, mixed scripts       -> lower length floor,
                                                             Hangul ratio rule
    Math    : LaTeX / tables / separators repeat legally  -> loose repetition rules
    Code    : licences, line lengths, binary-ish blobs    -> StarCoder-style rules

Thresholds live in configs/data/filters_v1.yaml.  A rule is a (feature, op,
threshold) triple; the first failing rule is reported as the rejection
reason so that the stats tell you *which* rule removes data.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from .quality import code_features, text_features

OPS: dict[str, Callable[[Any, Any], bool]] = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
    "==": operator.eq, "!=": operator.ne,
    "in": lambda a, b: a in b, "not_in": lambda a, b: a not in b,
}


@dataclass(frozen=True)
class Rule:
    feature: str          # key in features dict or metadata (prefix "meta.")
    op: str               # must hold for the document to be KEPT
    value: Any
    name: str | None = None

    @property
    def label(self) -> str:
        return self.name or f"{self.feature}{self.op}{self.value}"

    def passes(self, feats: dict[str, Any], meta: dict[str, Any]) -> bool:
        if self.feature.startswith("meta."):
            v = meta.get(self.feature[5:])
            if v is None:
                return True  # metadata missing -> rule does not apply
        else:
            v = feats.get(self.feature)
            if v is None:
                return True
        try:
            return OPS[self.op](v, self.value)
        except TypeError:
            return True


class DomainFilter:
    def __init__(self, domain: str, rules: list[Rule], feature_fn: Callable[[str], dict[str, Any]]):
        self.domain = domain
        self.rules = rules
        self.feature_fn = feature_fn

    def apply(self, text: str, meta: dict[str, Any] | None = None) -> tuple[bool, str | None, dict[str, Any]]:
        """Return (keep, reason, features)."""
        meta = meta or {}
        if not text or not text.strip():
            return False, "empty", {"n_chars": 0}
        feats = self.feature_fn(text)
        for rule in self.rules:
            if not rule.passes(feats, meta):
                return False, rule.label, feats
        return True, None, feats


def _parse_rules(items: list[dict[str, Any]]) -> list[Rule]:
    rules = []
    for it in items:
        rules.append(Rule(feature=it["feature"], op=it["op"], value=it["value"], name=it.get("name")))
    return rules


def load_filters(config_path: str | Path) -> dict[str, DomainFilter]:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = {}
    for domain, spec in cfg["domains"].items():
        fn = code_features if spec.get("feature_set", "text") == "code" else text_features
        out[domain] = DomainFilter(domain, _parse_rules(spec["rules"]), fn)
    return out


def filter_config_version(config_path: str | Path) -> str:
    with open(config_path, encoding="utf-8") as f:
        return str(yaml.safe_load(f).get("version", "unknown"))
