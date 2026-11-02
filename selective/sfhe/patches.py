"""Runtime patches over an unmodified Perseus (idempotent; each one guards on a marker attribute).

1. perseus.examples.bert.data: the tokenizer is loaded once (upstream reloads it per sentence: 872 Hub requests hit the
   rate limit), BERT_EXPORT points the BERT example at another export of the same architecture (fine-tuned weights), and
   SFHE_TEXT_DATA at another dataset of texts (a JSON {split: [[text, label], ...]}, e13_hatexplain.py) in place of
   SST-2's sentences; the tokenizer stays upstream's (bert-base-uncased vocabulary; another model must share it).
2. perseus.plan.placer.emit.assemble: a stage with no typed refresh sites (late injection's hand-over) has
   predicted_rel_err median / p90 = None, which the planner's log line formats with '%.3g' and crashes; the summary's
   dict is wrapped so those reads format as 'n/a' (the plan JSON is unchanged: json reads the stored None).
"""
from __future__ import annotations

import functools
import os
from pathlib import Path


def bert_data():
    try:
        from perseus.examples.bert import data as D
    except ImportError:
        return
    if getattr(D, "_sfhe_patched", False):
        return
    D.tokenizer = functools.lru_cache(maxsize=None)(D.tokenizer)
    upstream_export_dir = D.export_dir

    def export_dir():
        if os.environ.get("BERT_EXPORT"):
            return Path(os.path.expanduser(os.environ["BERT_EXPORT"]))
        return upstream_export_dir()
    D.export_dir = export_dir
    upstream_sentences = D.sentences

    @functools.lru_cache(maxsize=None)
    def _texts(path):
        import json
        return json.load(open(os.path.expanduser(path)))

    def sentences(n, split="validation"):
        if os.environ.get("SFHE_TEXT_DATA"):
            rows = _texts(os.environ["SFHE_TEXT_DATA"])[split]
            return [(t, int(lab)) for t, lab in rows[:n]]
        return upstream_sentences(n, split)
    D.sentences = sentences
    D._sfhe_patched = True


class _NA:
    def __format__(self, spec):
        return "n/a"


class _NoneFormats(dict):
    """dict whose None values read back as an object that formats as 'n/a' (json.dump still writes null)."""

    def __getitem__(self, k):
        v = dict.__getitem__(self, k)
        return _NA() if v is None else v


def planner_log():
    try:
        from perseus.plan.placer import emit
    except ImportError:
        return
    if getattr(emit, "_sfhe_patched", False):
        return
    upstream_assemble = emit.assemble

    @functools.wraps(upstream_assemble)
    def assemble(*a, **k):
        res = upstream_assemble(*a, **k)
        bq = (res.get("summary") or {}).get("bts_quality") or {}
        pe = bq.get("predicted_rel_err")
        if isinstance(pe, dict) and not isinstance(pe, _NoneFormats) and None in pe.values():
            bq["predicted_rel_err"] = _NoneFormats(pe)
        return res
    emit.assemble = assemble
    emit._sfhe_patched = True


def apply():
    bert_data()
    planner_log()
