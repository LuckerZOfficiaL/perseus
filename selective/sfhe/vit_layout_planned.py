"""E11: ViT pages of inputs with different private sizes, planned on the device: the BERT page driver
(examples/bert/selective_planned.py: per-page plaintexts re-recorded on the twin, per-page linears) over the ViT
SelectivePrefill's page / prefill_page, with the ViT linears (block linears + the patch embedding + the per-page linears)."""
from __future__ import annotations

from sfhe.bert_selective_planned import BertSelectivePlannedPrefill

from sfhe.vit_selective_planned import SelectivePlannedPrefill


class VitLayoutPlannedPrefill(BertSelectivePlannedPrefill):
    def linears(self):
        return SelectivePlannedPrefill.linears(self)
