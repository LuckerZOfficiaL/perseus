#!/usr/bin/env python
"""E21 (breadth), BERT-family encoders on every text task: the plaintext full / two-stream / redaction accuracy
(e13_hatexplain.accuracy, the same code as every E13-E15 text table) on the validation texts, with the task's detector
file (b5_text_sanitize.TASKS); the hiding baselines come from b5_text_sanitize.py / b5b_typed_sanitize.py run on the
same (task, model) with --out next to it. One file per (task, model): ~/data/selective-fhe/e21/text/<task>_<tag>_acc.json.

  python selective/e21_text.py --task drugsc --model ~/data/selective-fhe/e21/bert/drugsc_large --tag bert-large
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser("~/data/selective-fhe/e21/text")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    import e13_hatexplain as H13
    from b5_text_sanitize import TASKS
    from b5b_typed_sanitize import rows_of
    _data, privf, T = TASKS[a.task]
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    data = {"validation": rows_of(a.task, "validation")}
    priv = json.load(open(privf))
    os.makedirs(OUT, exist_ok=True)
    H13.accuracy(os.path.expanduser(a.model), tok, data, priv, f"{OUT}/{a.task}_{a.tag}_acc.json", max_len=T,
                 focus=("class0", 0))


if __name__ == "__main__":
    main()
