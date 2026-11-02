"""sfhe: selective (two-stream / late-injection) FHE on top of an UNMODIFIED Perseus checkout.

Everything of ours lives here (and in selective/ scripts); no Perseus file is edited, so pulling upstream Perseus only means
re-running our checks (selective/check_upstream.sh). The few behaviours we need from Perseus that it does not offer are
applied at import time as small, idempotent runtime patches (sfhe.patches); the planner's separate processes get them
through selective/sfhe_site/sitecustomize.py when SFHE_PATCHES=1.

Modules: dense_xpt (route-B attention: plaintext keys as PCMMs, page layouts, both chunks), ccmm_mha_xpt (route A'),
vit_selective / bert_selective (the encrypted stream of each model), *_planned (device drivers), vit_late* (late injection).
"""
from . import patches

patches.apply()

CONFIGS = __import__("os").path.join(__import__("os").path.dirname(__file__), "configs")
