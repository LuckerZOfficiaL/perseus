"""Loaded by any Python started with selective/sfhe_site on PYTHONPATH: with SFHE_PATCHES=1, apply sfhe's runtime patches
(the planner runs in processes of upstream's make_plan.sh). Then the system's own sitecustomize, if any."""
import importlib.util
import os
import sys

if os.environ.get("SFHE_PATCHES") not in (None, "", "0"):
    here = os.path.dirname(os.path.abspath(__file__))
    sel = os.path.dirname(here)
    if sel not in sys.path:
        sys.path.append(sel)
    try:
        import sfhe  # noqa: F401  (applies the patches)
    except Exception as e:  # never break an unrelated interpreter
        print(f"[sfhe_site] patches not applied: {e}", file=sys.stderr)

_sys_site = "/usr/lib/python3.12/sitecustomize.py"
if os.path.exists(_sys_site) and os.path.abspath(__file__) != _sys_site:
    spec = importlib.util.spec_from_file_location("_system_sitecustomize", _sys_site)
    try:
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    except Exception:
        pass
