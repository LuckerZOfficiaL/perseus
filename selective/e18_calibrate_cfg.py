#!/usr/bin/env python
"""Upstream's Llama approximation calibration (perseus.examples.llama.calibrate) with its softmax knobs read from this
checkout's python/perseus/configs/approximation/gpt2.yaml (calibrate.softmax_knobs builds the path one "perseus" too
deep here). Arguments as upstream's."""
import os
import sys

from omegaconf import OmegaConf

import perseus
from perseus.examples.llama import calibrate as C

C.softmax_knobs = lambda: OmegaConf.load(os.path.join(os.path.dirname(perseus.__file__), "configs", "approximation",
                                                      "gpt2.yaml")).softmax

if __name__ == "__main__":
    C.main(sys.argv[1:])
