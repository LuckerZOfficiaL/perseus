#!/usr/bin/env python
"""Baseline B6b (image side, GPU): generative face replacement, the DeepPrivacy-style anonymisation. Each photo of the
encrypted face runs (faces5_{s,m,l}_orig.json) gets its private patches (the detected faces, 16 x 16 patches of the
224 x 224 model input) repainted by a diffusion inpainting model (Stable Diffusion 1.5 inpainting, fp16, prompt "a
photo of a person's face"): the 224 px input is upscaled to 512, inpainted inside the patch mask, scaled back to 224, and
pasted into the original outside the mask (so only the private pixels change). Seeded per photo. Writes one PNG per
photo under --out (the photo's relative path, .png); b6_image_anon.py scores them as method "genface".
Needs diffusers (the anon venv): ~/data/venvs/anon/bin/python selective/b6b_gen_faces.py
--ages: the E15 age photos instead (validation split of e15_ages.py; default --out ~/data/selective-fhe/e15/ages_genface).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
E = os.path.expanduser("~/data/selective-fhe/e12")
PATCH, GRID = 16, 14


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    ap.add_argument("--prompt", default="a photo of a person's face, natural, realistic")
    ap.add_argument("--negative", default="blurry, deformed, cartoon, painting")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help=f"default {E}/b6b_genface")
    ap.add_argument("--ages", action="store_true", help="the E15 age photos (e15_ages.py, validation split)")
    a = ap.parse_args()
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    from PIL import Image
    import e11_faces_common as F
    if a.ages:
        import e15_ages
        root = e15_ages.OUT
        det = json.load(open(f"{root}/faces.json"))["detections"]
        files = sorted(r[0] for r in json.load(open(f"{root}/ages.json"))["validation"])
        a.out = a.out or f"{root}_genface"
    else:
        root = F.ROOT
        det = json.load(open(F.FACES))["detections"]
        files = sorted({r["file"] for b in ("s", "m", "l") for r in json.load(open(f"{E}/faces5_{b}_orig.json"))["results"]})
        a.out = a.out or f"{E}/b6b_genface"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(a.model, torch_dtype=torch.float16, variant="fp16",
                                                          safety_checker=None).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    done = 0
    for k, rel in enumerate(files):
        dst = os.path.join(a.out, os.path.splitext(rel)[0] + ".png")
        if os.path.exists(dst):
            continue
        img = Image.open(os.path.join(root, rel)).convert("RGB").resize((224, 224), Image.BILINEAR)
        mask = np.zeros((224, 224), dtype=np.uint8)
        for p in det[rel]["patches"]:
            r, c = divmod(int(p), GRID)
            mask[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = 255
        gen = pipe(prompt=a.prompt, negative_prompt=a.negative, image=img.resize((512, 512), Image.BICUBIC),
                   mask_image=Image.fromarray(mask).resize((512, 512), Image.NEAREST), height=512, width=512,
                   num_inference_steps=a.steps, guidance_scale=a.guidance,
                   generator=torch.Generator("cuda").manual_seed(a.seed + k)).images[0]
        gen = np.asarray(gen.resize((224, 224), Image.BILINEAR))
        out = np.where(mask[..., None] > 0, gen, np.asarray(img))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        Image.fromarray(out.astype(np.uint8)).save(dst)
        done += 1
        if done % 50 == 0:
            print(f"[b6b] {k + 1}/{len(files)}", flush=True)
    print(f"[b6b] {len(files)} photos ({done} new) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
