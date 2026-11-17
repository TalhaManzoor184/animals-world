"""
Runs ON KAGGLE (free GPU). Pushed automatically by pipeline/free_mode.py. Do not run on GitHub.

Reads job.json from the attached Kaggle dataset, renders each shot with the open-source
Wan 2.2 TI2V-5B model (Apache 2.0), and writes:
  /kaggle/working/out/shot_XX.mp4
  /kaggle/working/out/done.json   (updated after every shot, so partial results survive)

Shot kinds:
  portrait → image-to-video from the character's reference image (identity stays locked)
  action   → text-to-video from a full scene description
"""
import gc
import glob
import json
import os
import subprocess
import sys
import time
import traceback

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers>=0.35.0", "transformers>=4.51",
                "accelerate", "ftfy", "imageio", "imageio-ffmpeg", "sentencepiece"], check=True)
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # shots 1/4 OOM'd from fragmentation  # /kaggle/working is only ~20 GB; weights go to /tmp

import torch  # noqa: E402
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline  # noqa: E402
from diffusers.utils import export_to_video  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
OUT = "/kaggle/working/out"
os.makedirs(OUT, exist_ok=True)

job_path = (glob.glob("/kaggle/input/**/job.json", recursive=True) or [None])[0]
if not job_path:
    raise SystemExit("job.json not found under /kaggle/input")
BASE = os.path.dirname(job_path)
job = json.load(open(job_path))
cfg = job.get("render", {})
W, H = int(cfg.get("width", 480)), int(cfg.get("height", 832))
STEPS = int(cfg.get("steps", 30))
GUIDE = float(cfg.get("guidance", 5.0))
NEG = job.get("negative", "")

cap = torch.cuda.get_device_capability(0)
DTYPE = torch.bfloat16 if cap[0] >= 8 else torch.float16   # T4/P100 have no fast bf16
print(f"GPU {torch.cuda.get_device_name(0)} cap={cap} dtype={DTYPE} size={W}x{H} steps={STEPS}", flush=True)

done = {"job_id": job["job_id"], "gpu": torch.cuda.get_device_name(0), "shots": {}, "errors": {}}


def save_done():
    json.dump(done, open(f"{OUT}/done.json", "w"), indent=2)


def fit(img: Image.Image) -> Image.Image:
    return ImageOps.fit(img.convert("RGB"), (W, H), Image.LANCZOS, centering=(0.5, 0.4))


EMBEDS = {}


def load(cls, shots):
    """Load the pipeline, encode every prompt for these shots, then DROP the ~11 GB text encoder.
    Keeping it in RAM next to the transformer got the process OOM-killed on Kaggle (~29 GB RAM)."""
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = cls.from_pretrained(MODEL, vae=vae, torch_dtype=DTYPE)
    pipe.text_encoder.to("cuda")
    with torch.no_grad():
        for s in shots:
            pe, ne = pipe.encode_prompt(prompt=s["prompt"], negative_prompt=NEG,
                                        do_classifier_free_guidance=GUIDE > 1.0, num_videos_per_prompt=1,
                                        max_sequence_length=512, device=torch.device("cuda"))
            EMBEDS[s["n"]] = (pe.to("cpu"), ne.to("cpu") if ne is not None else None)
    te = pipe.text_encoder
    pipe.register_modules(text_encoder=None)
    del te
    gc.collect()
    torch.cuda.empty_cache()
    print(f"encoded {len(shots)} prompts; text encoder released", flush=True)
    pipe.enable_model_cpu_offload()
    for m in ("enable_tiling", "enable_slicing"):        # avoids the memory spike at VAE decode
        try:
            getattr(pipe.vae, m)()
        except Exception:
            pass
    return pipe


def frames_ok(frames) -> bool:
    import numpy as np
    arr = np.asarray(frames[len(frames) // 2], dtype="float32")
    return np.isfinite(arr).all() and arr.mean() > (2 if arr.max() > 1.5 else 0.01)


def render(pipe, shot, use_image: bool):
    gc.collect()
    torch.cuda.empty_cache()
    pe, ne = EMBEDS[shot["n"]]
    kw = dict(prompt_embeds=pe.to("cuda"), negative_prompt_embeds=ne.to("cuda") if ne is not None else None,
              height=H, width=W, num_frames=int(shot["frames"]),
              guidance_scale=GUIDE, num_inference_steps=STEPS,
              generator=torch.Generator("cpu").manual_seed(int(shot.get("seed", 42))))
    if use_image:
        kw["image"] = fit(Image.open(os.path.join(BASE, shot["image"])))
    t0 = time.time()
    frames = pipe(**kw).frames[0]
    del kw
    gc.collect()
    if not frames_ok(frames):
        raise RuntimeError("blank/NaN frames (fp16 overflow?)")
    path = f"{OUT}/shot_{shot['n']:02d}.mp4"
    export_to_video(frames, path, fps=24)
    done["shots"][str(shot["n"])] = {"file": os.path.basename(path), "seconds": round(time.time() - t0)}
    done["errors"].pop(str(shot["n"]), None)
    save_done()
    print(f"✔ shot {shot['n']} ({shot['kind']}) in {time.time() - t0:.0f}s", flush=True)


save_done()
for kind, cls, use_image in (("portrait", WanImageToVideoPipeline, True), ("action", WanPipeline, False)):
    shots = [s for s in job["shots"] if s["kind"] == kind]
    if not shots:
        continue
    pipe = load(cls, shots)
    for s in shots:
        for attempt in range(2):
            try:
                render(pipe, s, use_image)
                break
            except Exception as e:
                done["errors"][str(s["n"])] = f"{type(e).__name__}: {e}"
                save_done()
                traceback.print_exc()
                torch.cuda.empty_cache()
                s["seed"] = int(s.get("seed", 42)) + 1
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

done["finished"] = True
save_done()
print("ALL DONE", json.dumps(done), flush=True)
