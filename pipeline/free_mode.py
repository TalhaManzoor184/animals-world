#!/usr/bin/env python3
"""
FREE production mode: $0 per day.

  Script/ideas/captions → Gemini API free tier (text model)
  Voices               → Gemini free-tier TTS, falling back to Kokoro-82M (open source, runs on the GitHub runner)
  Video                → Wan 2.2 TI2V-5B (open source) on Kaggle's free GPU
  Edit + publish       → ffmpeg on GitHub Actions (same as paid mode)

Flow (driven by pipeline/daily.py):
  kickoff  : episode → shot list → voices → upload job to a private Kaggle dataset → start the Kaggle GPU kernel
  collect  : (hours later) download the clips, edit, thumbnail; daily.py then QC-gates and publishes

  python pipeline/free_mode.py kickoff [--ep EP001]   # --ep: use an existing episode instead of writing a new one
  python pipeline/free_mode.py status
  python pipeline/free_mode.py collect
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import studio  # noqa: E402

ROOT = studio.ROOT
KERNEL_SRC = Path(__file__).resolve().parent / "free" / "render_kernel.py"
DATASET_SLUG = os.getenv("KAGGLE_DATASET_SLUG", "animals-world-jobs")
KERNEL_SLUG = os.getenv("KAGGLE_KERNEL_SLUG", "animals-world-render")
FREE_W, FREE_H = int(os.getenv("FREE_WIDTH", "480")), int(os.getenv("FREE_HEIGHT", "832"))
FREE_STEPS = int(os.getenv("FREE_STEPS", "30"))
ACTION_FRAMES = int(os.getenv("FREE_ACTION_FRAMES", "81"))          # 81 frames @24fps ≈ 3.4s
TTS_ENGINE = os.getenv("TTS_ENGINE", "auto")                          # auto | gemini | kokoro
FPS = 24


def log(m: str) -> None:
    print(m, flush=True)


def kaggle_user() -> str:
    u = os.getenv("KAGGLE_USERNAME", "").strip()
    if not u:
        studio.die("KAGGLE_USERNAME / KAGGLE_KEY secrets are not set.")
    return u


def kaggle(*args: str, check: bool = True) -> str:
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if check and r.returncode != 0:
        studio.die(f"kaggle {' '.join(args)} failed:\n{out[-2000:]}")
    return out


# ============================================================ shots
def portrait_image(key: str) -> Path:
    ch = studio.load_chars()["characters"][key]
    for p in (ch.get("portrait"), ch.get("sheet"), ch.get("ref")):
        if p and (ROOT / p).is_file():
            return ROOT / p
    studio.die(f"No reference image for {key}")


def build_shots(ep: dict) -> list[dict]:
    """Every spoken line → a close-up 'portrait' shot of the speaker (image-to-video from their reference,
    so identity is locked). Scenes with ≤1 line also get one 'action' shot (text-to-video) for the physical gag."""
    chars = studio.load_chars()
    C, S = chars["characters"], chars["style"]
    shots: list[dict] = []
    for sc in ep["scenes"]:
        lines = sc.get("dialogue", [])
        im, v = sc["image"], sc["video"]
        action = None
        if len(lines) <= 1:
            cast = ", ".join(f"{C[k]['name']} the {C[k]['look']}" for k in sc["cast"])
            action = {"kind": "action", "beat": sc["beat"],
                      "prompt": (f"Photorealistic cinematic vertical video. {cast}. {v['motion']}. "
                                 f"Setting: {im['environment']}. Lighting: {im['lighting']}. Camera: {v['camera_move']}. "
                                 f"{S['visual']}. Natural animal movement, one continuous shot.")}
        line_shots = []
        for dl in lines:
            ch = C[dl["speaker"]]
            line_shots.append({
                "kind": "portrait", "beat": sc["beat"], "speaker": dl["speaker"], "dialogue": dl,
                "image_src": str(portrait_image(dl["speaker"]).relative_to(ROOT)),
                "prompt": (f"Photorealistic close-up of {ch['name']}, {ch['look']}. The {ch['species']} looks into the "
                           f"camera and talks with a {dl['emotion']} expression, mouth moving naturally as if speaking, "
                           f"subtle head and ear movement, blinking. Keep the face, fur colours, clothing and "
                           f"accessories exactly the same. Slow gentle camera push-in. {S['visual']}.")})
        early = bool(lines) and lines[0]["at"][0] < 2.0
        shots += (line_shots + [action] if early else [action] + line_shots) if action else line_shots
    shots = [s for s in shots if s]
    for i, s in enumerate(shots, 1):
        s["n"] = i
        s["seed"] = 1000 + i
    return shots


# ============================================================ voices
def wav_seconds(p: Path) -> float:
    with wave.open(str(p)) as w:
        return w.getnframes() / float(w.getframerate())


def tts_gemini(text: str, emotion: str, ch: dict, out: Path) -> None:
    from google.genai import types
    c = studio.client()
    resp = c.models.generate_content(
        model=studio.TTS_MODEL,
        contents=f"Say this {emotion}, as a {ch['voice']['description']}: {text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=ch["voice"]["tts_voice"])))))
    pcm = resp.candidates[0].content.parts[0].inline_data.data
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000); wf.writeframes(pcm)


_KOKORO: dict = {}


def tts_kokoro(text: str, ch: dict, out: Path) -> None:
    try:
        from kokoro import KPipeline
    except ImportError:
        log("  installing Kokoro TTS (one-time per run)…")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kokoro>=0.9", "soundfile"], check=True)
        from kokoro import KPipeline
    import numpy as np
    import soundfile as sf
    voice = ch["voice"].get("kokoro_voice", "am_michael")
    lang = voice[0]  # 'a' = American, 'b' = British
    if lang not in _KOKORO:
        _KOKORO[lang] = KPipeline(lang_code=lang)
    chunks = [a for _, _, a in _KOKORO[lang](text, voice=voice, speed=ch["voice"].get("kokoro_speed", 1.0))]
    audio = np.concatenate([np.asarray(c) for c in chunks])
    sf.write(str(out), audio, 24000, subtype="PCM_16")


def make_voice(text: str, emotion: str, key: str, out: Path) -> str:
    ch = studio.load_chars()["characters"][key]
    if TTS_ENGINE in ("auto", "gemini"):
        try:
            tts_gemini(text, emotion, ch, out)
            return "gemini"
        except BaseException as e:  # quota (429), SystemExit from missing key, etc.
            if TTS_ENGINE == "gemini":
                raise
            log(f"  Gemini TTS unavailable ({type(e).__name__}: {str(e)[:120]}); using Kokoro")
    tts_kokoro(text, ch, out)
    return "kokoro"


def frames_for(seconds: float) -> int:
    """Wan needs 4k+1 frames."""
    f = max(49, math.ceil(seconds * FPS))
    return min(121, 4 * math.ceil((f - 1) / 4) + 1)


# ============================================================ free episode (what the editor consumes)
def build_free_episode(ep: dict, d: Path, shots: list[dict]) -> dict:
    vo = studio.sub(d, "09_Voice_Over")
    scenes = []
    for s in shots:
        if s["kind"] == "portrait":
            dl = s["dialogue"]
            wav = vo / f"scene_{s['n']:02d}_line_01_{s['speaker']}.wav"
            if not wav.exists():
                engine = make_voice(dl["line"], dl["emotion"], s["speaker"], wav)
                log(f"  🔊 shot {s['n']}: {s['speaker']} ({engine}) \"{dl['line']}\"")
            secs = wav_seconds(wav)
            dur = max(1.6, secs + 0.55)
            s["frames"] = frames_for(dur)
            dur = min(dur, s["frames"] / FPS)
            scenes.append({"n": s["n"], "beat": s["beat"], "cast": [s["speaker"]], "clip_seconds": s["frames"] / FPS,
                           "use": [0.0, round(dur, 3)],
                           "dialogue": [{**dl, "at": [0.25, round(min(0.25 + secs, dur), 3)]}],
                           "punch": [[0.25, dur, 1.12]] if dl.get("highlight") else []})
        else:
            s["frames"] = ACTION_FRAMES
            scenes.append({"n": s["n"], "beat": s["beat"], "cast": [], "clip_seconds": ACTION_FRAMES / FPS,
                           "use": [0.0, round(ACTION_FRAMES / FPS, 3)], "dialogue": []})
    free = {**ep, "scenes": scenes, "production": "free"}
    free["edit"] = {**ep["edit"], "export": {**ep["edit"]["export"]}}
    (d / "episode.free.json").write_text(json.dumps(free, indent=2, ensure_ascii=False))
    (d / "shots.json").write_text(json.dumps(shots, indent=2, ensure_ascii=False))
    total = sum(sc["use"][1] for sc in scenes)
    log(f"  shot list: {len(shots)} shots, ~{total:.1f}s final runtime")
    if total < 15:
        log("  ⚠ under 15s: consider a longer script")
    return free


# ============================================================ kickoff
def push_dataset(job: dict, shots: list[dict], user: str) -> None:
    ds = Path(tempfile.mkdtemp(prefix="kds_"))
    for s in shots:
        if s["kind"] == "portrait":
            name = f"img_{s['speaker']}{Path(s['image_src']).suffix}"
            shutil.copy(ROOT / s["image_src"], ds / name)
            s["image"] = name
    job["shots"] = [{k: s[k] for k in ("n", "kind", "prompt", "frames", "seed", "image") if k in s} for s in shots]
    (ds / "job.json").write_text(json.dumps(job, indent=2))
    (ds / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET_SLUG, "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "CC0-1.0"}]}))
    exists = "ready" in kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False).lower()
    if exists:
        kaggle("datasets", "version", "-p", str(ds), "-m", job["job_id"], "--dir-mode", "zip")
    else:
        kaggle("datasets", "create", "-p", str(ds), "--dir-mode", "zip")   # private by default
    for _ in range(60):  # wait until the new version is processed, or the kernel may mount the old one
        time.sleep(10)
        if "ready" in kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False).lower():
            break
    time.sleep(30)
    log(f"  ✔ job uploaded to kaggle.com/datasets/{user}/{DATASET_SLUG} (private)")


def push_kernel(user: str) -> None:
    kd = Path(tempfile.mkdtemp(prefix="kk_"))
    shutil.copy(KERNEL_SRC, kd / "render_kernel.py")
    (kd / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "render_kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "machine_shape": os.getenv("KAGGLE_GPU", "NvidiaTeslaT4"),
        "dataset_sources": [f"{user}/{DATASET_SLUG}"], "competition_sources": [], "kernel_sources": [],
        "model_sources": []}, indent=2))
    log(kaggle("kernels", "push", "-p", str(kd)))


def kickoff(ep_id: str | None, day_state: dict | None = None) -> dict:
    """Returns the state fields to save: episode, free_job."""
    user = kaggle_user()
    if not ep_id:
        a = argparse.Namespace(ep=None, scene=None, force=False)
        studio.cmd_ideas(a)
        ep_id = a.ep
    ep, d = studio.load_ep(ep_id)
    log(f"→ free kickoff for {ep_id}: {ep['title']}")
    shots = build_shots(ep)
    build_free_episode(ep, d, shots)
    job_id = f"{ep_id}-{int(time.time())}"
    job = {"job_id": job_id, "episode": ep_id,
           "negative": "cartoon, anime, blurry, deformed, extra limbs, duplicate animals, morphing face, "
                       "changing clothes, text, watermark, subtitles, worst quality, static frame",
           "render": {"width": FREE_W, "height": FREE_H, "steps": FREE_STEPS, "guidance": 5.0}}
    push_dataset(job, shots, user)
    push_kernel(user)
    log(f"✔ Kaggle GPU render started: kaggle.com/code/{user}/{KERNEL_SLUG}  (job {job_id})")
    return {"episode": ep_id, "free_job": job_id}


# ============================================================ collect
def status() -> str:
    out = kaggle("kernels", "status", f"{kaggle_user()}/{KERNEL_SLUG}", check=False).lower()
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in out:
            return s
    return out[:200]


def collect(ep_id: str, job_id: str) -> bool:
    """True when clips are in place and the final video is edited. False = still rendering (try later)."""
    ep, d = studio.load_ep(ep_id)
    st = status()
    log(f"  Kaggle kernel status: {st}")
    if st in ("running", "queued"):
        return False
    tmp = Path(tempfile.mkdtemp(prefix="kout_"))
    kaggle("kernels", "output", f"{kaggle_user()}/{KERNEL_SLUG}", "-p", str(tmp))
    done_p = next(tmp.rglob("done.json"), None)
    if not done_p:
        studio.die(f"Kaggle run ended ({st}) without done.json; open the kernel log on kaggle.com.")
    done = json.loads(done_p.read_text())
    if done.get("job_id") != job_id:
        studio.die(f"Kaggle output is for job {done.get('job_id')}, expected {job_id} (stale run). Re-run kickoff.")
    shots = json.loads((d / "shots.json").read_text())
    vid_dir = studio.sub(d, "08_Video_Assets")
    missing = []
    for s in shots:
        f = next(tmp.rglob(f"shot_{s['n']:02d}.mp4"), None)
        if f:
            shutil.copy(f, vid_dir / studio.scene_name(s["n"], "mp4"))
        else:
            missing.append(s["n"])
    log(f"  GPU: {done.get('gpu')}  rendered: {len(shots) - len(missing)}/{len(shots)}  "
        f"times: {[v['seconds'] for v in done['shots'].values()]}")
    if missing:
        free = json.loads((d / "episode.free.json").read_text())
        free["scenes"] = [sc for sc in free["scenes"] if sc["n"] not in missing]
        (d / "episode.free.json").write_text(json.dumps(free, indent=2, ensure_ascii=False))
        log(f"  ⚠ shots {missing} failed ({done.get('errors')}); editing without them")
        if not free["scenes"]:
            studio.die("No shots rendered.")
    edit_and_thumb(ep_id)
    return True


def edit_and_thumb(ep_id: str) -> None:
    studio.EPISODE_FILE = "episode.free.json"
    studio.DIALOGUE_MODE = "tts"
    a = argparse.Namespace(ep=ep_id, scene=None, force=False)
    studio.cmd_edit(a)
    free_thumb(ep_id)
    studio.cmd_qc(a)


# ============================================================ free thumbnail / post frames
def final_video(ep_id: str) -> Path | None:
    d = studio.ep_dir(ep_id)
    return next((d / "11_Final_Edited_Video").glob("*_final.mp4"), None)


def key_shot(ep: dict) -> dict:
    """The punchline shot (highlighted line), else the middle portrait shot, else the middle shot."""
    sc = ep["scenes"]
    return (next((x for x in sc if any(dl.get("highlight") for dl in x.get("dialogue", []))), None)
            or (lambda p: p[len(p) // 2] if p else None)([x for x in sc if x.get("dialogue")])
            or sc[len(sc) // 2])


def shot_frame(d: Path, shot: dict, out: Path) -> Path:
    """Clean frame (no burned captions) from the raw clip, scaled to fill 1080x1920."""
    clip = d / "08_Video_Assets" / studio.scene_name(shot["n"], "mp4")
    t = (shot["use"][0] + shot["use"][1]) / 2
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(clip), "-frames:v", "1",
                    "-vf", "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920",
                    str(out)], check=True)
    return out


def load_free(ep_id: str) -> tuple[dict, Path]:
    d = studio.ep_dir(ep_id)
    return json.loads((d / "episode.free.json").read_text(encoding="utf-8")), d


def free_thumb(ep_id: str) -> Path:
    from PIL import Image, ImageDraw, ImageFont
    ep, d = load_free(ep_id)
    out_dir = studio.sub(d, "12_Thumbnail")
    raw = shot_frame(d, key_shot(ep), out_dir / "thumbnail_raw.png")
    im = Image.open(raw).convert("RGB").resize((1080, 1920))
    text = "".join(ch for ch in ep["thumbnail"].get("overlay_text", "") if ord(ch) < 0x10000).strip().upper()
    if text:
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype(str(next(studio.FONT_DIR.glob("*.ttf"))), 140)
        w = dr.textbbox((0, 0), text, font=font, stroke_width=12)[2]
        dr.text(((1080 - w) // 2, 200), text, font=font, fill=(255, 229, 0), stroke_width=12, stroke_fill=(0, 0, 0))
    final = out_dir / f"{ep['id']}_thumbnail.jpg"
    im.save(final, quality=92)
    return final


def free_post(ep_id: str, day: Path) -> Path:
    """Meme still from today's reel + caption written by the free Gemini text model."""
    from PIL import Image, ImageDraw, ImageFont
    ep, d = load_free(ep_id)
    vid = final_video(ep_id)
    if not vid:
        studio.die("No reel yet today to make the post from.")
    shot = key_shot(ep)
    dl0 = (shot.get("dialogue") or [None])[0]
    day.mkdir(parents=True, exist_ok=True)
    raw = shot_frame(d, shot, day / "post_raw.png")
    meta = {"meme_text": ep["title"], "caption": ep["metadata"]["instagram"]["caption"],
            "question": ep["metadata"]["instagram"]["engagement_question"],
            "hashtags": ep["metadata"]["instagram"]["hashtags"]}
    try:
        from google.genai import types
        prompt = (f"Write a meme post for this still from our funny animal reel '{ep['title']}'. "
                  f"Story: {ep['concept']['story']}. The frame shows "
                  f"{studio.load_chars()['characters'][shot['cast'][0]]['name'] if shot.get('cast') else 'the squad'}"
                  f"{' saying: ' + dl0['line'] if dl0 else ''}. Return JSON keys: meme_text (max 8 words, "
                  f"relatable top-text meme), caption (1–2 lines, 1 emoji ok, tell people the full video is on the page), "
                  f"question, hashtags (6–8).")
        resp = studio.client().models.generate_content(
            model=studio.TEXT_MODEL, contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.9))
        meta.update({k: v for k, v in json.loads(resp.text).items() if v})
    except BaseException as e:
        log(f"  (caption fallback to episode metadata: {type(e).__name__})")
    im = Image.open(raw).convert("RGB").resize((1080, 1920)).crop((0, 285, 1080, 1635))  # 4:5 centre
    text = "".join(ch for ch in meta["meme_text"] if ord(ch) < 0x10000).strip().upper()
    if text:
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype(str(next(studio.FONT_DIR.glob("*.ttf"))), 76)
        words, rows, cur = text.split(), [], ""
        for w_ in words:
            if dr.textlength((cur + " " + w_).strip(), font=font) > 980:
                rows.append(cur)
                cur = w_
            else:
                cur = (cur + " " + w_).strip()
        rows.append(cur)
        y = 40
        for r in rows[:3]:
            w = dr.textbbox((0, 0), r, font=font, stroke_width=8)[2]
            dr.text(((1080 - w) // 2, y), r, font=font, fill="white", stroke_width=8, stroke_fill="black")
            y += 92
    out = day / "post.jpg"
    im.save(out, quality=92)
    (day / "post.json").write_text(json.dumps({"caption": meta["caption"], "question": meta["question"],
                                               "hashtags": meta["hashtags"]}, indent=2, ensure_ascii=False))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["kickoff", "status", "collect", "shots"])
    p.add_argument("--ep", help="episode id (kickoff/collect/shots)")
    p.add_argument("--job", help="job id (collect)")
    a = p.parse_args()
    if a.cmd == "kickoff":
        print(json.dumps(kickoff(a.ep)))
    elif a.cmd == "status":
        print(status())
    elif a.cmd == "shots":  # local dry run: shot list + voices, no Kaggle
        ep, d = studio.load_ep(a.ep)
        build_free_episode(ep, d, build_shots(ep))
    else:
        print("done" if collect(a.ep, a.job) else "still rendering")


if __name__ == "__main__":
    main()
