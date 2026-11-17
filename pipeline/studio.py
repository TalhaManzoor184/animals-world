#!/usr/bin/env python3
"""
Real Animal Comedy Shorts Studio: Gemini API production pipeline.

One source of truth per episode: episodes/<EP>/episode.json
Every step reads it, writes real files into the 15-folder delivery structure,
and records what actually exists in status.json. Nothing is ever marked done
unless the file is on disk. Nothing is ever published.

Usage (from repo root):
  python pipeline/studio.py plan   EP001          # composed prompts + cost estimate, no API calls
  python pipeline/studio.py docs   EP001          # write the 15 delivery folders (text parts)
  python pipeline/studio.py refs                  # one-time: clean full-body character sheets
  python pipeline/studio.py images EP001 [--scene 2] [--force]
  python pipeline/studio.py videos EP001 [--scene 2] [--force]
  python pipeline/studio.py voice  EP001          # TTS lines (tts mode) or extract Veo dialogue (veo mode)
  python pipeline/studio.py align  EP001          # optional: Gemini times the captions to the real audio
  python pipeline/studio.py edit   EP001          # final 1080x1920 MP4 with captions + music
  python pipeline/studio.py thumb  EP001
  python pipeline/studio.py qc     EP001          # quality-control checklist
  python pipeline/studio.py all    EP001          # docs -> images -> videos -> voice -> align -> edit -> thumb -> qc
  python pipeline/studio.py ideas                 # 10 new concepts + full episode.json for the best one
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAR_FILE = ROOT / "config" / "characters.json"
EPISODES = ROOT / "episodes"
FONT_DIR = ROOT / "assets" / "fonts"

# ---- Models & settings (override with env vars / GitHub secrets+vars) ----
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-flash-latest")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-3.1-flash-image")          # Nano Banana 2: multi-reference consistency
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "veo-3.1-fast-generate-preview")   # or veo-3.1-generate-preview for top quality
TTS_MODEL = os.getenv("TTS_MODEL", "gemini-2.5-flash-preview-tts")
VIDEO_MODE = os.getenv("VIDEO_MODE", "first_frame")     # first_frame | reference
DIALOGUE_MODE = os.getenv("DIALOGUE_MODE", "veo")       # veo (native lip-synced audio) | tts (Gemini TTS overlay)
VIDEO_RES = os.getenv("VIDEO_RES", "1080p")             # 720p | 1080p
PRICE_VIDEO_PER_S = float(os.getenv("PRICE_VIDEO_PER_S", "0.12"))   # check ai.google.dev/pricing
PRICE_IMAGE = float(os.getenv("PRICE_IMAGE", "0.07"))
VEO_TIMEOUT_S = int(os.getenv("VEO_TIMEOUT_S", "900"))

FOLDERS = [
    "01_Episode_Concept", "02_Full_Script", "03_Scene_Breakdown",
    "04_Character_Reference_Requirements", "05_Image_Generation_Prompts", "06_Image_Assets",
    "07_Video_Generation_Prompts", "08_Video_Assets", "09_Voice_Over", "10_Sound_Effects",
    "11_Final_Edited_Video", "12_Thumbnail", "13_YouTube_Metadata",
    "14_Instagram_Metadata", "15_Facebook_Metadata",
]


# ============================================================ helpers
def die(msg: str) -> None:
    print(f"\n✖ {msg}", file=sys.stderr)
    sys.exit(1)


def log(msg: str) -> None:
    print(msg, flush=True)


def load_chars() -> dict:
    return json.loads(CHAR_FILE.read_text(encoding="utf-8"))


def ep_dir(ep: str) -> Path:
    if (EPISODES / ep).is_dir():
        return EPISODES / ep
    hits = sorted(EPISODES.glob(f"{ep}_*"))
    if not hits:
        die(f"Episode folder for '{ep}' not found under {EPISODES}")
    return hits[0]


EPISODE_FILE = os.getenv("EPISODE_FILE", "episode.json")   # free mode edits from episode.free.json


def load_ep(ep: str) -> tuple[dict, Path]:
    d = ep_dir(ep)
    return json.loads((d / EPISODE_FILE).read_text(encoding="utf-8")), d


def sub(d: Path, folder: str) -> Path:
    p = d / folder
    p.mkdir(parents=True, exist_ok=True)
    return p


def update_status(d: Path, step: str, files: list[Path], note: str = "") -> None:
    sp = d / "status.json"
    st = json.loads(sp.read_text()) if sp.exists() else {}
    st[step] = {
        "at": dt.datetime.now().isoformat(timespec="seconds"),
        "files": [str(f.relative_to(d)) for f in files if f.exists()],
        "note": note,
    }
    sp.write_text(json.dumps(st, indent=2))


def run(cmd: list[str], cwd: Path | None = None) -> None:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        die("ffmpeg failed:\n" + " ".join(cmd) + "\n" + r.stderr[-3000:])


def probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
                        "-show_format", str(path)], capture_output=True, text=True)
    return json.loads(r.stdout or "{}")


def has_audio(path: Path) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe(path).get("streams", []))


def scene_name(n: int, ext: str) -> str:
    return f"scene_{n:02d}.{ext}"


def client():
    try:
        from google import genai
    except ImportError:
        die("google-genai not installed: pip install -r pipeline/requirements.txt")
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        die("GEMINI_API_KEY is not set (add it as a GitHub secret or export it locally).")
    return genai.Client(api_key=key)


def ref_path(ch: dict) -> Path:
    """Prefer the clean generated sheet; fall back to the board crop."""
    sheet = ROOT / ch["sheet"]
    return sheet if sheet.exists() else ROOT / ch["ref"]


# ============================================================ prompt composition
def compose_image_prompt(scene: dict, chars: dict) -> str:
    C, S, im = chars["characters"], chars["style"], scene["image"]
    lines = ["Create ONE photorealistic vertical 9:16 film still.", ""]
    lines.append("CHARACTERS (match the attached reference image for each one exactly: same species, "
                 "fur/feather colours, face, proportions, clothing and accessories):")
    for k in scene["cast"]:
        ch = C[k]
        lines.append(f"- {ch['name'].upper()}: {ch['look']}. Expression: {im['expressions'].get(k, 'natural')}.")
    lines += [
        "",
        f"ACTION / BODY POSITION: {im['action']}",
        f"ENVIRONMENT: {im['environment']}",
        f"LIGHTING: {im['lighting']}",
        f"CAMERA: {im['camera']}",
        f"COMPOSITION: {im['composition']}",
        "",
        f"QUALITY: {S['visual']}.",
        f"Each listed character appears exactly once. No other animals or people.",
        f"AVOID: {S['negative']}.",
    ]
    return "\n".join(lines)


def compose_video_prompt(scene: dict, chars: dict, mode: str = DIALOGUE_MODE) -> str:
    C, S, v = chars["characters"], chars["style"], scene["video"]
    cast = ", ".join(f"{C[k]['name']} ({C[k]['look']})" for k in scene["cast"])
    parts = [
        f"Photorealistic cinematic vertical 9:16 shot. Characters: {cast}.",
        f"Starting pose: {v['start']}.",
        f"Action: {v['motion']}.",
        f"Camera: {v['camera_move']}. One continuous shot, no cuts.",
        f"Environment motion: {v['environment_motion']}.",
        f"Ending pose: {v['end']}.",
    ]
    if mode == "veo" and scene.get("dialogue"):
        for dl in scene["dialogue"]:
            ch = C[dl["speaker"]]
            parts.append(f'{ch["name"]} the {ch["species"]} says ({dl["emotion"]}, '
                         f'{ch["voice"]["description"]}): "{dl["line"]}"')
        parts.append("Only the speaking animal moves its mouth while talking.")
    elif scene.get("dialogue"):
        names = ", ".join(C[d["speaker"]]["name"] for d in scene["dialogue"])
        parts.append(f"{names} move their mouths as if talking, but there is NO spoken dialogue in the audio.")
    parts.append(f"Sound effects: {v['sfx']}. No background music. No subtitles or on-screen text.")
    parts.append("Natural, physically believable animal movement. Faces, species, clothing and accessories "
                 "stay identical for the whole shot; no morphing.")
    return "\n".join(parts)


def video_negative(chars: dict) -> str:
    return ("melting face, morphing, species change, clothing change, missing sunglasses, extra limbs, "
            "duplicate animals, cartoon, subtitles, text, watermark, background music, " + chars["style"]["negative"])


# ============================================================ docs (15 folders)
def fmt_t(s: float) -> str:
    return f"{s:05.2f}s"


def timeline(ep: dict) -> list[tuple[dict, float]]:
    out, t = [], 0.0
    for sc in ep["scenes"]:
        out.append((sc, t))
        t += sc["use"][1] - sc["use"][0]
    return out


def total_duration(ep: dict) -> float:
    return sum(s["use"][1] - s["use"][0] for s in ep["scenes"])


def cmd_docs(args) -> None:
    ep, d = load_ep(args.ep)
    chars = load_chars()
    C = chars["characters"]
    w = lambda folder, name, text: (sub(d, folder) / name).write_text(text.strip() + "\n", encoding="utf-8")
    cast_all = [ep["main_character"]] + ep["supporting"]
    c = ep["concept"]

    w("01_Episode_Concept", "concept.md", f"""
# {ep['id']}: {ep['title']}
**Category:** {ep['category']}
**Main:** {C[ep['main_character']]['name']} | **Supporting:** {', '.join(C[k]['name'] for k in ep['supporting'])}
**Target duration:** {total_duration(ep):.1f}s (9:16, 1080x1920)

**Story:** {c['story']}

**Hook (0-3s):** {c['hook']}

**Main comedy moment:** {c['comedy_moment']}

**Punchline:** {c['punchline']}

**Viral potential:** {c['viral_notes']}
""")

    # 02 full script
    s = [f"# {ep['title']}: Full Script\n"]
    for sc, t0 in timeline(ep):
        a, b = sc["use"]
        s.append(f"## Scene {sc['n']}: {sc['beat']}  [{fmt_t(t0)} – {fmt_t(t0 + b - a)}]")
        s.append(f"*{sc['image']['action']}*\n")
        s.append(f"**Action:** {sc['video']['motion']}.\n")
        for dl in sc.get("dialogue", []):
            s.append(f"**{C[dl['speaker']]['name'].upper()}** *({dl['emotion']})*: \"{dl['line']}\"  "
                     f"`@{fmt_t(t0 + dl['at'][0] - a)}`")
        s.append(f"\n**SFX:** {sc['video']['sfx']}  \n**Camera:** {sc['image']['camera']}; {sc['video']['camera_move']}\n")
    s.append(f"**Music:** {ep['music']['brief']}")
    w("02_Full_Script", "script.md", "\n".join(s))

    # 03 breakdown
    rows = ["| Scene | Timeline | Clip trim | Cast | Beat | Expressions |", "|---|---|---|---|---|---|"]
    for sc, t0 in timeline(ep):
        a, b = sc["use"]
        ex = "; ".join(f"{C[k]['name']}: {v}" for k, v in sc["image"]["expressions"].items())
        rows.append(f"| {sc['n']} | {fmt_t(t0)}–{fmt_t(t0 + b - a)} | {a}–{b}s of {sc['clip_seconds']}s | "
                    f"{', '.join(C[k]['name'] for k in sc['cast'])} | {sc['beat']} | {ex} |")
    w("03_Scene_Breakdown", "breakdown.md", "# Scene Breakdown\n\n" + "\n".join(rows) +
      f"\n\n**Edit notes:** {json.dumps(ep['edit'], indent=1)}")

    # 04 character refs
    r = ["# Character Reference Requirements\n",
         "Every image call attaches the reference for each character in the shot. "
         "Clean full-body sheets (`*_sheet.png`) are used when present, otherwise the board crops.\n"]
    for k in cast_all:
        ch = C[k]
        r.append(f"## {ch['name']} ({ch['role']})\n- Look (locked): {ch['look']}\n- Personality: {ch['personality']}\n"
                 f"- Voice: {ch['voice']['description']} (TTS voice: {ch['voice']['tts_voice']})\n"
                 f"- Reference in use: `{ref_path(ch).relative_to(ROOT)}`\n")
    w("04_Character_Reference_Requirements", "characters.md", "\n".join(r))

    # 05 / 07 prompts
    w("05_Image_Generation_Prompts", "image_prompts.md", "\n\n".join(
        f"## Scene {sc['n']}: refs: {', '.join(C[k]['name'] for k in sc['cast'])}\n```\n{compose_image_prompt(sc, chars)}\n```"
        for sc in ep["scenes"]))
    w("07_Video_Generation_Prompts", "video_prompts.md", "\n\n".join(
        f"## Scene {sc['n']}: {sc['clip_seconds']}s, image-to-video from scene_{sc['n']:02d}.png\n```\n"
        f"{compose_video_prompt(sc, chars)}\n```\nNegative: `{video_negative(chars)}`" for sc in ep["scenes"]))

    # 09 voice script, 10 sfx
    v = ["# Voice-Over Script\n", f"Mode: **{DIALOGUE_MODE}** (veo = lip-synced audio generated inside each clip; "
         "tts = Gemini TTS lines mixed in at these times)\n", "| Time | Speaker | Line | Emotion | Voice |", "|---|---|---|---|---|"]
    for sc, t0 in timeline(ep):
        for dl in sc.get("dialogue", []):
            ch = C[dl["speaker"]]
            v.append(f"| {fmt_t(t0 + dl['at'][0] - sc['use'][0])} | {ch['name']} | {dl['line']} | {dl['emotion']} | "
                     f"{ch['voice']['description']} ({ch['voice']['tts_voice']}) |")
    w("09_Voice_Over", "voice_script.md", "\n".join(v))
    w("10_Sound_Effects", "sfx_and_music.md", "# Sound Effects & Music\n\n" + "\n".join(
        f"- **Scene {sc['n']}:** {sc['video']['sfx']}" for sc in ep["scenes"]) +
      f"\n\n**Music:** {ep['music']['brief']}\n\nFile: `{ep['music']['file']}` at {ep['music']['volume_db']} dB. "
      f"{ep['music']['source_rule']}")

    # 13-15 metadata
    yt, ig, fb = ep["metadata"]["youtube"], ep["metadata"]["instagram"], ep["metadata"]["facebook"]
    w("13_YouTube_Metadata", "youtube.md", f"""
# YouTube Shorts
**Title:** {yt['title']}

**Description:**
{yt['description']}

**Hashtags:** {' '.join(yt['hashtags'])}

**Tags:** {', '.join(yt['tags'])}

**Pinned comment:** {yt['pinned_comment']}

**Thumbnail concept:** {yt['thumbnail_concept']}

**Disclosure:** {ep.get('ai_disclosure', '')}
""")
    w("14_Instagram_Metadata", "instagram.md",
      f"# Instagram Reel\n{ig['caption']}\n\n{ig['engagement_question']}\n\n{' '.join(ig['hashtags'])}")
    w("15_Facebook_Metadata", "facebook.md",
      f"# Facebook Reel\n{fb['caption']}\n\n{fb['engagement_question']}\n\n{' '.join(fb['hashtags'])}")
    write_asset_readmes(ep, d)
    log(f"✔ docs written to {d.relative_to(ROOT)}")


def write_asset_readmes(ep: dict, d: Path) -> None:
    """Asset folders list ONLY files that exist on disk."""
    hint = {"06_Image_Assets": "images", "08_Video_Assets": "videos", "09_Voice_Over": "voice",
            "11_Final_Edited_Video": "edit", "12_Thumbnail": "thumb"}
    for folder, step in hint.items():
        p = sub(d, folder)
        files = sorted(f for f in p.iterdir() if f.is_file() and not f.name.endswith(".md") and f.name != "README.txt")
        if files:
            body = "Generated files:\n" + "\n".join(f"- {f.name} ({f.stat().st_size // 1024} KB)" for f in files)
        else:
            body = f"NOT GENERATED YET.\nRun: python pipeline/studio.py {step} {ep['id']}"
        (p / "README.txt").write_text(body + "\n")


# ============================================================ plan (dry run)
def cmd_plan(args) -> None:
    ep, d = load_ep(args.ep)
    chars = load_chars()
    for sc in ep["scenes"]:
        log(f"\n===== SCENE {sc['n']}: IMAGE PROMPT =====\n{compose_image_prompt(sc, chars)}")
        log(f"\n----- SCENE {sc['n']}: VIDEO PROMPT ({VIDEO_MODEL}, {VIDEO_MODE}) -----\n{compose_video_prompt(sc, chars)}")
    n_img = len(ep["scenes"]) + 1
    vid_s = sum(s["clip_seconds"] for s in ep["scenes"])
    n_tts = sum(len(s.get("dialogue", [])) for s in ep["scenes"]) if DIALOGUE_MODE == "tts" else 0
    log(f"\nFinal runtime: {total_duration(ep):.1f}s")
    log(f"Estimated cost: {n_img} images × ${PRICE_IMAGE} + {vid_s}s video × ${PRICE_VIDEO_PER_S}/s "
        f"= ~${n_img * PRICE_IMAGE + vid_s * PRICE_VIDEO_PER_S:.2f}"
        + (f" (+{n_tts} short TTS calls)" if n_tts else "") + "  (re-rolls cost extra; check current pricing)")


# ============================================================ image generation
def gen_image(c, prompt: str, refs: list[tuple[str, Path]], out: Path, aspect: str = "9:16") -> None:
    from google.genai import types
    from PIL import Image

    contents: list = []
    for label, p in refs:
        contents += [f"Reference image for {label}:", Image.open(p)]
    contents.append(prompt)
    cfg = types.GenerateContentConfig(response_modalities=["IMAGE"],
                                      image_config=types.ImageConfig(aspect_ratio=aspect))
    for attempt in range(3):
        resp = c.models.generate_content(model=IMAGE_MODEL, contents=contents, config=cfg)
        for cand in resp.candidates or []:
            for part in (cand.content.parts if cand.content else None) or []:
                if part.inline_data and part.inline_data.data:
                    Image.open(io.BytesIO(part.inline_data.data)).convert("RGB").save(out)
                    return
        reason = getattr(resp, "prompt_feedback", None) or [getattr(x, "finish_reason", None) for x in resp.candidates or []]
        log(f"  ! no image returned (attempt {attempt + 1}): {reason}")
        time.sleep(5)
    die(f"Image generation failed for {out.name}; prompt may be blocked. Edit the scene in episode.json and retry.")


def cmd_refs(args) -> None:
    chars = load_chars()
    c = client()
    made = []
    for k, ch in chars["characters"].items():
        out = ROOT / ch["sheet"]
        if out.exists() and not args.force:
            log(f"• {ch['name']}: sheet exists, skipping")
            continue
        prompt = (f"Using the attached reference, create a clean photorealistic FULL-BODY character reference photo of "
                  f"this exact character: {ch['look']}. Standing naturally, front three-quarter view, whole body and "
                  f"all accessories visible, neutral friendly expression, plain light-grey studio background, soft even "
                  f"studio lighting. Keep species, colours, face, clothing and accessories identical to the reference. "
                  f"No text. Avoid: {chars['style']['negative']}.")
        log(f"→ {ch['name']}: generating reference sheet")
        gen_image(c, prompt, [(ch["name"], ROOT / ch["ref"])], out, aspect="3:4")
        made.append(out)
    log(f"✔ {len(made)} sheet(s) generated. Review them in references/characters/ and delete any you dislike, then rerun.")


def scenes_for(ep: dict, only: int | None) -> list[dict]:
    return [s for s in ep["scenes"] if only is None or s["n"] == only]


def cmd_images(args) -> None:
    ep, d = load_ep(args.ep)
    chars = load_chars()
    C = chars["characters"]
    c = client()
    out_dir = sub(d, "06_Image_Assets")
    made = []
    for sc in scenes_for(ep, args.scene):
        out = out_dir / scene_name(sc["n"], "png")
        if out.exists() and not args.force:
            log(f"• scene {sc['n']}: exists, skipping (use --force to redo)")
            continue
        log(f"→ scene {sc['n']}: generating first-frame image")
        gen_image(c, compose_image_prompt(sc, chars), [(C[k]["name"], ref_path(C[k])) for k in sc["cast"]], out)
        made.append(out)
    update_status(d, "images", sorted(out_dir.glob("scene_*.png")))
    write_asset_readmes(ep, d)
    log(f"✔ {len(made)} image(s) generated → {out_dir.relative_to(ROOT)}  (REVIEW before running videos)")


# ============================================================ video generation (Veo)
def cmd_videos(args) -> None:
    from google.genai import types

    ep, d = load_ep(args.ep)
    chars = load_chars()
    C = chars["characters"]
    c = client()
    img_dir, out_dir = sub(d, "06_Image_Assets"), sub(d, "08_Video_Assets")
    made = []
    for sc in scenes_for(ep, args.scene):
        out = out_dir / scene_name(sc["n"], "mp4")
        if out.exists() and not args.force:
            log(f"• scene {sc['n']}: exists, skipping")
            continue
        cfg = dict(aspect_ratio="9:16", resolution=VIDEO_RES, duration_seconds=sc["clip_seconds"],
                   negative_prompt=video_negative(chars), number_of_videos=1)
        kwargs = dict(model=VIDEO_MODEL, prompt=compose_video_prompt(sc, chars))
        if VIDEO_MODE == "reference":
            refs = [types.VideoGenerationReferenceImage(image=types.Image.from_file(location=str(ref_path(C[k]))),
                                                        reference_type="asset") for k in sc["cast"][:3]]
            cfg["reference_images"] = refs
        else:
            first = img_dir / scene_name(sc["n"], "png")
            if not first.exists():
                die(f"Missing first frame {first.relative_to(ROOT)}; run `images` first.")
            kwargs["image"] = types.Image.from_file(location=str(first))
        log(f"→ scene {sc['n']}: Veo request ({VIDEO_MODEL}, {VIDEO_RES}, {sc['clip_seconds']}s)")
        op = c.models.generate_videos(config=types.GenerateVideosConfig(**cfg), **kwargs)
        t0 = time.time()
        while not op.done:
            if time.time() - t0 > VEO_TIMEOUT_S:
                die(f"Veo timed out on scene {sc['n']} (operation {op.name}).")
            time.sleep(15)
            op = c.operations.get(op)
        if getattr(op, "error", None):
            die(f"Veo error on scene {sc['n']}: {op.error}")
        resp = op.response or getattr(op, "result", None)
        vids = getattr(resp, "generated_videos", None) or []
        if not vids:
            reasons = getattr(resp, "rai_media_filtered_reasons", None)
            die(f"Veo returned no video for scene {sc['n']} (filtered: {reasons}). Soften the prompt and retry.")
        c.files.download(file=vids[0].video)
        vids[0].video.save(str(out))
        log(f"  ✔ saved {out.name} ({int(time.time() - t0)}s)")
        made.append(out)
    update_status(d, "videos", sorted(out_dir.glob("scene_*.mp4")), f"model={VIDEO_MODEL} mode={VIDEO_MODE}")
    write_asset_readmes(ep, d)
    log(f"✔ {len(made)} clip(s) generated. Watch them all before editing.")


# ============================================================ voice
def cmd_voice(args) -> None:
    ep, d = load_ep(args.ep)
    C = load_chars()["characters"]
    out_dir = sub(d, "09_Voice_Over")
    made = []
    if DIALOGUE_MODE == "tts":
        from google.genai import types
        c = client()
        for sc in ep["scenes"]:
            for i, dl in enumerate(sc.get("dialogue", []), 1):
                out = out_dir / f"scene_{sc['n']:02d}_line_{i:02d}_{dl['speaker']}.wav"
                if out.exists() and not args.force:
                    continue
                ch = C[dl["speaker"]]
                resp = c.models.generate_content(
                    model=TTS_MODEL,
                    contents=f"Say this {dl['emotion']}, as a {ch['voice']['description']}: {dl['line']}",
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=ch["voice"]["tts_voice"])))))
                pcm = resp.candidates[0].content.parts[0].inline_data.data
                with wave.open(str(out), "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000); wf.writeframes(pcm)
                log(f"  ✔ {out.name}")
                made.append(out)
    else:
        vid_dir = d / "08_Video_Assets"
        for sc in ep["scenes"]:
            src = vid_dir / scene_name(sc["n"], "mp4")
            if not src.exists():
                log(f"• scene {sc['n']}: no clip yet, skipping")
                continue
            if not has_audio(src):
                log(f"• scene {sc['n']}: clip has no audio track")
                continue
            out = out_dir / scene_name(sc["n"], "wav")
            run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "48000", str(out)])
            made.append(out)
        log("  (veo mode: dialogue + SFX live inside each clip; extracted for review/re-use)")
    update_status(d, "voice", sorted(out_dir.glob("*.wav")), f"mode={DIALOGUE_MODE}")
    write_asset_readmes(ep, d)
    log(f"✔ voice step: {len(made)} file(s)")


# ============================================================ caption alignment (optional)
def cmd_align(args) -> None:
    from google.genai import types

    ep, d = load_ep(args.ep)
    c = client()
    vid_dir = d / "08_Video_Assets"
    result = {}
    for sc in ep["scenes"]:
        src = vid_dir / scene_name(sc["n"], "mp4")
        if not sc.get("dialogue") or not src.exists() or DIALOGUE_MODE != "veo":
            continue
        lines = [dl["line"] for dl in sc["dialogue"]]
        prompt = ("This clip contains these spoken lines, in order:\n" +
                  "\n".join(f"{i}: {l}" for i, l in enumerate(lines)) +
                  "\nReturn JSON: a list of {\"i\": int, \"start\": seconds, \"end\": seconds} giving when each line is "
                  "actually heard. If a line is not spoken at all, use start=-1.")
        resp = c.models.generate_content(
            model=TEXT_MODEL,
            contents=[types.Part.from_bytes(data=src.read_bytes(), mime_type="video/mp4"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"))
        try:
            items = json.loads(resp.text)
            result[str(sc["n"])] = [{"i": int(x["i"]), "start": float(x["start"]), "end": float(x["end"])} for x in items]
            missing = [lines[x["i"]] for x in result[str(sc["n"])] if x["start"] < 0]
            log(f"  scene {sc['n']}: aligned" + (f"  ⚠ NOT SPOKEN: {missing}" if missing else ""))
        except Exception as e:  # keep script timings if the model returns junk
            log(f"  scene {sc['n']}: alignment unusable ({e}); script timings will be used")
    out = sub(d, "09_Voice_Over") / "caption_timings.json"
    out.write_text(json.dumps(result, indent=2))
    log(f"✔ wrote {out.relative_to(ROOT)}")


# ============================================================ edit
SPEAKER_COLOURS = {"milo": "&H45A5FF&", "bruno": "&HFFB040&", "rocky": "&HD0D0D0&",
                   "ducky": "&H00D7FF&", "coco": "&HB07CFF&", "chiku": "&H3C3CFF&"}  # ASS override = &HBBGGRR&


def ass_time(t: float) -> str:
    t = max(t, 0)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def caption_events(ep: dict, d: Path) -> list[tuple[float, float, str, str, bool]]:
    """(start, end, speaker, text, highlight) on the final timeline, chunked ≤4 words."""
    aligned = {}
    ct = d / "09_Voice_Over" / "caption_timings.json"
    if ct.exists():
        aligned = json.loads(ct.read_text())
    ev = []
    for sc, t0 in timeline(ep):
        a, b = sc["use"]
        for i, dl in enumerate(sc.get("dialogue", [])):
            s, e = dl["at"]
            for x in aligned.get(str(sc["n"]), []):
                if x["i"] == i and x["start"] >= 0:
                    s, e = x["start"], max(x["end"], x["start"] + 0.4)
            s, e = max(s, a), min(e, b)
            if e <= s:
                continue
            words = dl["line"].split()
            n = -(-len(words) // 4)  # balanced chunks of at most 4 words
            size = -(-len(words) // n)
            chunks = [words[j:j + size] for j in range(0, len(words), size)]
            span = (e - s) / len(chunks)
            for j, ch in enumerate(chunks):
                cs = t0 + (s - a) + j * span
                ev.append((cs, cs + span, dl["speaker"], " ".join(ch), bool(dl.get("highlight"))))
    return ev


def write_ass(ep: dict, d: Path, path: Path) -> None:
    C = load_chars()["characters"]
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Poppins,82,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,2,2,80,80,640,1
Style: Who,Poppins,46,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,1,0,1,5,0,2,80,80,760,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for s, e, spk, txt, hi in caption_events(ep, d):
        col = "{\\c&H00E5FF&}" if hi else ""
        txt = txt.replace("{", "(").replace("}", ")")
        pop = "{\\fscx85\\fscy85\\t(0,90,\\fscx100\\fscy100)}"
        lines.append(f"Dialogue: 1,{ass_time(s)},{ass_time(e)},Cap,,0,0,0,,{pop}{col}{txt}")
        lines.append(f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Who,,0,0,0,,{{\\c{SPEAKER_COLOURS.get(spk, '&HFFFFFF&')}}}"
                     f"{C[spk]['name'].upper()}")
    path.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")


def segment_plan(sc: dict) -> list[tuple[float, float, float]]:
    """Split the used range into (start, end, zoom) pieces for punch-ins."""
    a, b = sc["use"]
    cuts = {a, b}
    for s, e, _ in sc.get("punch", []):
        cuts |= {min(max(s, a), b), min(max(e, a), b)}
    pts = sorted(cuts)
    segs = []
    for s, e in zip(pts, pts[1:]):
        if e - s < 0.04:
            continue
        z = next((zz for ps, pe, zz in sc.get("punch", []) if ps <= s and e <= pe), 1.0)
        segs.append((s, e, z))
    return segs


def cmd_edit(args) -> None:
    ep, d = load_ep(args.ep)
    vid_dir, out_dir = d / "08_Video_Assets", sub(d, "11_Final_Edited_Video")
    missing = [scene_name(s["n"], "mp4") for s in ep["scenes"] if not (vid_dir / scene_name(s["n"], "mp4")).exists()]
    if missing:
        die(f"Cannot edit; missing clips in 08_Video_Assets: {missing}")
    ex = ep["edit"]["export"]
    W, H, FPS = ex["width"], ex["height"], ex["fps"]
    tmp = Path(tempfile.mkdtemp(prefix="edit_"))
    pieces = []
    for sc in ep["scenes"]:
        src = vid_dir / scene_name(sc["n"], "mp4")
        audio = has_audio(src)
        for k, (s, e, z) in enumerate(segment_plan(sc)):
            vf = [f"scale={W}:{H}:force_original_aspect_ratio=increase", f"crop={W}:{H}"]
            if z > 1.0:
                vf += [f"crop=iw/{z}:ih/{z}", f"scale={W}:{H}"]
            if sc.get("flash_in") and k == 0:
                vf.append("fade=t=in:st=0:d=0.14:color=white")
            vf += [f"fps={FPS}", "setsar=1", "format=yuv420p"]
            piece = tmp / f"p_{sc['n']:02d}_{k:02d}.mp4"
            cmd = ["ffmpeg", "-y", "-ss", f"{s:.3f}", "-t", f"{e - s:.3f}", "-i", str(src)]
            if not audio:
                cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            cmd += ["-vf", ",".join(vf), "-map", "0:v:0", "-map", "0:a:0" if audio else "1:a:0",
                    "-af", "aresample=48000,aformat=channel_layouts=stereo", "-shortest",
                    "-t", f"{e - s:.3f}",  # hard cap: silent-audio pieces otherwise run long and captions drift
                    "-c:v", "libx264", "-crf", "14", "-preset", "veryfast", "-c:a", "pcm_s16le", str(piece)]
            run(cmd)
            pieces.append(piece)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in pieces))
    joined = tmp / "joined.mkv"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", joined.name], cwd=tmp)

    dur = total_duration(ep)
    write_ass(ep, d, tmp / "captions.ass")
    shutil.copy(tmp / "captions.ass", out_dir / f"{ep['id']}_captions.ass")

    inputs = ["-i", joined.name]
    amix = ["[0:a]volume=1.0[base]"]
    labels = ["[base]"]
    idx = 1
    # TTS dialogue overlay
    if DIALOGUE_MODE == "tts":
        for sc, t0 in timeline(ep):
            for i, dl in enumerate(sc.get("dialogue", []), 1):
                wav = d / "09_Voice_Over" / f"scene_{sc['n']:02d}_line_{i:02d}_{dl['speaker']}.wav"
                if not wav.exists():
                    die(f"Missing TTS line {wav.name}; run `voice` first.")
                ms = int((t0 + dl["at"][0] - sc["use"][0]) * 1000)
                inputs += ["-i", str(wav)]
                amix.append(f"[{idx}:a]aresample=48000,adelay={ms}|{ms},volume=1.6[v{idx}]")
                labels.append(f"[v{idx}]")
                idx += 1
    # Music bed with duck before the punchline
    music = ROOT / ep["music"]["file"] if ep.get("music", {}).get("file") else None
    if music and music.is_file():
        duck = ""
        hl = [(s, e) for s, e, _, _, h in caption_events(ep, d) if h]
        if hl:
            p0 = hl[0][0]
            duck = f",volume='if(between(t,{p0 - 0.5:.2f},{p0 + 1.5:.2f}),0.05,1)':eval=frame"
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        amix.append(f"[{idx}:a]aresample=48000,volume={ep['music']['volume_db']}dB{duck},"
                    f"afade=t=out:st={dur - 1.2:.2f}:d=1.2[mus]")
        labels.append("[mus]")
        idx += 1
    else:
        log(f"  (no music file at {ep['music']['file']}; exporting without music)")
    fc = ";".join(amix) + f";{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=first," \
                          f"loudnorm=I=-14:TP=-1.5:LRA=11[aout]"
    fc += f";[0:v]subtitles=captions.ass:fontsdir='{FONT_DIR}'[vout]"
    final = out_dir / f"{ep['id']}_{ep['slug']}_final.mp4"
    run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[vout]", "-map", "[aout]", "-t", f"{dur:.3f}",
         "-c:v", ex["vcodec"], "-crf", str(ex["crf"]), "-preset", "medium", "-r", str(FPS), "-pix_fmt", "yuv420p",
         "-c:a", ex["acodec"], "-b:a", ex["abitrate"], "-ar", "48000", "-movflags", "+faststart", str(final)], cwd=tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    update_status(d, "edit", [final])
    write_asset_readmes(ep, d)
    log(f"✔ final video: {final.relative_to(ROOT)} ({dur:.1f}s)")


# ============================================================ thumbnail
def cmd_thumb(args) -> None:
    from PIL import Image, ImageDraw, ImageFont

    ep, d = load_ep(args.ep)
    chars = load_chars()
    C = chars["characters"]
    th = ep["thumbnail"]
    out_dir = sub(d, "12_Thumbnail")
    raw = out_dir / "thumbnail_raw.png"
    if not raw.exists() or args.force:
        c = client()
        prompt = (f"{th['prompt']}\nCharacters (match references exactly): " +
                  "; ".join(f"{C[k]['name']}: {C[k]['look']}" for k in th["cast"]) +
                  f".\nLeave the top quarter of the frame as simple sky for a title. {chars['style']['visual']}. "
                  f"Avoid: {chars['style']['negative']}.")
        gen_image(c, prompt, [(C[k]["name"], ref_path(C[k])) for k in th["cast"]], raw)
    im = Image.open(raw).convert("RGB").resize((1080, 1920))
    text = re.sub(r"[^\u0000-￿]", "", th["overlay_text"]).strip().upper()  # strip emoji the font can't draw
    if text:
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype(str(next(FONT_DIR.glob("*.ttf"))), 150)
        bw = dr.textbbox((0, 0), text, font=font, stroke_width=12)
        x = (1080 - (bw[2] - bw[0])) // 2
        dr.text((x, 180), text, font=font, fill=(255, 229, 0), stroke_width=12, stroke_fill=(0, 0, 0))
    final = out_dir / f"{ep['id']}_thumbnail.jpg"
    im.save(final, quality=92)
    update_status(d, "thumb", [raw, final])
    write_asset_readmes(ep, d)
    log(f"✔ thumbnail: {final.relative_to(ROOT)}")


# ============================================================ QC
def cmd_qc(args) -> None:
    ep, d = load_ep(args.ep)
    final = next((d / "11_Final_Edited_Video").glob("*_final.mp4"), None) if (d / "11_Final_Edited_Video").exists() else None
    auto = []
    if final:
        info = probe(final)
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        dur = float(info["format"]["duration"])
        num, den = map(int, v["r_frame_rate"].split("/"))
        auto += [
            ("Video is vertical 1080x1920", v["width"] == 1080 and v["height"] == 1920),
            ("H.264 @ 30fps", v["codec_name"] == "h264" and round(num / den) == 30),
            (f"Duration 15–35s ({dur:.1f}s)", 15 <= dur <= 35),
            ("Has audio track", any(s["codec_type"] == "audio" for s in info["streams"])),
        ]
    else:
        auto.append(("Final video exists", False))
    auto += [
        ("Captions file exists", any((d / "11_Final_Edited_Video").glob("*.ass")) if (d / "11_Final_Edited_Video").exists() else False),
        ("Thumbnail exists", (d / "12_Thumbnail" / f"{ep['id']}_thumbnail.jpg").exists()),
        (f"YouTube title ≤100 chars ({len(ep['metadata']['youtube']['title'])})", len(ep["metadata"]["youtube"]["title"]) <= 100),
        ("Music file is from assets/music (royalty-free)", bool(ep["music"].get("file")) and (ROOT / ep["music"]["file"]).is_file()),
    ]
    manual = ["Opening 3 seconds hook works on mute", "Comedy is understandable", "Characters consistent across scenes",
              "Accessories correct in every shot", "Dialogue matches mouth movement", "Audio clear, no garbled words",
              "Captions readable and not under the Shorts UI", "Ending lands the punchline",
              "No animal shown harmed, frightened or in distress", "AI-content disclosure ticked on upload"]
    lines = ["# QC Report: " + ep["id"], "", "## Automated"]
    lines += [f"- [{'x' if ok else ' '}] {name}" + ("" if ok else "  ← FIX") for name, ok in auto]
    lines += ["", "## Manual review (you)"] + [f"- [ ] {m}" for m in manual]
    lines += ["", "Publishing is manual: nothing is uploaded by this pipeline."]
    (d / "QC_REPORT.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))


# ============================================================ ideas → new episode
def cmd_ideas(args) -> None:
    from google.genai import types

    chars = load_chars()
    hist_p = EPISODES / "history.json"
    history = json.loads(hist_p.read_text()) if hist_p.exists() else []
    example = json.loads(ep_dir("EP001").joinpath("episode.json").read_text())
    next_id = f"EP{len(history) + 1:03d}"
    FREE_HINT = ("FORMAT NOTE (free production mode): every spoken line is shown as a close-up of that ONE character "
                 "talking to camera, and each scene may add one short action shot with no dialogue. So write the comedy "
                 "as punchy one-liners and reactions (max 6 lines total, each under 10 words), with the physical gag in "
                 "the action shots.\n") if os.getenv("PRODUCTION_MODE", "paid") == "free" else ""
    prompt = f"""You are the head writer of a photorealistic talking-animal comedy Shorts channel (English, US/UK/CA audience).
CHARACTERS (never change them): {json.dumps({k: {x: v[x] for x in ('name', 'species', 'personality', 'role')} for k, v in chars['characters'].items()})}
ALREADY MADE (do not repeat premises or punchlines): {json.dumps([{'title': h['title'], 'concept': h['concept']} for h in history])}
CATEGORIES: friendships, unexpected reactions, food stealing, misunderstandings, pranks, crazy adventures, competitions, family situations, plot twists, relatable everyday comedy.
{FREE_HINT}RULES: 15–35s total; 3–5 scenes; each scene is ONE continuous 8-second AI video shot (no cuts inside) with at most 3 characters;
simple, short, funny English dialogue; visual comedy first; strong first-3-second hook; harmless fictional situations only:
no animal ever harmed, frightened, in danger or distress; no brands, no real people, no copyrighted music.

TASK: Invent 10 original concepts, score each 1–25 (hook, entertainment, replay, shareability, simplicity), select the best,
and write its full production episode in EXACTLY the same JSON schema as this example (same keys, use character keys like "milo"):
{json.dumps(example)}

Return JSON: {{"concepts": [{{"title","main","support":[],"category","concept","hook","comedy","punchline","duration_s","score","viral_note"}} x10],
"selected_index": int, "episode": {{...full schema with "id": "{next_id}"...}}}}"""
    c = client()
    resp = c.models.generate_content(model=TEXT_MODEL, contents=prompt,
                                     config=types.GenerateContentConfig(response_mime_type="application/json",
                                                                        temperature=1.0))
    data = json.loads(resp.text)
    epi = data["episode"]
    epi["id"] = next_id
    # validate
    keys = set(chars["characters"])
    for sc in epi["scenes"]:
        for k in ("n", "beat", "cast", "clip_seconds", "use", "image", "video"):
            if k not in sc:
                die(f"Generated episode scene missing '{k}'; rerun `ideas`.")
        bad = [x for x in sc["cast"] + [dl["speaker"] for dl in sc.get("dialogue", [])] if x not in keys]
        if bad:
            die(f"Generated episode uses unknown characters {bad}; rerun `ideas`.")
    if not 15 <= total_duration(epi) <= 35:
        log(f"  ⚠ runtime {total_duration(epi):.1f}s is outside 15–35s; adjust 'use' trims before producing.")
    tracks = sorted((ROOT / "assets" / "music").glob("*.mp3"))
    epi.setdefault("music", {"brief": "", "volume_db": -20, "source_rule": "royalty-free only"})
    epi["music"]["file"] = str(random.choice(tracks).relative_to(ROOT)) if tracks else ""
    epi["music"].setdefault("volume_db", -20)
    slug = re.sub(r"[^a-z0-9]+", "_", epi.get("slug") or epi["title"].lower()).strip("_")[:40]
    epi["slug"] = slug
    d = EPISODES / f"{next_id}_{slug}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "episode.json").write_text(json.dumps(epi, indent=2, ensure_ascii=False))
    md = [f"# Concepts for {next_id}\n", "| # | Title | Lead | Score | Concept |", "|---|---|---|---|---|"]
    for i, cc in enumerate(data["concepts"]):
        star = " ⭐" if i == data.get("selected_index") else ""
        md.append(f"| {i + 1} | {cc.get('title')}{star} | {cc.get('main')} | {cc.get('score')} | {cc.get('concept')} |")
    (d / "00_concepts.md").write_text("\n".join(md) + "\n")
    history.append({"id": next_id, "title": epi["title"], "concept": epi["concept"]["story"],
                    "cast": sorted({k for s in epi["scenes"] for k in s["cast"]})})
    hist_p.write_text(json.dumps(history, indent=2))
    log(f"✔ new episode {d.relative_to(ROOT)}: \"{epi['title']}\". Review episode.json, then run `docs` and `images`.")
    args.ep = next_id
    cmd_docs(args)


# ============================================================ all
def cmd_all(args) -> None:
    for step in (cmd_docs, cmd_images, cmd_videos, cmd_voice, cmd_align, cmd_edit, cmd_thumb, cmd_qc):
        log(f"\n######## {step.__name__[4:].upper()} ########")
        step(args)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["plan", "docs", "refs", "images", "videos", "voice", "align",
                                       "edit", "thumb", "qc", "all", "ideas"])
    p.add_argument("ep", nargs="?", help="episode id, e.g. EP001")
    p.add_argument("--scene", type=int, help="only this scene number")
    p.add_argument("--force", action="store_true", help="regenerate even if the file exists")
    a = p.parse_args()
    if a.command not in ("refs", "ideas") and not a.ep:
        p.error("this command needs an episode id, e.g. EP001")
    globals()[f"cmd_{a.command}"](a)


if __name__ == "__main__":
    main()
