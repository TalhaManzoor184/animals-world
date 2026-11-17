#!/usr/bin/env python3
"""
Daily autopilot: 1 image post, 1 story, 1 reel per day.

  python pipeline/daily.py post    # Gemini writes a gag → Nano Banana image (4:5) → Facebook photo + Instagram post
  python pipeline/daily.py reel    # new episode (ideas) → produce → QC gate → YouTube Shorts + Instagram Reel + Facebook Reel
  python pipeline/daily.py story   # 9:16 teaser (today's reel thumbnail, else today's post) → Facebook + Instagram story
  python pipeline/daily.py all
  python pipeline/daily.py kickoff # FREE mode (PRODUCTION_MODE=free): start today's Kaggle GPU render first

Safe to re-run: today's state lives in daily/<YYYY-MM-DD>/, and anything already published is skipped.
AUTO_PUBLISH=false → review mode: YouTube uploads as private, Instagram/Facebook are skipped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import studio  # noqa: E402
import publish  # noqa: E402

ROOT = studio.ROOT
TODAY = os.getenv("DAILY_DATE") or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
DAY = ROOT / "daily" / TODAY
AUTO_PUBLISH = os.getenv("AUTO_PUBLISH", "true").lower() == "true"
FREE = os.getenv("PRODUCTION_MODE", "paid").lower() == "free"


def ns(**kw):
    return argparse.Namespace(scene=None, force=False, **kw)


def state() -> dict:
    p = DAY / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(**kw) -> None:
    DAY.mkdir(parents=True, exist_ok=True)
    st = {**state(), **kw}
    (DAY / "state.json").write_text(json.dumps(st, indent=2))


def ledger() -> publish.Ledger:
    return publish.Ledger(DAY / "published.json")


def step(L: publish.Ledger, key: str, fn, *a, **kw):
    """Run one publish action once per day; record its result."""
    if L.done(key):
        print(f"  • {key}: already published today, skipping")
        return L.data[key].get("result")
    res = fn(*a, **kw)
    if res is not None:
        L.mark(key, {"result": res})
    return res


# ============================================================ image post
def make_post() -> Path:
    out = DAY / "post.jpg"
    meta_p = DAY / "post.json"
    if out.exists() and meta_p.exists():
        return out
    from google.genai import types
    from PIL import Image

    chars = studio.load_chars()
    hist_p = ROOT / "daily" / "posts_history.json"
    hist = json.loads(hist_p.read_text()) if hist_p.exists() else []
    prompt = f"""You write daily single-image meme posts for a photorealistic talking-animal comedy page (English, US/UK/CA).
CHARACTERS: {json.dumps({k: {x: v[x] for x in ('name', 'species', 'personality')} for k, v in chars['characters'].items()})}
RECENT POSTS (do not repeat): {json.dumps(hist[-40:])}
Make ONE relatable, instantly-funny photo gag (like a meme still) with 1–3 characters. Harmless, no brands, no real people,
no animal in danger or distress. Return JSON with keys:
"cast" (list of character keys), "action", "expressions" (object: key → expression), "environment", "lighting", "camera",
"composition", "caption" (1–2 short punchy lines, may include 1–2 emoji), "question" (an engagement question),
"hashtags" (6–10 relevant hashtags with #)."""
    c = studio.client()
    resp = c.models.generate_content(model=studio.TEXT_MODEL, contents=prompt,
                                     config=types.GenerateContentConfig(response_mime_type="application/json",
                                                                        temperature=1.0))
    g = json.loads(resp.text)
    g["cast"] = [k for k in g.get("cast", []) if k in chars["characters"]][:3] or ["milo"]
    scene = {"cast": g["cast"], "image": {k: g.get(k, "") for k in
                                          ("action", "environment", "lighting", "camera", "composition")}}
    scene["image"]["expressions"] = g.get("expressions", {})
    p = studio.compose_image_prompt(scene, chars).replace("vertical 9:16 film still", "4:5 portrait photo")
    raw = DAY / "post_raw.png"
    DAY.mkdir(parents=True, exist_ok=True)
    C = chars["characters"]
    studio.gen_image(c, p, [(C[k]["name"], studio.ref_path(C[k])) for k in g["cast"]], raw, aspect="4:5")
    Image.open(raw).convert("RGB").resize((1080, 1350)).save(out, quality=92)
    meta_p.write_text(json.dumps(g, indent=2, ensure_ascii=False))
    hist.append(g["caption"])
    hist_p.write_text(json.dumps(hist[-120:], indent=2, ensure_ascii=False))
    return out


def cmd_post(_):
    if FREE:
        import free_mode
        ep_id = state().get("episode")
        if not ep_id or not free_mode.final_video(ep_id):
            print("  (free mode) no reel finished today yet; the post is made from the reel, skipping")
            return
        img = DAY / "post.jpg"
        if not img.exists():
            img = free_mode.free_post(ep_id, DAY)
    else:
        img = make_post()
    g = json.loads((DAY / "post.json").read_text())
    text = f"{g['caption']}\n\n{g.get('question', '')}\n\n{' '.join(g.get('hashtags', []))}".strip()
    if not AUTO_PUBLISH:
        print(f"  (review mode) post ready at {img.relative_to(ROOT)}; not published")
        return
    L = ledger()
    fb = step(L, "facebook_post", publish.fb_photo_post, img, text)
    url = fb[1] if fb else ""
    if url:
        step(L, "instagram_post", publish.ig_image, url, text)
    elif publish._env("IG_USER_ID", "FB_PAGE_TOKEN"):
        print("  ! Instagram post skipped: needs the Facebook upload's public URL")


# ============================================================ reel
def cmd_kickoff(_):
    """Free mode only: start today's Kaggle GPU render (the reel job publishes it later)."""
    if not FREE:
        print("  (paid mode) nothing to kick off")
        return
    import free_mode
    st = state()
    if st.get("free_job"):
        print(f"  • already started today: {st['free_job']}")
        return
    save_state(**free_mode.kickoff(os.getenv("FREE_EPISODE") or None))


def cmd_reel(_):
    if ledger().done("youtube_reel") and (not AUTO_PUBLISH or ledger().done("facebook_reel")):
        print("  • today's reel already published")
        return
    st = state()
    if FREE:
        import free_mode
        if not st.get("free_job"):
            studio.die("Free mode: no Kaggle render was started today (the kickoff job did not run or failed).")
        ep_id = st["episode"]
        if not free_mode.final_video(ep_id) or not st.get("edited"):
            if not free_mode.collect(ep_id, st["free_job"]):
                print("  ⏳ Kaggle is still rendering; the next scheduled reel run will try again.")
                return
            save_state(edited=True)
        studio.EPISODE_FILE = "episode.free.json"
    else:
        ep_id = st.get("episode")
        if not ep_id:
            a = ns(ep=None)
            studio.cmd_ideas(a)
            ep_id = a.ep
            save_state(episode=ep_id)
        print(f"→ today's episode: {ep_id}")
        a = ns(ep=ep_id)
        for fn in (studio.cmd_docs, studio.cmd_images, studio.cmd_videos, studio.cmd_voice):
            fn(a)
        try:
            studio.cmd_align(a)
        except SystemExit:
            print("  ! caption alignment failed; using script timings")
        studio.cmd_edit(a)
        try:
            studio.cmd_thumb(a)
        except SystemExit:
            print("  ! thumbnail failed (non-blocking)")
        studio.cmd_qc(a)
    publish_reel(ep_id)


def publish_reel(ep_id: str) -> None:
    ep, d = studio.load_ep(ep_id)
    final = next((d / "11_Final_Edited_Video").glob("*_final.mp4"))
    info = studio.probe(final)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    dur = float(info["format"]["duration"])
    min_s = 8 if FREE else 15
    gate = v["width"] == 1080 and v["height"] == 1920 and min_s <= dur <= 60 and \
        any(s["codec_type"] == "audio" for s in info["streams"])
    if not gate:
        studio.die(f"QC gate failed for {final.name} ({v['width']}x{v['height']}, {dur:.1f}s); not publishing.")

    m = ep["metadata"]
    yt = m["youtube"]
    L = ledger()
    step(L, "youtube_reel", publish.youtube_upload, final, yt["title"],
         yt["description"], yt.get("tags", []), public=AUTO_PUBLISH)
    if not AUTO_PUBLISH:
        print("  (review mode) Instagram/Facebook reels skipped")
        return
    ig = m["instagram"]
    step(L, "instagram_reel", publish.ig_reel, final,
         f"{ig['caption']}\n\n{ig['engagement_question']}\n\n{' '.join(ig['hashtags'])}")
    fb = m["facebook"]
    step(L, "facebook_reel", publish.fb_reel, final,
         f"{fb['caption']}\n\n{fb['engagement_question']}\n\n{' '.join(fb['hashtags'])}")


# ============================================================ story
def make_story() -> Path:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    out = DAY / "story.jpg"
    font_p = str(next(studio.FONT_DIR.glob("*.ttf")))
    ep_id = state().get("episode")
    thumb = None
    if ep_id:
        d = studio.ep_dir(ep_id)
        t = d / "12_Thumbnail" / f"{ep_id}_thumbnail.jpg"
        thumb = t if t.exists() else None
    if thumb:
        canvas = Image.open(thumb).convert("RGB").resize((1080, 1920))
        banner = "NEW VIDEO OUT NOW"
    else:
        if FREE:
            studio.die("Free mode: no reel thumbnail today, so no story (nothing paid is generated in free mode).")
        post = make_post()
        src = Image.open(post).convert("RGB")
        canvas = src.resize((1080, 1920)).filter(ImageFilter.GaussianBlur(40))
        fg = src.resize((1000, 1250))
        canvas.paste(fg, (40, 330))
        banner = "NEW POST ON OUR PAGE"
    dr = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(font_p, 84)
    w = dr.textbbox((0, 0), banner, font=font, stroke_width=8)[2]
    dr.rounded_rectangle((60, 1460, 1020, 1610), radius=40, fill=(255, 214, 0))
    dr.text(((1080 - w) // 2, 1485), banner, font=font, fill=(20, 20, 20))
    DAY.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=92)
    return out


def cmd_story(_):
    img = make_story()
    if not AUTO_PUBLISH:
        print(f"  (review mode) story ready at {img.relative_to(ROOT)}; not published")
        return
    L = ledger()
    fb = step(L, "facebook_story", publish.fb_story, img)
    url = fb[1] if fb else ""
    if url:
        step(L, "instagram_story", publish.ig_image, url, "", story=True)


# ============================================================ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["kickoff", "post", "reel", "story", "all"])
    a = p.parse_args()
    jobs = (["kickoff"] if FREE else ["post", "reel", "story"]) if a.what == "all" else [a.what]
    failed = []
    for j in jobs:
        print(f"\n######## DAILY {j.upper()} ({TODAY}) ########")
        try:
            globals()[f"cmd_{j}"](a)
        except SystemExit as e:
            if e.code not in (0, None):
                failed.append(j)
        except Exception:
            traceback.print_exc()
            failed.append(j)
    if failed:
        print(f"\n✖ failed: {failed}")
        sys.exit(1)
    print("\n✔ daily run complete")


if __name__ == "__main__":
    main()
