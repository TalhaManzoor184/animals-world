"""
Publishers for YouTube Shorts, Instagram (feed / story / reel) and Facebook Page (photo / story / reel).

Every platform is optional: if its secrets are missing it is skipped with a log line.
Every successful publish is recorded in a JSON log; a re-run never posts the same item twice.

Secrets (GitHub → Settings → Secrets → Actions):
  YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN     (scope: youtube.upload)
  FB_PAGE_ID, FB_PAGE_TOKEN                             (long-lived Page token: pages_manage_posts,
                                                         pages_read_engagement, pages_show_list,
                                                         instagram_basic, instagram_content_publish)
  IG_USER_ID                                            (Instagram Business/Creator id linked to the Page)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

GRAPH = f"https://graph.facebook.com/{os.getenv('GRAPH_VERSION', 'v24.0')}"
RUPLOAD = f"https://rupload.facebook.com/video-upload/{os.getenv('GRAPH_VERSION', 'v24.0')}"
TIMEOUT = 120


class PublishError(RuntimeError):
    pass


def _env(*names: str) -> dict | None:
    vals = {n: os.getenv(n, "").strip() for n in names}
    return vals if all(vals.values()) else None


def _check(r: requests.Response, what: str) -> dict:
    try:
        data = r.json()
    except ValueError:
        data = {"raw": r.text[:500]}
    if r.status_code >= 400 or (isinstance(data, dict) and "error" in data):
        raise PublishError(f"{what} failed [{r.status_code}]: {json.dumps(data)[:800]}")
    return data


# ----------------------------------------------------------------- idempotency log
class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def done(self, key: str) -> bool:
        return key in self.data

    def mark(self, key: str, value: dict) -> None:
        self.data[key] = {**value, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


# ================================================================= YouTube
def youtube_upload(video: Path, title: str, description: str, tags: list[str], public: bool = True) -> str | None:
    cfg = _env("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN")
    if not cfg:
        print("  • YouTube: secrets not set, skipping")
        return None
    tok = _check(requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": cfg["YT_CLIENT_ID"], "client_secret": cfg["YT_CLIENT_SECRET"],
        "refresh_token": cfg["YT_REFRESH_TOKEN"], "grant_type": "refresh_token"}, timeout=TIMEOUT),
        "YouTube token refresh")["access_token"]
    meta = {
        "snippet": {"title": title[:100], "description": description[:4900], "tags": tags[:30],
                    "categoryId": "23", "defaultLanguage": "en", "defaultAudioLanguage": "en"},
        "status": {"privacyStatus": "public" if public else "private", "selfDeclaredMadeForKids": False,
                   "containsSyntheticMedia": True, "embeddable": True},
    }
    size = video.stat().st_size
    r = requests.post("https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
                      headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json; charset=UTF-8",
                               "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(size)},
                      data=json.dumps(meta), timeout=TIMEOUT)
    _check(r, "YouTube upload init")
    with video.open("rb") as f:
        up = requests.put(r.headers["Location"], headers={"Authorization": f"Bearer {tok}", "Content-Type": "video/mp4"},
                          data=f, timeout=900)
    vid = _check(up, "YouTube upload")["id"]
    print(f"  ✔ YouTube: https://youtube.com/shorts/{vid}" + ("" if public else " (private, review mode)"))
    return vid


# ================================================================= Facebook Page
def _fb() -> dict | None:
    cfg = _env("FB_PAGE_ID", "FB_PAGE_TOKEN")
    if not cfg:
        print("  • Facebook: secrets not set, skipping")
    return cfg


def _fb_photo(cfg: dict, image: Path, message: str = "", published: bool = True) -> tuple[str, str]:
    """Upload a photo to the Page. Returns (photo_id, public CDN url)."""
    with image.open("rb") as f:
        data = _check(requests.post(f"{GRAPH}/{cfg['FB_PAGE_ID']}/photos",
                                    data={"access_token": cfg["FB_PAGE_TOKEN"], "message": message,
                                          "published": "true" if published else "false"},
                                    files={"source": (image.name, f, "image/jpeg")}, timeout=TIMEOUT),
                      "Facebook photo upload")
    pid = data["id"]
    info = _check(requests.get(f"{GRAPH}/{pid}", params={"fields": "images", "access_token": cfg["FB_PAGE_TOKEN"]},
                               timeout=TIMEOUT), "Facebook photo lookup")
    url = max(info.get("images", [{}]), key=lambda x: x.get("width", 0)).get("source", "")
    return pid, url


def fb_photo_post(image: Path, message: str) -> tuple[str, str] | None:
    cfg = _fb()
    if not cfg:
        return None
    pid, url = _fb_photo(cfg, image, message, published=True)
    print(f"  ✔ Facebook photo post: {pid}")
    return pid, url


def fb_story(image: Path) -> tuple[str, str] | None:
    cfg = _fb()
    if not cfg:
        return None
    pid, url = _fb_photo(cfg, image, published=False)
    data = _check(requests.post(f"{GRAPH}/{cfg['FB_PAGE_ID']}/photo_stories",
                                data={"photo_id": pid, "access_token": cfg["FB_PAGE_TOKEN"]}, timeout=TIMEOUT),
                  "Facebook story")
    print(f"  ✔ Facebook story: {data.get('post_id', pid)}")
    return data.get("post_id", pid), url


def fb_reel(video: Path, description: str) -> str | None:
    cfg = _fb()
    if not cfg:
        return None
    page, tok = cfg["FB_PAGE_ID"], cfg["FB_PAGE_TOKEN"]
    start = _check(requests.post(f"{GRAPH}/{page}/video_reels", data={"upload_phase": "start", "access_token": tok},
                                 timeout=TIMEOUT), "Facebook reel start")
    vid = start["video_id"]
    with video.open("rb") as f:
        _check(requests.post(f"{RUPLOAD}/{vid}", headers={"Authorization": f"OAuth {tok}", "offset": "0",
                                                          "file_size": str(video.stat().st_size)},
                             data=f, timeout=900), "Facebook reel upload")
    _check(requests.post(f"{GRAPH}/{page}/video_reels", data={
        "upload_phase": "finish", "video_id": vid, "video_state": "PUBLISHED",
        "description": description, "access_token": tok}, timeout=TIMEOUT), "Facebook reel publish")
    print(f"  ✔ Facebook reel: {vid}")
    return vid


# ================================================================= Instagram
def _ig() -> dict | None:
    cfg = _env("IG_USER_ID", "FB_PAGE_TOKEN")
    if not cfg:
        print("  • Instagram: secrets not set, skipping")
    return cfg


def _ig_wait_and_publish(cfg: dict, creation_id: str, what: str, max_wait: int = 600) -> str:
    tok, t0 = cfg["FB_PAGE_TOKEN"], time.time()
    while True:
        st = _check(requests.get(f"{GRAPH}/{creation_id}", params={"fields": "status_code,status",
                                                                     "access_token": tok}, timeout=TIMEOUT),
                    f"Instagram {what} status")
        code = st.get("status_code")
        if code == "FINISHED":
            break
        if code in ("ERROR", "EXPIRED"):
            raise PublishError(f"Instagram {what} processing {code}: {st.get('status')}")
        if time.time() - t0 > max_wait:
            raise PublishError(f"Instagram {what} still processing after {max_wait}s")
        time.sleep(10)
    pub = _check(requests.post(f"{GRAPH}/{cfg['IG_USER_ID']}/media_publish",
                               data={"creation_id": creation_id, "access_token": tok}, timeout=TIMEOUT),
                 f"Instagram {what} publish")
    return pub["id"]


def ig_image(image_url: str, caption: str, story: bool = False) -> str | None:
    """image_url must be public (we reuse the Facebook CDN url from the Page upload)."""
    cfg = _ig()
    if not cfg:
        return None
    if not image_url:
        raise PublishError("Instagram needs a public image URL (Facebook upload must run first)")
    data = {"image_url": image_url, "access_token": cfg["FB_PAGE_TOKEN"]}
    data.update({"media_type": "STORIES"} if story else {"caption": caption})
    c = _check(requests.post(f"{GRAPH}/{cfg['IG_USER_ID']}/media", data=data, timeout=TIMEOUT),
               "Instagram media container")
    mid = _ig_wait_and_publish(cfg, c["id"], "story" if story else "post")
    print(f"  ✔ Instagram {'story' if story else 'post'}: {mid}")
    return mid


def ig_reel(video: Path, caption: str) -> str | None:
    cfg = _ig()
    if not cfg:
        return None
    tok = cfg["FB_PAGE_TOKEN"]
    c = _check(requests.post(f"{GRAPH}/{cfg['IG_USER_ID']}/media", data={
        "media_type": "REELS", "upload_type": "resumable", "caption": caption, "share_to_feed": "true",
        "access_token": tok}, timeout=TIMEOUT), "Instagram reel container")
    with video.open("rb") as f:
        _check(requests.post(c["uri"], headers={"Authorization": f"OAuth {tok}", "offset": "0",
                                                "file_size": str(video.stat().st_size)}, data=f, timeout=900),
               "Instagram reel upload")
    mid = _ig_wait_and_publish(cfg, c["id"], "reel", max_wait=900)
    print(f"  ✔ Instagram reel: {mid}")
    return mid
