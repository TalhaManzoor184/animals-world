# Real Animal Comedy Shorts Studio

A Gemini API pipeline for photorealistic talking-animal comedy Shorts/Reels (9:16, 1080x1920, 15–35s), starring
**Milo, Bruno, Rocky, Ducky, Coco and Chiku**.

```
config/characters.json          ← character bible (looks, voices, reference paths); edit here and every prompt follows
references/boards/              ← your two original character boards
references/characters/*_crop    ← per-character references cut from board 2 (*_sheet.png get added by `refs`)
episodes/00_concepts_batch_01.md← 10 scored concepts
episodes/history.json           ← episode log (keeps `ideas` from repeating itself)
episodes/EP001_operation_watermelon/
   episode.json                 ← SINGLE SOURCE OF TRUTH for the episode
   01_… → 15_…                  ← the 15-folder delivery structure
   status.json / QC_REPORT.md   ← what actually exists + checklist
pipeline/studio.py              ← the pipeline
.github/workflows/produce-episode.yml
```

## Models (Nov 2026; override with env vars)
| Step | Default | Env var |
|---|---|---|
| Scene stills + reference sheets + thumbnail | `gemini-3.1-flash-image` (Nano Banana 2, multi-reference) | `IMAGE_MODEL` |
| Video (8s clips, native audio and lip-synced dialogue) | `veo-3.1-fast-generate-preview` (use `veo-3.1-generate-preview` for best quality) | `VIDEO_MODEL` |
| Voices (only if `DIALOGUE_MODE=tts`) | `gemini-2.5-flash-preview-tts` | `TTS_MODEL` |
| Ideas + caption alignment | `gemini-flash-latest` | `TEXT_MODEL` |

Veo, the video model, needs a **paid** Gemini API key (billing enabled).

## How it works
1. `refs` (one time): turns each board crop into a clean full-body reference sheet. **Review these.** They lock the look of every later shot.
2. `images EP001`: one first-frame still per scene, generated with the reference image of every character in that shot attached.
3. `videos EP001`: Veo animates each still (image-to-video, 9:16, 8s) with dialogue spoken in-clip.
   - `VIDEO_MODE=reference` skips the stills and feeds up to 3 character sheets straight to Veo instead.
   - `DIALOGUE_MODE=tts` makes Veo stay silent on dialogue, then Gemini TTS voices are mixed in. This gives more consistent voices but loose lip-sync.
4. `voice`: extracts each clip's audio (veo mode) or generates the TTS lines (tts mode).
5. `align` (optional): Gemini listens to each clip and times the captions to where the lines are actually spoken.
6. `edit`: trims, punch-in zooms, white flash, burned-in captions (with speaker tags and a yellow punchline), music bed ducked under the punchline, loudness at −14 LUFS, and export to H.264 1080x1920 30fps.
7. `thumb`, then `qc`.

Every step skips files that already exist, so if one scene is bad, run `--scene N --force` for just that scene.
**Nothing is ever uploaded.** You review and publish yourself (tick YouTube's *altered or synthetic content* box).

## Run locally
```bash
pip install -r pipeline/requirements.txt      # plus ffmpeg on PATH
export GEMINI_API_KEY=...
python pipeline/studio.py plan EP001           # free: shows every prompt + cost estimate
python pipeline/studio.py refs
python pipeline/studio.py images EP001         # review 06_Image_Assets
python pipeline/studio.py videos EP001         # review 08_Video_Assets
python pipeline/studio.py voice EP001 && python pipeline/studio.py align EP001
python pipeline/studio.py edit EP001 && python pipeline/studio.py thumb EP001 && python pipeline/studio.py qc EP001
python pipeline/studio.py ideas                # next episode: 10 concepts + full episode.json
```

## 🆓 FREE mode (default: repo variable `PRODUCTION_MODE=free`)
Costs $0 per day.

| Part | Free tool |
|---|---|
| Ideas, script, captions | Gemini API **free tier** (text model) |
| Voices | Gemini free-tier TTS, and when its daily quota runs out, **Kokoro-82M** (open source, runs on the GitHub runner) |
| Video clips | **Wan 2.2 TI2V-5B** (open source, Apache 2.0) on **Kaggle's free GPU** (`pipeline/free/render_kernel.py`) |
| Edit, captions, publish | ffmpeg + GitHub Actions (public repos get unlimited Actions minutes) |
| Post + story | Clean frame from today's reel, plus meme text written by free Gemini |

How a free reel is built: every spoken line becomes a **close-up of that character talking**, animated from their reference image, so the face never changes. Each scene can also get one short **action shot** (text-to-video) for the physical gag. Clips are 480x832 and upscaled to 1080x1920.

Daily flow:
1. **05:13 UTC** `kickoff` writes the episode, builds the shot list and voices, uploads a private Kaggle dataset, and starts the Kaggle GPU kernel.
2. **15:41 UTC** `reel` downloads the clips, edits, and publishes. If Kaggle is still rendering, the 18:11 run retries.
3. **18:37** post, then **21:23** story, both made from today's reel.

**Setup for free mode**
1. Make a free Kaggle account, **verify your phone number** (required for GPU), then go to Settings → API → *Create New Token*.
2. Add repo secrets `KAGGLE_USERNAME` and `KAGGLE_KEY` (from `kaggle.json`) and `GEMINI_API_KEY` (a free key works).
3. Put sharp, high-res, front-facing images of each character in `references/portraits/` (`milo.png`, `bruno.png`, …).
   These are animated directly, so their quality is the quality of every reel. Until you add them, the small board crops are used.
4. First test: Actions → *Daily autopilot* → run `kickoff`, wait ~2–3 h, then run `reel` (with `AUTO_PUBLISH=false` to review first).
   Set repo variable `FREE_EPISODE=EP001` to render the existing episode instead of writing a new one, then delete the variable.

Limits to know:
- Kaggle gives a weekly free GPU quota, enough for roughly one reel a day.
- No lip-sync: mouths move, but not exactly to the words.
- Quality is below Veo.
- Tuning variables: `FREE_STEPS` (default 30), `FREE_WIDTH`/`FREE_HEIGHT` (480x832; 704x1280 looks better but is ~3x slower), `TTS_ENGINE` (auto/gemini/kokoro).
- Switch back to Veo any time with `PRODUCTION_MODE=paid`.

## Daily autopilot (`.github/workflows/daily.yml` + `pipeline/daily.py`)
Every day, on its own (times in UTC; PKT = +5):

| UTC | PKT | Job | Goes to |
|---|---|---|---|
| 05:13 | 10:13 AM | **kickoff** (free mode only): start the Kaggle render | none |
| 15:41 (+18:11 retry) | 8:41 PM | **reel** | YouTube Shorts + Instagram Reel + Facebook Reel |
| 18:37 | 11:37 PM | **post**: meme still from the reel (free) / generated image (paid) | Facebook Page photo + Instagram feed |
| 21:23 | 2:23 AM | **story**: 9:16 teaser of today's reel | Facebook story + Instagram story |

- **Idempotent:** `daily/<date>/published.json` logs every post, so a re-run never double-posts. A failed reel resumes the same episode, because clips are cached between runs.
- **Review mode:** set repo variable `AUTO_PUBLISH=false`. YouTube then uploads as *private*, and Instagram/Facebook are skipped. Files are still built and downloadable from the run's artifact.
- **QC gate:** the reel is not posted unless the final file is 1080x1920, 15–60s, and has audio.
- Any platform whose secrets are missing is simply skipped.
- Run any job now: Actions → *Daily autopilot* → Run workflow → post / reel / story / all.

### Secrets (Settings → Secrets and variables → Actions)
| Secret | What |
|---|---|
| `GEMINI_API_KEY` | Gemini API key (a free key is enough for free mode; paid mode/Veo needs billing) |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Free mode: Kaggle account (phone-verified) API token |
| `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN` | Google Cloud OAuth client (Desktop app) + refresh token with scope `https://www.googleapis.com/auth/youtube.upload`, for the channel's Google account |
| `FB_PAGE_ID`, `FB_PAGE_TOKEN` | Facebook Page id + **long-lived Page access token** with `pages_manage_posts, pages_read_engagement, pages_show_list, instagram_basic, instagram_content_publish` |
| `IG_USER_ID` | Instagram **Business/Creator** account id linked to that Page (`GET /{page-id}?fields=instagram_business_account`) |

Instagram images are published from the public URL of the Facebook upload, so the Instagram post and story need the Facebook secrets too.

### Manual production (`produce-episode.yml`)
Actions → *Produce episode* → choose an episode + step. Useful for redoing one scene (`--scene 3 --force`) by hand.

Generated media (images, clips, audio) is **not committed**. It lives in the Actions cache between runs and in each run's
artifact (14 days) for download. Only episode JSON, docs and logs are committed.

## Cost (approximate; check ai.google.dev/pricing)
- Reel: 3–5 clips × 8s on Veo 3.1 Fast 1080p (~$0.12/s), plus images, comes to **~$3–5 per day**.
- Post + story: ~1 image, about **$0.07 per day**.
- **So budget about $100–150 per month** for the full daily autopilot. Set a budget alert in Google Cloud billing.
  To cut cost: `VIDEO_RES=720p`, or run the reel cron every other day.

## Music
Drop royalty-free tracks in `assets/music/` (YouTube Audio Library works). EP001 expects `sneaky_pizzicato.mp3`.
Without it, the edit exports with Veo's own SFX only.

## Character canon note
The brief and the two boards disagree on a few accessories. The bible follows **board 2**, because those images are the
references Gemini sees: Ducky and Chiku wear gold chains, Rocky wears a black t-shirt, Milo's hoodie is plain red with
white drawstrings, and Rocky wears normal black sunglasses (not the pixel "meme" ones from board 1). To change any of these, edit the
`look` line in `config/characters.json` and regenerate the sheets with `refs --force`.
