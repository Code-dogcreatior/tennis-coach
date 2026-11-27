# Reproducible download of the local YouTube test clips (metadata saved next to each file as *.info.json).
# Only public videos, no cookies / login, no bulk scraping. Check the "license" field in info.json before use.
#   powershell -ExecutionPolicy Bypass -File scripts/fetch_youtube.ps1
$ErrorActionPreference = "Continue"

# Creative Commons (CC BY) clips - license verified via yt-dlp "license" field on 2026-11-17
$cc = @("5lf-t4obkkU", "nW5QJN-0EeQ", "Rw3ZDfTTD9M")
foreach ($id in $cc) {
  uv run yt-dlp -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]" --merge-output-format mp4 `
    --write-info-json --no-playlist -o "data/raw/youtube_cc/%(id)s/%(id)s.%(ext)s" "https://www.youtube.com/watch?v=$id"
}

# Standard-YouTube-licence clips: local testing only (not added to the reference library, not redistributed)
$std = @("pg_dxIvJRZo", "WsQ0ghZg5FE", "eBa-sy2Mt1s", "kJUuzlk6d8M", "rTXgMbyjMuU", "gdJXQ69yuMw", "TdlXwYS_Neo", "5_SnJPl4cz4")
foreach ($id in $std) {
  uv run yt-dlp -f "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]" --merge-output-format mp4 `
    --write-info-json --no-playlist --download-sections "*0-180" -o "data/raw/youtube/%(id)s/%(id)s.%(ext)s" "https://www.youtube.com/watch?v=$id"
}
# the side-by-side serve clip is split into its two camera views
ffmpeg -v error -y -i data/raw/youtube/eBa-sy2Mt1s/eBa-sy2Mt1s.mp4 -vf "crop=iw/2:ih:0:0" -an data/raw/youtube/eBa-sy2Mt1s/eBa_left.mp4
ffmpeg -v error -y -i data/raw/youtube/eBa-sy2Mt1s/eBa-sy2Mt1s.mp4 -vf "crop=iw/2:ih:iw/2:0" -an data/raw/youtube/eBa-sy2Mt1s/eBa_right.mp4
