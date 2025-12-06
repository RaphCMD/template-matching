# Multithreaded Template Matcher with Scale-Invariant Normalized Cross-Correlation

**Raphael Bomshakian**

**Ryan Ahlborn**

## Video Demonstration

https://youtu.be/JaWbpM0Q1ro

To use this script you need:

- A snippet of your target image in the working directory.
- A folder that contains the target image in the working directory.

## Setup

- Python 3.10+ is recommended.
- Install dependencies: `pip install flask opencv-python numpy pillow`
- Default assets: `targetdefault.png` and `searchfolder_default/` live in the repo root; feel free to swap in your own.

## Run the web UI

1. From the repository root: `python app.py`
2. Open `http://127.0.0.1:5000` in your browser.
3. Choose the default target or upload your own; point the search directory at a folder of PNGs (defaults to `searchfolder_default`).
4. Submit the form to start matching. Results are ranked in the table, saved to `match_results_<timestamp>.csv`, and cleaned copies land in `<search_dir>_clean/`.

## Notes

- The matcher uses multithreading by default; disable it in the UI if you want sequential runs.
- Image pyramids are enabled by default for scale invariance; uncheck the option to restrict to a single scale.
- Cleaned images and correlation maps (if enabled) are written beside your search directory to avoid mutating the originals.
