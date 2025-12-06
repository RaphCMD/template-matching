import csv
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Defaults aligned with the sample assets in the repo.
DEFAULT_TARGET = Path("targetdefault.png")
DEFAULT_SEARCH_DIR = Path("searchfolder_default")

# Matching configuration
METHOD = cv2.TM_CCOEFF_NORMED
NUM_WORKERS = 6
MAX_PYRAMID_LEVELS = 3
LOG_SCALE_FACTOR = 0.0001


def clean_png(input_path: Path, output_path: Path) -> bool:
    """
    Remove ICC profiles so OpenCV can ingest the PNG reliably.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return True
    try:
        img = Image.open(input_path)
        img.save(output_path, "PNG", icc_profile=None)
        return True
    except Exception:
        return False


def _build_correlation_map(
    best_res: np.ndarray,
    img_shape: Tuple[int, int],
    query_shape: Tuple[int, int],
    best_scale: float,
    log_scale_factor: float,
) -> np.ndarray:
    img_height, img_width = img_shape
    corr_map = np.zeros((img_height, img_width), dtype=np.uint8)
    qh, qw = query_shape
    offset_h, offset_w = qh // 2, qw // 2
    res_height, res_width = best_res.shape

    for i in range(res_height):
        for j in range(res_width):
            center_i = int(i * best_scale + offset_h)
            center_j = int(j * best_scale + offset_w)
            if 0 <= center_i < img_height and 0 <= center_j < img_width:
                score = max(best_res[i, j], 0)
                intensity = 255 * np.log(1 + log_scale_factor * score) / np.log(
                    1 + log_scale_factor
                )
                corr_map[center_i, center_j] = intensity.astype(np.uint8)

    return corr_map


def _process_image(
    file_path: Path,
    query: np.ndarray,
    query_shape: Tuple[int, int],
    save_correlation_maps: bool,
    correlation_folder: Optional[Path],
    max_pyramid_levels: int,
    log_scale_factor: float,
) -> Optional[float]:
    img = cv2.imread(str(file_path))
    if img is None:
        return None

    if img.shape[0] < query_shape[0] or img.shape[1] < query_shape[1]:
        return None

    best_score = -1.0
    best_res = None
    best_scale = 1.0
    current_img = img

    for level in range(max_pyramid_levels):
        if current_img.shape[0] < query_shape[0] or current_img.shape[1] < query_shape[1]:
            break

        res = cv2.matchTemplate(current_img, query, METHOD)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        score = max_val

        if score > best_score:
            best_score = score
            best_res = res
            best_scale = 1.0 / (2**level)

        current_img = cv2.pyrDown(current_img)

    if save_correlation_maps and best_res is not None and correlation_folder:
        corr_map = _build_correlation_map(
            best_res, img.shape[:2], query_shape, best_scale, log_scale_factor
        )
        corr_path = correlation_folder / f"{file_path.stem}_corr.png"
        cv2.imwrite(str(corr_path), corr_map)

    return best_score if best_score > -1 else None


def run_template_match(
    query_path: Path,
    search_dir: Path,
    *,
    num_workers: int = NUM_WORKERS,
    max_pyramid_levels: int = MAX_PYRAMID_LEVELS,
    log_scale_factor: float = LOG_SCALE_FACTOR,
    save_correlation_maps: bool = False,
    use_pyramids: bool = True,
    use_multithreading: bool = True,
) -> Tuple[List[dict], Optional[Path]]:
    """
    Execute template matching for a target PNG against all PNGs in a directory.

    Returns a tuple of (results, csv_path) where results is a list of dictionaries:
    [{'score': float, 'filename': str, 'filepath': Path}, ...] sorted by score desc.
    """
    query_path = Path(query_path)
    search_dir = Path(search_dir)

    if not query_path.exists():
        raise FileNotFoundError(f"Target image not found: {query_path}")
    if not search_dir.exists() or not search_dir.is_dir():
        raise FileNotFoundError(f"Search directory not found: {search_dir}")

    png_files = sorted([p for p in search_dir.iterdir() if p.suffix.lower() == ".png"])
    if not png_files:
        return [], None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_folder = search_dir.parent / f"{search_dir.name}_clean"
    clean_folder.mkdir(parents=True, exist_ok=True)
    correlation_folder = None
    if save_correlation_maps:
        correlation_folder = search_dir.parent / f"{search_dir.name}_correlation_maps"
        correlation_folder.mkdir(parents=True, exist_ok=True)

    clean_query_path = clean_folder / f"{query_path.stem}_clean_{timestamp}.png"
    if not clean_png(query_path, clean_query_path):
        raise ValueError(f"Could not clean target image: {query_path}")

    for file in png_files:
        clean_png(file, clean_folder / file.name)

    query = cv2.imread(str(clean_query_path))
    if query is None:
        raise ValueError(f"Failed to load cleaned target image: {clean_query_path}")
    query_shape = query.shape[:2]

    effective_levels = max_pyramid_levels if use_pyramids else 1
    results: List[Tuple[float, Path]] = []

    def run_single(file: Path) -> Optional[float]:
        return _process_image(
            clean_folder / file.name,
            query,
            query_shape,
            save_correlation_maps,
            correlation_folder,
            effective_levels,
            log_scale_factor,
        )

    if use_multithreading:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(run_single, file): file for file in png_files}
            for future in as_completed(futures):
                score = future.result()
                original_file = futures[future]
                if score is not None:
                    results.append((score, original_file))
    else:
        for file in png_files:
            score = run_single(file)
            if score is not None:
                results.append((score, file))

    results.sort(key=lambda x: x[0], reverse=True)

    csv_path = Path.cwd() / f"match_results_{timestamp}.csv"
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Score", "Filename"])
        for score, file in results:
            writer.writerow([f"{score:.4f}", file.name])

    formatted_results = [
        {"score": float(score), "filename": file.name, "filepath": file}
        for score, file in results
    ]
    return formatted_results, csv_path
