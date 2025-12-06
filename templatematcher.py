import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from threading import Lock
import csv
import datetime

# --- CONFIG ---
query_path = "target.png"
folder_path = "searchfolder"
clean_folder = "searchfolder_clean"  # Folder for cleaned images
correlation_folder = "correlation_maps"  # Folder for correlation maps
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"match_results_{timestamp}.csv"



method = cv2.TM_CCOEFF_NORMED
num_workers = 6  # Matches 6-core processor
max_pyramid_levels = 3  # Number of pyramid levels for scale invariance
log_scale_factor = 0.0001  # Controls logarithmic tolerance (higher = stricter)
save_correlation_maps = False  # Boolean to toggle correlation map saving (Purely for fun lol, slows it down by orders of magnitude)

# Create folders
os.makedirs(clean_folder, exist_ok=True)
if save_correlation_maps:
    os.makedirs(correlation_folder, exist_ok=True)

# Initializion
all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]
total_files = len(all_files)
completed_files = 0
matched_files = 0
completed_matches = 0
file_lock = Lock()
match_lock = Lock()
csv_lock = Lock()


def print_file_progress(file, phase):
    global completed_files
    global matched_files

    fileCount = completed_files if phase != "Matching" else matched_files
    with file_lock:
        fileCount += 1
        percentage = (fileCount / total_files) * 100  
        print(f"{phase} file {fileCount}/{total_files} ({percentage:.2f}%): {file}")

    if phase == "Matching":
        matched_files = fileCount
        return
    completed_files = fileCount
    

# Print match completion percentage


# Update CSV with new result, keeping it sorted in ascending order
def update_csv(new_result):
    with csv_lock:
        # Read existing results
        existing_results = []
        try:
            with open(results_file, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None)  # Skip header
                if header:
                    for row in reader:
                        if len(row) == 2:
                            existing_results.append((float(row[0]), row[1]))
        except FileNotFoundError:
            pass  # File doesn't exist yet

        # Add new result
        if new_result:
            existing_results.append(new_result)

        # Sort by score (ascending)
        existing_results.sort(key=lambda x: x[0], reverse=False)

        # Write back to CSV
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Score', 'Filename'])
            for score, name in existing_results:
                writer.writerow([f"{score:.4f}", name])

# Preprocess images to remove ICC profiles
def clean_png(input_path, output_path, file_name):
    if os.path.exists(output_path):
        print_file_progress(file_name, "Skipping cleaning (already exists)")
        return True
    print_file_progress(file_name, "Cleaning")
    try:
        img = Image.open(input_path)
        img.save(output_path, "PNG", icc_profile=None)
        return True
    except Exception as e:
        print(f"Error cleaning {file_name}: {str(e)}")
        return False

# Clean target.png
completed_files = -1
print(f"Cleaning query image: {query_path}")
clean_query_path = os.path.join("./", "target_clean.png")
if not clean_png(query_path, clean_query_path, "target.png"):
    raise ValueError("Failed to clean query image.")


# Clean all images in folder
print(f"\nCleaning {total_files} PNGs from {folder_path} to {clean_folder}...")
for file in all_files:
    clean_png(
        os.path.join(folder_path, file),
        os.path.join(clean_folder, file),
        file
    )

# Update folder_path for matching
folder_path = clean_folder

# Load query image
print(f"\nLoading query image: {clean_query_path}")
query = cv2.imread(clean_query_path)
if query is None:
    raise ValueError(f"Failed to load query image: {clean_query_path}")

query_shape = query.shape[:2]
print(f"Query image shape: {query_shape}")

# Process image for template matching with pyramids
def process_image(file):
    completed_files = 0
    print_file_progress(file, "Matching")
    if not file.lower().endswith(".png"):
        return None

    path = os.path.join(folder_path, file)
    try:
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load {file}")
            # print_match_progress()
            return None

        if img.shape[0] < query.shape[0] or img.shape[1] < query.shape[1]:
            print(f"Skipping {file}: Image too small (shape: {img.shape[:2]})")
            # print_match_progress()
            return None

        # Create image pyramid for source image
        best_score = -1
        best_res = None
        best_scale = 1.0
        current_img = img
        for level in range(max_pyramid_levels):
            if current_img.shape[0] < query.shape[0] or current_img.shape[1] < query.shape[1]:
                break
            res = cv2.matchTemplate(current_img, query, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            score = max_val  # TM_CCOEFF_NORMED gives max_val in [0, 1]
            if score > best_score:
                best_score = score
                best_res = res
                best_scale = 1.0 / (2 ** level)  # Track scale for map sizing
            current_img = cv2.pyrDown(current_img)

        # Save correlation map with logarithmic scaling, aligned to kernel center
        if save_correlation_maps and best_res is not None:
            # Initialize output map with same size as input image
            img_height, img_width = img.shape[:2]
            corr_map = np.zeros((img_height, img_width), dtype=np.uint8)
            
            # Compute offsets for kernel center
            qh, qw = query_shape
            offset_h, offset_w = qh // 2, qw // 2
            
            # Map correlation scores to center positions with logarithmic scaling
            res_height, res_width = best_res.shape
            for i in range(res_height):
                for j in range(res_width):
                    # Center position in original image (adjusted for scale)
                    center_i = int(i * best_scale + offset_h)
                    center_j = int(j * best_scale + offset_w)
                    if 0 <= center_i < img_height and 0 <= center_j < img_width:
                        # Logarithmic scaling: clip negative scores, emphasize near-1.0
                        score = max(best_res[i, j], 0)  # Clip negative scores
                        intensity = 255 * np.log(1 + log_scale_factor * score) / np.log(1 + log_scale_factor)
                        corr_map[center_i, center_j] = intensity.astype(np.uint8)

            # Save as PNG
            corr_path = os.path.join(correlation_folder, f"{os.path.splitext(file)[0]}_corr.png")
            cv2.imwrite(corr_path, corr_map)

        # print_match_progress()
        return (best_score, file) if best_score > -1 else None
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        # print_match_progress()
        return None

# Run multithreaded match
results = []
print(f"\nStarting template matching with {num_workers} workers...")
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(process_image, f): f for f in all_files}
    for future in as_completed(futures):
        result = future.result()
        if result:
            update_csv(result)  # Update CSV in real time

print(f"\nResults saved to {results_file}")
# Sort results file in descending order by score
with open(results_file, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader, None)
    results = []
    for row in reader:
        if len(row) == 2:
            results.append((float(row[0]), row[1]))

results.sort(key=lambda x: x[0], reverse=True)

with open(results_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Score', 'Filename'])
    for score, name in results:
        writer.writerow([f"{score:.4f}", name])

if results:
    print(f"Top match: {results[0][1]}")
else:
    print("No matches found.")
if save_correlation_maps:
    print(f"Correlation maps saved to {correlation_folder}")
else:
    print("Correlation maps skipped (save_correlation_maps=False)")
