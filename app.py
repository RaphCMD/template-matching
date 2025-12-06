import datetime
from pathlib import Path

from flask import Flask, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from matcher_backend import (
    DEFAULT_SEARCH_DIR,
    DEFAULT_TARGET,
    run_template_match,
)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = Path("uploads")
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def build_target_preview_url(target_path: Path) -> str:
    return url_for("target_preview", path=str(target_path.resolve()))


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def defaults_context() -> dict:
    return {
        "default_target": DEFAULT_TARGET.name,
        "default_search_dir": str(DEFAULT_SEARCH_DIR),
        "default_preview_url": build_target_preview_url(DEFAULT_TARGET),
    }


def resolve_search_dir(raw_path: str) -> Path:
    path = Path(raw_path.strip()).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path)


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _is_allowed_target_path(file_path: Path) -> bool:
    resolved = file_path.resolve()
    allowed_dirs = [
        DEFAULT_TARGET.resolve().parent,
        app.config["UPLOAD_FOLDER"].resolve(),
    ]
    if resolved == DEFAULT_TARGET.resolve():
        return True
    return any(_is_relative_to(resolved, base) for base in allowed_dirs)


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        results=None,
        csv_path=None,
        target_label=DEFAULT_TARGET.name,
        target_preview_url=build_target_preview_url(DEFAULT_TARGET),
        search_dir=str(DEFAULT_SEARCH_DIR),
        use_pyramids=True,
        use_multithreading=True,
        elapsed_time=None,
        error=None,
        defaults=defaults_context(),
    )


@app.route("/search", methods=["POST"])
def search():
    target_source = request.form.get("target_source", "default")
    search_dir_input = request.form.get("search_dir", "").strip()
    search_dir = (
        resolve_search_dir(search_dir_input)
        if search_dir_input
        else resolve_search_dir(str(DEFAULT_SEARCH_DIR))
    )
    use_pyramids = request.form.get("use_pyramids") is not None
    use_multithreading = request.form.get("use_multithreading") is not None

    target_path = DEFAULT_TARGET
    error = None
    elapsed_time = None

    if target_source == "upload":
        target_file = request.files.get("target_file")
        if not target_file or target_file.filename == "":
            error = "Please choose a target image to upload."
        elif not allowed_file(target_file.filename):
            error = "Supported target formats: png, jpg, jpeg, bmp."
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = secure_filename(target_file.filename)
            saved_path = app.config["UPLOAD_FOLDER"] / f"{timestamp}_{filename}"
            target_file.save(saved_path)
            target_path = saved_path

    target_preview_url = build_target_preview_url(target_path)

    if error:
        return render_template(
            "index.html",
            results=None,
            csv_path=None,
            target_label=target_path.name,
            target_preview_url=target_preview_url,
            search_dir=str(search_dir),
            use_pyramids=use_pyramids,
            use_multithreading=use_multithreading,
            elapsed_time=elapsed_time,
            error=error,
            defaults=defaults_context(),
        )

    start_time = datetime.datetime.now()
    try:
        results, csv_path = run_template_match(
            target_path,
            search_dir,
            use_pyramids=use_pyramids,
            use_multithreading=use_multithreading,
        )
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    except Exception as exc:
        if start_time:
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        return render_template(
            "index.html",
            results=None,
            csv_path=None,
            target_label=target_path.name,
            target_preview_url=target_preview_url,
            search_dir=str(search_dir),
            use_pyramids=use_pyramids,
            use_multithreading=use_multithreading,
            elapsed_time=elapsed_time,
            error=str(exc),
            defaults=defaults_context(),
        )

    display_results = []
    for idx, item in enumerate(results, start=1):
        preview_url = url_for(
            "preview_image",
            folder=str(search_dir),
            filename=item["filename"],
        )
        display_results.append(
            {
                "rank": idx,
                "filename": item["filename"],
                "score": item["score"],
                "preview_url": preview_url,
            }
        )

    return render_template(
        "index.html",
        results=display_results,
        csv_path=csv_path,
        target_label=target_path.name,
        target_preview_url=target_preview_url,
        search_dir=str(search_dir),
        use_pyramids=use_pyramids,
        use_multithreading=use_multithreading,
        elapsed_time=elapsed_time,
        error=None,
        defaults=defaults_context(),
    )


@app.route("/preview")
def preview_image():
    folder = request.args.get("folder", "")
    filename = request.args.get("filename", "")
    if not folder or not filename:
        return ("", 404)

    folder_path = Path(folder).expanduser().resolve()
    file_path = (folder_path / filename).resolve()

    # Simple guard against path traversal
    if not file_path.exists() or not str(file_path).startswith(str(folder_path)):
        return ("", 404)

    return send_file(file_path)


@app.route("/target-preview")
def target_preview():
    raw_path = request.args.get("path", "")
    if not raw_path:
        return ("", 404)

    file_path = Path(raw_path).expanduser().resolve()
    if not file_path.exists() or not _is_allowed_target_path(file_path):
        return ("", 404)

    return send_file(file_path)


if __name__ == "__main__":
    app.run(debug=True)
