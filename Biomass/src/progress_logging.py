
def log_progress(enable_progress_prints: bool, message: str) -> None:
    if enable_progress_prints:
        print(f"[INFO] {message}")
