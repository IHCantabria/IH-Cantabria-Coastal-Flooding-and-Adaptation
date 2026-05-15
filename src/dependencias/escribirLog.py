from __future__ import annotations

from datetime import datetime
from pathlib import Path


def escribir_log(log_file, mensaje):
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {mensaje}\n")
