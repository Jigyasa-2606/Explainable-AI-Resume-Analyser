from __future__ import annotations

import os
from pathlib import Path


def load_local_env() -> None:
    here = Path(__file__).resolve().parent
    env_path = next((p for p in (here.parent / ".env", here / ".env") if p.exists()), None)
    if env_path is None:
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
