from __future__ import annotations

import sys

from ablation_core import main


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main("no_summary_screening")
