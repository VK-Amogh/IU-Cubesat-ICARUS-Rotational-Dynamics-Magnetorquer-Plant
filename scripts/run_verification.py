"""Run all verification scenarios and generate plots + summary artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from spacecraft_dynamics.verification import run_verification_suite

    output_dir = project_root / "outputs"
    summary = run_verification_suite(output_dir)

    print("Verification artifacts generated in:", output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
