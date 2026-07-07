from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aido_bba.demo_pipeline import run_demo


def main() -> int:
    print("AIDO-BBA compact demonstration")
    print("--------------------------------")
    try:
        summary = run_demo(REPO_ROOT / "demo" / "demo_config.json")
    except Exception as exc:
        print(f"DEMO STATUS: FAILED\n{exc}")
        print("See demo/demo_outputs/failure_log.csv")
        return 1

    print("Input validation: PASSED")
    print(f"Matched patients: {summary['n_matched_patients']}")
    print(f"Genes retained: {summary['n_input_genes']}")
    print("Repeated modelling: COMPLETED")
    print("Patient reliability audit: COMPLETED")
    print("Representation audit: COMPLETED")
    print("Output schema validation: PASSED")
    print("Outputs written to: demo/demo_outputs/")
    print("DEMO STATUS: SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
