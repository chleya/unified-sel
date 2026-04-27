from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_proposal_replay import replay_proposal_case, summarize_replay
from core.rewrite_provider_capture import capture_jsonl_to_replay_dataset


def run_capture_replay(path: Path) -> dict[str, object]:
    replay_cases = capture_jsonl_to_replay_dataset(path)
    rows = [replay_proposal_case(case) for case in replay_cases]
    return {
        "capture_file": str(path),
        "summary": summarize_replay(rows),
        "cases": [row.__dict__ for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-file", default=str(PROJECT_ROOT / "data" / "lsg" / "provider_capture_v0.jsonl"))
    parser.add_argument("--label", default="smoke")
    args = parser.parse_args()

    result = run_capture_replay(Path(args.capture_file))
    out_dir = PROJECT_ROOT / "results" / "capability_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rewrite_provider_capture_replay_{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"[OK] wrote {out_path}")
    if not result["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
