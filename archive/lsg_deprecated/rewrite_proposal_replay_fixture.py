from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_proposal_replay import replay_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "lsg" / "proposal_replay_v0.json"))
    parser.add_argument("--label", default="smoke")
    parser.add_argument("--disagreement-threshold", type=float, default=0.25)
    args = parser.parse_args()

    result = replay_dataset(Path(args.dataset), disagreement_threshold=args.disagreement_threshold)
    out_dir = PROJECT_ROOT / "results" / "capability_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rewrite_proposal_replay_fixture_{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"[OK] wrote {out_path}")
    if not result["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
