from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_source_distribution import ProposalSourceSpec, compare_sources


def parse_source(value: str) -> ProposalSourceSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("source must be name:kind:path")
    name, kind, path = parts
    if kind not in {"replay_json", "capture_jsonl"}:
        raise argparse.ArgumentTypeError("kind must be replay_json or capture_jsonl")
    return ProposalSourceSpec(name=name, kind=kind, path=Path(path))


def default_sources() -> list[ProposalSourceSpec]:
    return [
        ProposalSourceSpec(
            "hand_authored_replay",
            "replay_json",
            PROJECT_ROOT / "data" / "lsg" / "proposal_replay_v0.json",
        ),
        ProposalSourceSpec(
            "provider_capture_fixture",
            "capture_jsonl",
            PROJECT_ROOT / "data" / "lsg" / "provider_capture_v0.jsonl",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=parse_source, default=[])
    parser.add_argument("--label", default="smoke")
    parser.add_argument("--disagreement-threshold", type=float, default=0.25)
    args = parser.parse_args()

    specs = args.source if args.source else default_sources()
    result = compare_sources(specs, disagreement_threshold=args.disagreement_threshold)
    out_dir = PROJECT_ROOT / "results" / "capability_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rewrite_source_distribution_compare_{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[OK] wrote {out_path}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
