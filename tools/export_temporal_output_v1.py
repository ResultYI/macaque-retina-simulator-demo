"""Export a separate experimental curve from an existing verified cache."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import numpy as np
from retina_platform.replay_data import ReplayData
from retina_platform.temporal_output import EXPERIMENTAL, replay_prediction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--case", choices=["cell6_g"+str(i) for i in range(6)], default="cell6_g3")
    p.add_argument("--experimental", action="store_true", required=True,
                   help="Explicitly authorize the INCONCLUSIVE output candidate; never a default")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    metadata_path = args.output.with_suffix(".json")
    if args.output.exists() or metadata_path.exists():
        raise ValueError("Use a new output path; original curves are never overwritten")
    data = ReplayData(args.project_root).load_case(args.case)
    rate, metadata = replay_prediction(data, data.nodes[data.target])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, time_source_s=np.arange(7100)*.001,
                            expected_count_rate_Hz=rate, probability=rate*.001)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    print(json.dumps(dict(mode=EXPERIMENTAL, output=str(args.output), metadata=str(metadata_path)),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
