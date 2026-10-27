"""Inspect an independent patch request. This command never starts a solver."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.local_patch import audit_patch_sources, load_patch_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='New source-audit JSON; existing files are preserved')
    args = parser.parse_args()
    result = audit_patch_sources(load_patch_config(args.config), ROOT)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf8') as stream:
            stream.write(encoded)
    print(json.dumps({k: result[k] for k in ('status', 'request_sha256', 'missing_or_changed_count', 'runtime_ready')}, ensure_ascii=False))
    return 2 if not result['runtime_ready'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
