"""Explicit preparation/fit/resume/evaluate entry for the conditional RGC assay."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.rgc_input_clamp_fit import prepare, fit, evaluate, save, now, guard, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "fit", "resume", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=ROOT/"configurations/calibration/rgc_cbem_conditional_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coordinator-release", type=Path)
    parser.add_argument("--test", action="store_true", help="consume the once-only sealed test evaluation after accepted candidate freeze")
    args = parser.parse_args()
    allowed = (ROOT/"reports/layer_calibration/rgc_cbem_conditional_v1").resolve()
    output = args.output.resolve()
    if not output.is_relative_to(allowed):
        parser.error("output must remain under the independent conditional report directory")
    if args.test and args.mode != "evaluate":
        parser.error("--test only belongs to evaluate")
    output.mkdir(parents=True, exist_ok=True)
    # The preparation checks an empty directory, so do not put a lock inside it yet.
    lock_path = output.with_name(output.name+".running")
    try:
        with lock_path.open("x", encoding="utf-8") as stream:
            json.dump(dict(pid=os.getpid(), parent_pid=os.getppid(), mode=args.mode, started=now()), stream)
    except FileExistsError:
        print(f"BLOCKED: run ownership lock exists: {lock_path}; coordinator must inspect its recorded PID before removing a stale lock", file=sys.stderr)
        return 1
    try:
        if args.mode == "dry-run":
            result = prepare(ROOT, args.protocol, output)
        elif args.mode in ("fit", "resume"):
            result = fit(ROOT, args.protocol, output, args.coordinator_release, resume=args.mode == "resume")
        else:
            result = evaluate(ROOT, args.protocol, output, args.coordinator_release, test=args.test)
        if args.mode != "dry-run":
            save(output/f"execution_receipt_{args.mode}.json", dict(mode=args.mode, result=result, timestamp=now(),
                 resources=guard(ROOT, output, read(args.protocol)), production_writeback=False))
        print(json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        save(output/f"failure_{args.mode}_{os.getpid()}.json", dict(status="BLOCKED", action=args.mode,
             error_type=type(exc).__name__, reason=str(exc), timestamp=now(), test=args.test))
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        lock_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
