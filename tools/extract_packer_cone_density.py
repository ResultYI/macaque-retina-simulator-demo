"""Extract only fixed PackerConeRc source nodes. Standard library; no model."""
import argparse
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import re

COMMIT = "cf22ac9d681482ca1fcc365a423ef0c2be7916a3"
SHA256 = "72408017cb5b6c4996b48c95050963057f61aab6e39154ea095062a8b5c35e9d"
RELATIVE = "toolbox/+simulator/+load/PackerConeRc.m"
ORIGINAL_REPO = r"D:\retipath\experiments\retipath_foveal_v1\exploratory_v2\vendor_pinned\isetmacaque_cf22ac9d"
PROJECT = Path(__file__).resolve().parents[1]
NUMBER = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?"
BLOCK = re.compile(r"\bd\s*=\s*\[(.*?)\];\s*(\w+)\.eccMM\s*=\s*d\(:,1\);\s*\2\.coneDensityPerSquaredMM\s*=\s*d\(:,2\)\s*\*\s*(1e3)\s*;", re.S)
EXPECTED = {"temporalMeridianPeripheral": 15, "temporalMeridianFoveal": 13,
            "nasalMeridianPeripheral": 19, "nasalMeridianFoveal": 13}
FIELDS = ["record_id", "meridian", "source_segment", "eccentricity_mm",
          "cone_density_cones_per_mm2", "source_eccentricity_literal",
          "source_density_literal", "density_multiplier", "source_line", "evidence_class"]

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def extract(raw):
    if sha(raw) != SHA256:
        raise ValueError("Source SHA256 differs from inspected fixed file; stop for review.")
    text = raw.decode("utf-8")
    blocks = {}
    for match in BLOCK.finditer(text):
        name = match.group(2)
        if name not in EXPECTED or name in blocks:
            raise ValueError("Unexpected or duplicate source matrix.")
        meridian = "temporal" if name.startswith("temporal") else "nasal"
        segment = "foveal" if name.endswith("Foveal") else "peripheral"
        rows = []
        first_line = text.count("\n", 0, match.start(1)) + 1
        for offset, line in enumerate(match.group(1).splitlines()):
            clean = line.replace("...", "").replace(";", "").strip()
            if not clean:
                continue
            values = re.fullmatch(rf"({NUMBER})\s+({NUMBER})", clean)
            if not values:
                raise ValueError(f"Unrecognized table row at line {first_line + offset}.")
            e, density = values.groups()
            if Decimal(e) <= 0 or Decimal(density) <= 0:
                raise ValueError("Nonpositive node; no synthetic repair is allowed.")
            rows.append({"record_id": f"{meridian}_{segment}_{len(rows)+1:03d}",
                         "meridian": meridian, "source_segment": segment,
                         "eccentricity_mm": format(Decimal(e), "f"),
                         "cone_density_cones_per_mm2": format(Decimal(density)*Decimal(match.group(3)), "f"),
                         "source_eccentricity_literal": e, "source_density_literal": density,
                         "density_multiplier": match.group(3), "source_line": first_line+offset,
                         "evidence_class": "published_code_tabulation"})
        if len(rows) != EXPECTED[name]:
            raise ValueError(f"Unexpected matrix count: {name}.")
        blocks[name] = rows
    if set(blocks) != set(EXPECTED):
        raise ValueError("Four expected source matrices were not recovered.")
    # Preserve each source function's foveal + peripheral concatenation.
    rows = []
    for meridian in ("temporal", "nasal"):
        group = blocks[f"{meridian}MeridianFoveal"] + blocks[f"{meridian}MeridianPeripheral"]
        eccentricities = [Decimal(row["eccentricity_mm"]) for row in group]
        if any(a >= b for a, b in zip(eccentricities, eccentricities[1:])):
            raise ValueError("Unexpected source node ordering; no reordering repair allowed.")
        rows.extend(group)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output = stream.getvalue().encode("utf-8")
    manifest = {
        "format_version": "1.0", "evidence_class": "published_code_tabulation",
        "species_scope": "macaque by source naming; species/subspecies and animal details not supplied in this file",
        "source": {"original_repository": ORIGINAL_REPO, "relative_path": RELATIVE,
                   "commit": COMMIT, "sha256": SHA256,
                   "snapshot": "data/source_snapshots/isetmacaque_cf22ac9d/PackerConeRc.m",
                   "commit_binding": "local .git/HEAD checked; whole working-tree cleanliness not verified"},
        "units": {"eccentricity": "mm", "cone_density": "cones/mm^2"},
        "rows": len(rows), "meridians": {},
        "code_locations": {name: {"first_data_line": group[0]["source_line"],
                                  "last_data_line": group[-1]["source_line"], "rows": len(group)}
                           for name, group in blocks.items()},
        "transform": "Exact Decimal interpretation of source literals; density multiplied by source literal 1e3. No interpolation or coordinate conversion.",
        "excluded_conversion": {"source_line": 14, "literal": "210", "unit": "um/deg",
                                "reason": "source model conversion; not established as a universal macaque constant"},
        "csv_sha256": sha(output), "script_sha256": sha(Path(__file__).read_bytes()),
        "checks": {"source_hash_matches": True, "four_matrices_and_counts_match": True,
                   "positive_nodes_and_source_order": True, "rows_preserve_source_line_and_literals": True},
        "validation_meaning": "source extraction consistency only; not physiological validation or dynamic numerical validation",
        "data_role": "reference_only",
        "limitations": ["no e=0 node", "no interpolation, gap filling or extrapolation",
                        "nasal/temporal meridians are not a complete 2-D distribution",
                        "original measurement attribution, digitization method, animal details and uncertainty remain unverified"]}
    for meridian in ("temporal", "nasal"):
        group = [row for row in rows if row["meridian"] == meridian]
        manifest["meridians"][meridian] = {"rows": len(group),
            "min_node_eccentricity_mm": group[0]["eccentricity_mm"],
            "max_node_eccentricity_mm": group[-1]["eccentricity_mm"],
            "coverage": "tabulated nodes only; bounds do not authorize interpolation or extrapolation"}
    return output, manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT/"data/source_snapshots/isetmacaque_cf22ac9d/PackerConeRc.m")
    parser.add_argument("--output-dir", type=Path, default=PROJECT/"data/published_code")
    parser.add_argument("--repo", type=Path, help="Optional original repository HEAD check (read only)")
    parser.add_argument("--verify", action="store_true", help="Compare existing CSV/manifest without writing")
    args = parser.parse_args()
    if args.repo and (args.repo/".git/HEAD").read_text().strip() != COMMIT:
        raise ValueError("Original repository HEAD differs from authorized commit.")
    raw = args.source.read_bytes()
    output, manifest = extract(raw)
    csv_path = args.output_dir/"packer_cone_density.csv"
    manifest_path = args.output_dir/"packer_cone_density.manifest.json"
    if args.verify:
        if csv_path.read_bytes() != output:
            raise ValueError("CSV does not exactly match source extraction.")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Manifest does not match source, script or CSV.")
        print(json.dumps({"mode": "read_only_verify", "rows": manifest["rows"],
                          "csv_sha256": manifest["csv_sha256"], "checks_passed": True,
                          "meaning": manifest["validation_meaning"]}, ensure_ascii=False))
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Refuse to overwrite an existing extraction; changes require review.
        if csv_path.exists() or manifest_path.exists():
            raise FileExistsError("Extraction already exists; use --verify or a reviewed new output directory.")
        csv_path.write_bytes(output)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"rows": manifest["rows"], "csv_sha256": manifest["csv_sha256"]}))

if __name__ == "__main__":
    main()
