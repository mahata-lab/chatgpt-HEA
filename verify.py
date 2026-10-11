"""Offline raw-data, scoring-table and exact figure verification."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import tempfile
import shutil
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    report = {"python": sys.version, "packages": {n: metadata.version(n) for n in
              ["numpy", "pandas", "scikit-learn", "matplotlib", "Pillow"]}}
    runs = pd.read_csv(ROOT / "data_rerun/runs.csv")
    raw = [json.loads(line) for line in (ROOT / "data_rerun/raw_responses.jsonl").read_text().splitlines()]
    assert len(raw) == len(runs) == 120
    keys = ["tier", "system", "run"]
    assert not runs.duplicated(keys).any()
    lookup = runs.set_index(keys)
    assert len({tuple(r[k] for k in keys) for r in raw}) == 120
    import config
    for r in raw:
        row = lookup.loc[tuple(r[k] for k in keys)]
        numeric, phase = json.loads(r["numeric_raw"]), json.loads(r["phase_raw"])
        for k in ["dS", "dH", "VEC", "delta"]:
            assert np.isclose(float(numeric[k]), float(r[k]), rtol=0, atol=1e-12)
            assert np.isclose(float(row[k]), float(r[k]), rtol=0, atol=1e-12)
        assert phase["phase"] == r["phase"] == row.phase
        for name, template in [("numeric_prompt", config.NUMERICAL_TEMPLATE), ("phase_prompt", config.PHASE_TEMPLATE)]:
            assert r[name] == template.format(system=r["system"], elements=config.elements_of(r["system"]))
    report["raw_records_verified"] = len(raw)
    with tempfile.TemporaryDirectory(prefix="hea_verify_", dir=ROOT) as td:
        work = Path(td)
        for name in ["config.py", "hea_descriptors.py", "score_predictions.py", "make_figures_v2.py", "reference_phases.csv"]:
            shutil.copy2(ROOT / name, work / name)
        shutil.copytree(ROOT / "data_rerun", work / "data_rerun")
        env = dict(os.environ, MPLBACKEND="Agg")
        for name in ["score_predictions.py", "make_figures_v2.py"]:
            result = subprocess.run([sys.executable, name], cwd=work, env=env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        report["tables"] = {}
        report["images"] = {}
        for reference in sorted((ROOT / "reference_outputs").glob("*.csv")):
            actual = work / "data_rerun" / reference.name
            pd.testing.assert_frame_equal(pd.read_csv(reference), pd.read_csv(actual), check_exact=False, rtol=1e-12, atol=1e-12)
            report["tables"][reference.name] = {"values_match": True, "bytes_match": sha(reference) == sha(actual)}
        for reference in sorted((ROOT / "reference_outputs").glob("*.png")):
            actual = work / "data_rerun" / reference.name
            with Image.open(reference) as old, Image.open(actual) as new:
                a, b = np.array(old.convert("RGBA")), np.array(new.convert("RGBA"))
                same = a.shape == b.shape and np.array_equal(a, b)
                report["images"][reference.name] = {"pixels_match": same, "bytes_match": sha(reference) == sha(actual),
                    "reference_size": old.size, "generated_size": new.size,
                    "reference_sha256": sha(reference), "generated_sha256": sha(actual)}
        report["all_pixels_match"] = all(v["pixels_match"] for v in report["images"].values())
    (ROOT / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["all_pixels_match"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
