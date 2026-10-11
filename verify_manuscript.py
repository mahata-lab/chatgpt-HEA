"""Regenerate the supplied manuscript images from their original CSV inputs."""
from pathlib import Path
import tempfile
import shutil
import subprocess
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'manuscript_figures'

def main():
    report = {'source': 'Original supplementary plotting script and saved CSVs', 'images': {}}
    with tempfile.TemporaryDirectory(prefix='hea_verify_', dir=ROOT) as td:
        work = Path(td)
        for f in SOURCE.iterdir():
            if f.is_file() and f.suffix in {'.py', '.csv'}:
                shutil.copy2(f, work / f.name)
        result = subprocess.run([sys.executable, 'evaluate.py'], cwd=work,
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        report['evaluation_tables'] = {}
        for reference in sorted(SOURCE.glob('*.csv')):
            pd.testing.assert_frame_equal(pd.read_csv(reference), pd.read_csv(work / reference.name),
                                          check_exact=False, rtol=1e-12, atol=1e-12)
            report['evaluation_tables'][reference.name] = True
        result = subprocess.run([sys.executable, 'make_figures.py'], cwd=work,
                                env=dict(os.environ, MPLBACKEND='Agg'), capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        for ref in sorted((SOURCE / 'reference_images').glob('*.png')):
            actual = work / ref.name
            with Image.open(ref) as a, Image.open(actual) as b:
                x, y = np.array(a.convert('RGBA')), np.array(b.convert('RGBA'))
                report['images'][ref.name] = {'pixels_match': x.shape == y.shape and np.array_equal(x, y),
                    'reference_size': a.size, 'generated_size': b.size,
                    'reference_sha256': hashlib.sha256(ref.read_bytes()).hexdigest(),
                    'generated_sha256': hashlib.sha256(actual.read_bytes()).hexdigest()}
        report['all_pixels_match'] = all(v['pixels_match'] for v in report['images'].values())
    (ROOT / 'manuscript_verification_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['all_pixels_match'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
