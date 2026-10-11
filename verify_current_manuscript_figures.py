"""Reproduce current manuscript Figures 3, 4 and 6 without API calls."""
from pathlib import Path
import json, shutil, subprocess, sys, tempfile
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
NAMES=['Figure3_recorded_parity.png','Figure4_recorded_classification.png','Fig.6_repeatability_boxplots.png']
with tempfile.TemporaryDirectory(prefix='current_figures_', dir=ROOT) as td:
    work=Path(td)
    shutil.copytree(ROOT/'reference_outputs',work/'reference_outputs',ignore=shutil.ignore_patterns('*.png'))
    shutil.copytree(ROOT/'manuscript_figures_v2',work/'manuscript_figures_v2',ignore=shutil.ignore_patterns('__pycache__'))
    (work/'manuscript_figures/reference_images').mkdir(parents=True)
    for name in ['make_manuscript_figures.py','make_repeatability_figures.py']:
        subprocess.run([sys.executable,str(work/'manuscript_figures_v2'/name)],check=True)
    checks={name:bool(np.array_equal(np.asarray(Image.open(ROOT/'manuscript_figures/reference_images'/name).convert('RGBA')),np.asarray(Image.open(work/'manuscript_figures/reference_images'/name).convert('RGBA')))) for name in NAMES}
report={'scope':'Current recorded-data manuscript Figures 3, 4 and 6','pixel_identity':checks,'all_match':all(checks.values())}
(ROOT/'current_manuscript_figure_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
sys.exit(0 if report['all_match'] else 1)

