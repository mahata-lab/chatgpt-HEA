"""Regenerate the figures used in the technically checked manuscript."""
from pathlib import Path
import hashlib, json, subprocess, sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'paper_figures'
OUT.mkdir(exist_ok=True)
for name in ['make_manuscript_figures.py','make_repeatability_figures.py']:
    subprocess.run([sys.executable,str(ROOT/'manuscript_figures_v2'/name)],check=True)
report={}
for src,dest in [('Figure3_recorded_parity.png','Figure_3.png'),('Figure4_recorded_classification.png','Figure_4.png'),('Fig.6_repeatability_boxplots.png','Figure_6.png')]:
    p=ROOT/'manuscript_figures/reference_images'/src
    target=OUT/dest
    same=bool(np.array_equal(np.asarray(Image.open(p).convert('RGBA')),np.asarray(Image.open(target).convert('RGBA')))) if target.exists() else None
    target.write_bytes(p.read_bytes())
    report[dest]={'matches_previous_verified_paper_image_pixels':same,'dimensions':list(Image.open(p).size),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(OUT/'verification_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if any(v['matches_previous_verified_paper_image_pixels'] is False for v in report.values()):
    raise SystemExit('Regenerated images differ from the previous verified references; inspect the report.')
