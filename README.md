# HEA benchmark: verified offline reproduction

## Current revised manuscript

Run `python verify_current_manuscript_figures.py` after installing the pinned
dependencies to reproduce the current recorded-data Figures 3, 4 and 6.
This check compares every RGBA pixel with the final manuscript references;
results are saved in `current_manuscript_figure_checks.json`. The producers
are in `manuscript_figures_v2/`, and final images are in
`manuscript_figures/reference_images/`. Figure 5 is a conceptual illustration.

The older standalone phase images in `reference_outputs/` were manually
annotated with panel labels. Their scoring values still match, but those two
annotated images fail the original `verify.py` pixel comparison. The current
combined Figure 4 is checked by the new command above.

## Historical image provenance

For the manuscript images `image_p57.png` and `image_p86.png`, run
`python verify_manuscript.py`. Their exact original pipeline is in
`manuscript_figures/`; see its README and `manuscript_verification_report.json`.

The separate recorded-data pipeline reproduces the four `figure_*.png` images supplied in the working
folder from the recorded August 18, 2026 dataset. It requires no API key or API
calls. Scientific source files are unchanged copies of the supplied pipeline.

## Run

Use Python 3.12, create a virtual environment, and run from this folder:

```sh
python -m pip install -r requirements.txt
python verify.py
```

`verify.py` checks the 120 paired records (240 recorded responses), numerical
JSON values, phase labels, instantiated prompts, all eight scoring tables, and
every RGBA pixel of the four images. It runs scoring and plotting in an isolated
temporary folder, leaves archived reference outputs untouched, writes
`verification_report.json`, and exits unsuccessfully if an image differs.

To retain newly generated outputs instead:

```sh
python score_predictions.py
python make_figures_v2.py
```

Outputs are written to `data_rerun/`. Matplotlib 3.10.5 is pinned to the version
embedded in the supplied PNGs. Exact rendering was verified on Windows with
Python 3.12.14; other platforms or font/rendering libraries can affect pixels.

## Image identities

| Reference image | What it shows | Producer |
| --- | --- | --- |
| `figure_phase_confusion_matrix.png` | Assigned reference phase versus majority predicted phase, including UNK column | `plot_phase_confusion_matrix()` |
| `figure_phase_performance.png` | Accuracy and macro-F1 for binary, ternary and quinary cases | `plot_phase_performance()` |
| `figure_reproducibility_dH_sd.png` | Mean within-system sample SD of mixing enthalpy over five runs | `plot_dh_reproducibility()` |
| `figure_phase_agreement.png` | Majority phase agreement across five runs | `plot_phase_agreement()` |

All four functions are in `make_figures_v2.py`. The original four-image verification passed before manual panel annotations
were added to the two phase images. The current verification report records
those image differences; the current manuscript check above passes.
Their encoded PNG bytes differ; pixel identity is not byte identity.
All eight CSV tables match within 1e-12 numerical tolerance; CSV bytes differ.
See `verification_report.json` for individual comparisons and image hashes.

## Included files

- `config.py`: case groups, exact prompts and original collection settings.
- `hea_descriptors.py`: formulas and adopted elemental/pair constants.
- `reference_phases.csv`: assigned labels and original reference notes.
- `score_predictions.py`: offline aggregation and metric calculation.
- `make_figures_v2.py`: the exact producer of the four supplied figures.
- `data_rerun/`: original runs, raw responses and collection manifest.
- `reference_outputs/`: original eight derived CSVs and four PNGs used for verification.
- `verify.py`, `requirements.txt`, `verification_report.json`, `SHA256SUMS.txt`:
  reproducibility checks, verified dependency versions and integrity information.

## Manuscript image reproduction

The parent folder contains the following manuscript assets, which are not outputs
of this package's plotting script:

| File | Content |
| --- | --- |
| `image_p14.png` | Conceptual HEA properties, characterization, methods and applications |
| `image_p31.png` | Conceptual evaluation workflow |
| `image_p52.png` | Older four-panel numerical parity plot |
| `image_p57.png` | Older combined confusion matrix and classification bars |
| `image_p82.png` | Conceptual autonomy versus human oversight diagram |
| `image_p86.png` | Older execution-validity, reproducibility and expert-score panels |

The three older scientific panels are now included under `manuscript_figures/`,
with their original plotting script, CSV inputs and source evaluation code.
Run `python verify_manuscript.py` from this root to check Figures 4, 5 and 6
against `image_p52.png`, `image_p57.png` and `image_p86.png`. These references
are unchanged copies of the supplied manuscript images.

The older scientific panels correspond to a separate supplementary pipeline.
Its `evaluate.py` generates numerical predictions with seeded random noise,
uses literal simulation convergence counts, and generates expert ratings with
random draws. Those panels are not evidence of measured model predictions,
executed simulations or independently collected expert ratings. They must not
be substituted for the four verified recorded-data figures.

## Interpretation

The dataset has 24 named cases but 23 distinct compositions: HfNbTaTiZr and
TiZrNbHfTa are identical compositions. All original cases are preserved.
Phase metrics measure agreement with assigned labels; this verification does
not validate those labels against materials literature. Descriptor errors are
relative to the adopted constants. Repeated agreement measures consistency,
not correctness. No manuscript, document-building helper, conceptual image,
API collector, credentials or environment is included. The supplementary synthetic
evaluation is retained separately for manuscript image provenance.

This folder is prepared for a GitHub push; no push has been performed. No
license has been assigned on the authors' behalf.
