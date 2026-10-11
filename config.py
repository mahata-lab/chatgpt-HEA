"""Locked configuration for the one-time HEA benchmark rerun.

IMPORTANT: Do not tune prompts, alloy sets, temperatures, or scoring rules after
looking at the rerun results. If this rerun is used in the paper, the Methods,
Table 4, dates, and repository must be updated to match this file exactly.
"""
from pathlib import Path
import re

MODEL = "gpt-4o-2024-08-06"
TEMPERATURE = 0.2
TOP_P = 1.0
MAX_TOKENS = 1500
N_RUNS = 5
DATA_DIR = Path("data_rerun")

# This is the 24-system set in the files you uploaded. It is NOT the same set
# implied by the current manuscript (which names CoCrFeNiTi and CoCrFeNiMo).
# Treat this as a new, frozen benchmark if you proceed with it.
TIERS = {
    "Simple (Binary)": [
        "NiCr", "NiAl", "FeCr", "CoNi", "CuNi", "AlTi", "MoNb", "TaW",
    ],
    "Complex (Ternary)": [
        "CoCrNi", "FeCrNi", "AlCoNi", "MoNbTa", "TiZrNb", "AlCrFe", "CuNiZn", "VNbTi",
    ],
    "Very Complex (Quinary)": [
        "CoCrFeMnNi", "AlCoCrFeNi", "HfNbTaTiZr", "MoNbTaVW",
        "TiZrNbHfTa", "AlCoCrFeCu", "CoCrFeNiCu", "AlCrFeMnNi",
    ],
}

PHASE_CLASSES = ["FCC", "BCC", "MULTI", "IM"]

# Exact wording used in the revised manuscript Section 2.2.
SYSTEM_INSTRUCTION = (
    "You are a computational materials-science assistant specializing in high-entropy alloys. "
    "Provide technically precise, quantitative answers with units, state assumptions explicitly, "
    "and clearly flag any value that is an estimate rather than a database-referenced quantity."
)

# Keep numerical-property and phase-classification prompts separate.  This avoids
# feeding the model formulas/constants that make some descriptors trivial while
# asking it to recall databases for others.  The prompts are intentionally short
# and identical across systems except for composition.
NUMERICAL_TEMPLATE = """For the equiatomic {system} alloy (equal atomic fractions of {elements}), report estimates of the following standard HEA descriptors using your own scientific knowledge and the standard definitions normally used in the HEA literature:
1. configurational entropy of mixing dS in J mol^-1 K^-1,
2. mixing enthalpy dH in kJ mol^-1,
3. valence electron concentration VEC in electrons/atom,
4. atomic-size mismatch delta in percent.
Do not use external tools or databases during this response. If a value depends on recalled tabulated data, treat it as an estimate.
Return ONLY one JSON object with numeric keys dS, dH, VEC, delta and a short string key notes. Do not include markdown fences."""

PHASE_TEMPLATE = """Predict the single most likely room-temperature equilibrium phase classification of the equiatomic {system} alloy (equal atomic fractions of {elements}). Choose exactly one label from FCC, BCC, MULTI, or IM. Use thermodynamic/size/VEC reasoning and your materials-science knowledge; do not use external tools or databases during this response.
Return ONLY one JSON object with keys phase, confidence (0 to 1), and a short string key notes. Do not include markdown fences."""


def elements_of(system: str) -> str:
    return ", ".join(re.findall(r"[A-Z][a-z]?", system))


def all_systems():
    for tier, systems in TIERS.items():
        for system in systems:
            yield tier, system
