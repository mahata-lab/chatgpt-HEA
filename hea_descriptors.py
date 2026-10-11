"""Closed-form HEA descriptors used as the reference benchmark.

The pair table is interpreted as OMEGA_ij = 4 * DeltaH_mix(equiatomic i-j),
so the multicomponent regular-solution expression is
    DeltaH_mix = sum_{i<j} OMEGA_ij * c_i * c_j.
Missing pair data is an ERROR, not silently treated as ideal.
"""
import re
import numpy as np

R_GAS = 8.314462618

ELEM = {
    "Al": dict(r=1.432, vec=3), "Ti": dict(r=1.462, vec=4),
    "V": dict(r=1.316, vec=5), "Cr": dict(r=1.249, vec=6),
    "Mn": dict(r=1.350, vec=7), "Fe": dict(r=1.241, vec=8),
    "Co": dict(r=1.251, vec=9), "Ni": dict(r=1.246, vec=10),
    "Cu": dict(r=1.278, vec=11), "Zn": dict(r=1.394, vec=12),
    "Zr": dict(r=1.603, vec=4), "Nb": dict(r=1.429, vec=5),
    "Mo": dict(r=1.363, vec=6), "Hf": dict(r=1.578, vec=4),
    "Ta": dict(r=1.430, vec=5), "W": dict(r=1.367, vec=6),
}

OMEGA = {
    ("Al", "Ni"): -88, ("Al", "Co"): -76, ("Al", "Fe"): -44,
    ("Al", "Cr"): -40, ("Al", "Ti"): -120, ("Al", "Cu"): -4,
    ("Al", "Mn"): -76, ("Al", "V"): -64, ("Al", "Zr"): -176,
    ("Al", "Nb"): -72, ("Al", "Mo"): -20, ("Al", "Ta"): -76,
    ("Al", "Hf"): -156, ("Al", "W"): -8, ("Al", "Zn"): 4,
    ("Ti", "Cr"): -28, ("Ti", "Fe"): -68, ("Ti", "Co"): -112,
    ("Ti", "Ni"): -140, ("Ti", "V"): -8, ("Ti", "Nb"): 8,
    ("Ti", "Mo"): -16, ("Ti", "Ta"): 4, ("Ti", "Zr"): 0,
    ("Ti", "Cu"): -36, ("Ti", "Hf"): 0, ("Ti", "W"): -24,
    ("V", "Cr"): -8, ("V", "Fe"): -28, ("V", "Co"): -56,
    ("V", "Ni"): -72, ("V", "Nb"): -4, ("V", "Mo"): 0,
    ("V", "Ta"): -4, ("V", "W"): -4, ("V", "Cu"): 20, ("V", "Zr"): -16,
    ("Cr", "Fe"): -4, ("Cr", "Co"): -16, ("Cr", "Ni"): -28,
    ("Cr", "Cu"): 48, ("Cr", "Mn"): 8, ("Cr", "Mo"): 0,
    ("Cr", "Nb"): -28, ("Cr", "Ta"): -28, ("Cr", "W"): 4, ("Cr", "Zr"): -48,
    ("Mn", "Fe"): 0, ("Mn", "Co"): -20, ("Mn", "Ni"): -32, ("Mn", "Cu"): 16,
    ("Fe", "Co"): -4, ("Fe", "Ni"): -8, ("Fe", "Cu"): 52,
    ("Fe", "Mo"): -8, ("Fe", "Nb"): -64, ("Fe", "Ta"): -60, ("Fe", "Zr"): -100,
    ("Co", "Ni"): 0, ("Co", "Cu"): 24, ("Co", "Mo"): -20,
    ("Co", "Nb"): -100, ("Co", "Ta"): -96, ("Co", "Zr"): -164,
    ("Ni", "Cu"): 16, ("Ni", "Zn"): -36, ("Ni", "Mo"): -28, ("Ni", "Nb"): -120,
    ("Ni", "Ta"): -116, ("Ni", "Zr"): -196, ("Ni", "W"): -12,
    ("Cu", "Zn"): -24,
    ("Nb", "Ta"): 0, ("Nb", "Mo"): -24, ("Nb", "W"): -32, ("Nb", "Zr"): 16,
    ("Nb", "Hf"): 16, ("Mo", "Ta"): -20, ("Mo", "W"): 0, ("Mo", "Hf"): -16,
    ("Ta", "W"): -28, ("Ta", "Hf"): 12, ("W", "Hf"): -24,
    ("Hf", "Zr"): 0, ("Zr", "Ta"): 12, ("Zr", "Mo"): -24, ("Zr", "W"): -36,
    ("Hf", "Ti"): 0,
}


def parse(system):
    syms = re.findall(r"[A-Z][a-z]?", system)
    if not syms:
        raise ValueError(f"Could not parse system {system!r}")
    unknown = [s for s in syms if s not in ELEM]
    if unknown:
        raise KeyError(f"Missing elemental data for {unknown} in {system}")
    n = len(syms)
    return syms, {s: 1.0 / n for s in syms}


def omega(a, b):
    if a == b:
        return 0.0
    if (a, b) in OMEGA:
        return float(OMEGA[(a, b)])
    if (b, a) in OMEGA:
        return float(OMEGA[(b, a)])
    raise KeyError(f"No OMEGA pair value for {a}-{b}; do not silently assume zero")


def validate_pairs(system):
    syms, _ = parse(system)
    missing = []
    for i, a in enumerate(syms):
        for b in syms[i+1:]:
            try:
                omega(a, b)
            except KeyError:
                missing.append(f"{a}-{b}")
    return missing


def delta_S_mix(system):
    _, c = parse(system)
    return -R_GAS * sum(x * np.log(x) for x in c.values())


def delta_H_mix(system):
    syms, c = parse(system)
    H = 0.0
    for i, a in enumerate(syms):
        for b in syms[i+1:]:
            H += omega(a, b) * c[a] * c[b]
    return H


def vec(system):
    _, c = parse(system)
    return sum(x * ELEM[s]["vec"] for s, x in c.items())


def delta_size(system):
    _, c = parse(system)
    rbar = sum(x * ELEM[s]["r"] for s, x in c.items())
    return 100.0 * np.sqrt(sum(x * (1.0 - ELEM[s]["r"] / rbar) ** 2 for s, x in c.items()))


def all_descriptors(system):
    return {
        "dS": float(delta_S_mix(system)),
        "dH": float(delta_H_mix(system)),
        "VEC": float(vec(system)),
        "delta": float(delta_size(system)),
    }
