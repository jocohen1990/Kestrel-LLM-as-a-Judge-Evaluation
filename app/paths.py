"""Central place for folder locations.

Building paths from this file's location (instead of relative strings like
"data/corpus") means the code works no matter which folder you run it from.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
GOLDEN_DIR = ROOT / "data" / "golden"
ATTACKS_DIR = ROOT / "data" / "attacks"
ROBUSTNESS_DIR = ROOT / "data" / "robustness"
BIAS_DIR = ROOT / "data" / "bias"
RESULTS_DIR = ROOT / "results"
