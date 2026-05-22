"""Shared styling and figure helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figures" / "outputs"
CONDITIONS = ("Pre", "Stim", "Post")
COLORS = {
    "Pre": "#2F5D8C",
    "Stim": "#C4512D",
    "Post": "#3B7A57",
    "neutral": "#6B7280",
    "accent": "#B48A2C",
}
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 180,
    "savefig.dpi": 240,
    "axes.linewidth": 0.8,
}


def setup() -> None:
    plt.rcParams.update(STYLE)


def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{name.lower()}.png"


def save(fig, name: str) -> Path:
    path = output_path(name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def read_per_subject(kind: str = "coupling") -> pd.DataFrame:
    path = ROOT / "data" / "derived" / f"per_subject_{kind}.csv"
    data = pd.read_csv(path)
    data["reliability_flag"] = data["reliability_flag"].fillna("")
    return data


def metric_matrix(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    keep = data.loc[(data["metric"] == metric) & (data["reliability_flag"] == "")]
    return keep.pivot(index="subject_id", columns="phase", values="value").reindex(columns=CONDITIONS)
