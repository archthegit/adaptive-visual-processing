from __future__ import annotations

from pathlib import Path

import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to use Experiment 1 plotting utilities.") from exc
    return plt


def plot_frame_relevance_by_layer(frame_scores: np.ndarray, path: str | Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(frame_scores, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Frame/time grid")
    ax.set_ylabel("Decoder layer")
    fig.colorbar(im, ax=ax, label="relevance")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cumulative_frame_relevance(cumulative_curve: np.ndarray, path: str | Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.linspace(0, 100, len(cumulative_curve))
    ax.plot(x, cumulative_curve)
    ax.set_xlabel("Frames ranked by relevance (%)")
    ax.set_ylabel("Cumulative relevance mass")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_layer_concentration(topk_mass: np.ndarray, entropy: np.ndarray, path: str | Path) -> None:
    plt = _pyplot()
    fig, ax1 = plt.subplots(figsize=(7, 4))
    layers = np.arange(len(topk_mass))
    ax1.plot(layers, topk_mass, label="top-k mass")
    ax1.set_xlabel("Decoder layer")
    ax1.set_ylabel("Top-k frame mass")
    ax2 = ax1.twinx()
    ax2.plot(layers, entropy, color="tab:orange", label="entropy")
    ax2.set_ylabel("Normalized entropy")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
