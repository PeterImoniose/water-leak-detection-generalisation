"""Shared matplotlib styling so every figure reads as one set."""
import matplotlib as mpl

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# Colour follows the leak class everywhere. No-leak is the neutral baseline.
LEAK_COLORS = {
    "NL": "#52514e",
    "OL": "#2a78d6",
    "LC": "#eb6834",
    "CC": "#1baf7a",
    "GL": "#eda100",
    "BG": "#c3c2b7",
}
# Evaluation protocols, ordered from most to least optimistic.
PROTOCOL_COLORS = {
    "random_window": "#c3c2b7",
    "grouped_condition": "#2a78d6",
    "leave_flow_out": "#eb6834",
    "leave_topology_out": "#1baf7a",
}
SEQ_CMAP = "Blues"


def apply():
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK_2,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.6,
            "font.family": "DejaVu Sans",
        }
    )
