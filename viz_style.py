import matplotlib

COLORS = {
    "fuBlue":     "#003366",
    "citeViolet": "#d9b7fa",
    "fu_green":   "#CCFF00",   # RGB (204, 255, 0)
}

_PGF_EXTRA = {
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "axes.unicode_minus": False,
    "pgf.preamble": (
        r"\usepackage[T1]{fontenc}"
        r"\usepackage{libertine}"
        r"\usepackage{amsmath,amssymb,dsfonds,mathrsfs,mathtools}"
    ),
}


def apply_thesis_style(backend=None):
    """
    Apply thesis-wide matplotlib style.

    Call BEFORE importing matplotlib.pyplot anywhere in the calling module.
    backend: None  — keep current backend (interactive / Agg default)
             "pdf" — non-interactive PDF output (for figures.py)
             "pgf" — LaTeX PGF output (for direct \\input{} in thesis)

    Returns the color dict {"fuBlue": ..., "citeViolet": ..., "fu_green": ...}.
    """
    if backend is not None:
        matplotlib.use(backend)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Linux Libertine"],
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (5.8, 3.3),   # matches thesis textwidth
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    if backend == "pgf":
        plt.rcParams.update(_PGF_EXTRA)

    return dict(COLORS)
