# Stochastic PDEs and Mean-Field Particle Systems

Doctoral thesis by Xiaohao Ji — Institut für Mathematik, Freie Universität Berlin, 2026.

**[→ Read the thesis (PDF)](thesis/build/main.pdf)**

## Topics

- Dean–Kawasaki equation and large deviations
- Anderson nonlinear Schrödinger equation and Bose–Einstein condensation
- Langevin dynamics on manifolds (S²)

## Reproducing Figures

```bash
python3 figures.py          # all figures → output/
python3 figures.py fdr      # single figure by name
python3 figures.py --list   # available figures
```

## Building the Thesis

```bash
cd thesis/
make        # → build/main.pdf
make watch  # continuous rebuild
```

Requires TeX Live 2023+ with `latexmk` and `pdflatex`.
