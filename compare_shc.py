"""
compare_shc.py  --  Compare SHC results across all computed materials

Usage:
    python compare_shc.py [Formula1 Formula2 ...]

    # Auto-detect all materials that have a completed shc_*.dat:
    python compare_shc.py

    # Specify explicitly:
    python compare_shc.py Pt W Ta Au

Output (written to 00_plots/):
    shc_comparison.csv     -- Table: Element, E_F, SHC@E_F, SHC_max, SHC_min
    shc_comparison_table.txt  -- Pretty terminal table (also printed to stdout)
    shc_spectra.png        -- Overlaid SHC spectra (energy axis = E - E_F)
    shc_bar.png            -- Bar chart of SHC at E_F

Units: [(hbar/e)(Ohm*cm)^-1]
"""

import os
import re
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(ROOT_DIR, "00_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def find_completed_formulas():
    """Return sorted list of formulas that have a shc_*.dat file."""
    pattern = os.path.join(ROOT_DIR, "calc", "*", "05_shc", "shc_*.dat")
    paths   = sorted(glob.glob(pattern))
    formulas = []
    for p in paths:
        # calc/<Formula>/05_shc/shc_<FORMULA>.dat
        parts = p.replace("\\", "/").split("/")
        idx = parts.index("calc")
        formulas.append(parts[idx + 1])
    return formulas


def read_shc_dat(formula):
    """
    Read shc_<FORMULA>.dat produced by run_shc.py.

    Returns
    -------
    energy  : np.ndarray  [eV]
    shc     : np.ndarray  [(hbar/e)(Ohm*cm)^-1]
    """
    dat_path = os.path.join(
        ROOT_DIR, "calc", formula, "05_shc",
        f"shc_{formula.upper()}.dat"
    )
    if not os.path.exists(dat_path):
        raise FileNotFoundError(dat_path)

    data = np.loadtxt(dat_path, comments="#")
    return data[:, 0], data[:, 1]


def read_fermi_energy(formula):
    """Parse E_Fermi from scf.out."""
    scf_out = os.path.join(ROOT_DIR, "calc", formula, "01_scf", "scf.out")
    if not os.path.exists(scf_out):
        return None
    pattern = re.compile(
        r"the\s+Fermi\s+energy\s+is\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )
    efermi = None
    with open(scf_out) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                efermi = float(m.group(1))
    return efermi


def shc_at_fermi(energy, shc, efermi):
    """Interpolate SHC at efermi (linear interpolation)."""
    return float(np.interp(efermi, energy, shc))


def shc_near_fermi(energy, shc, efermi, window=1.0):
    """
    Return (max, min) of SHC within efermi ± window eV.
    This captures prominent peaks near E_F relevant to experiment.
    """
    mask = np.abs(energy - efermi) <= window
    if not np.any(mask):
        return np.nan, np.nan
    sub = shc[mask]
    return float(np.max(sub)), float(np.min(sub))


# ------------------------------------------------------------------
# Table formatting
# ------------------------------------------------------------------

UNIT = "(ℏ/e)(Ω·cm)⁻¹"

def make_table(rows):
    """
    rows: list of dicts with keys:
        formula, efermi, shc_ef, shc_max, shc_min
    Returns a formatted string table.
    """
    header = (
        f"{'Element':<8}  {'E_F (eV)':>10}  "
        f"{'SHC@E_F':>12}  {'SHC_max±1eV':>13}  {'SHC_min±1eV':>13}"
    )
    divider = "-" * len(header)
    unit_line = (
        f"{'':8}  {'':>10}  "
        f"{'['+UNIT+']':>12}  {'['+UNIT+']':>13}  {'['+UNIT+']':>13}"
    )
    lines = [divider, header, unit_line, divider]
    for r in rows:
        ef  = f"{r['efermi']:.4f}"   if r['efermi']  is not None else "N/A"
        shc = f"{r['shc_ef']:.1f}"   if not np.isnan(r['shc_ef'])  else "N/A"
        mx  = f"{r['shc_max']:.1f}"  if not np.isnan(r['shc_max']) else "N/A"
        mn  = f"{r['shc_min']:.1f}"  if not np.isnan(r['shc_min']) else "N/A"
        lines.append(
            f"{r['formula']:<8}  {ef:>10}  {shc:>12}  {mx:>13}  {mn:>13}"
        )
    lines.append(divider)
    return "\n".join(lines)


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def plot_spectra(datasets, out_path):
    """
    Overlaid SHC spectra.  x-axis = E - E_F so curves are aligned.
    datasets: list of (formula, energy_arr, shc_arr, efermi)
    """
    n = len(datasets)
    colors = cm.tab20(np.linspace(0, 1, max(n, 1)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for (formula, energy, shc, efermi), color in zip(datasets, colors):
        shifted = energy - efermi if efermi is not None else energy
        ax.plot(shifted, shc, linewidth=1.2, label=formula, color=color)

    ax.axhline(0,   color="black", linewidth=0.7, linestyle="--")
    ax.axvline(0,   color="black", linewidth=0.7, linestyle=":",
               label="$E_F$")

    ax.set_xlabel("$E - E_F$  (eV)", fontsize=13)
    ax.set_ylabel(
        r"$\sigma_{xy}^z$  $[(\hbar/e)\,(\Omega\,\mathrm{cm})^{-1}]$",
        fontsize=13
    )
    ax.set_title("Spin Hall Conductivity — Material Comparison", fontsize=14)
    ax.set_xlim(-3, 3)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Spectra plot saved: {out_path}")


def plot_bar(rows, out_path):
    """Bar chart of SHC at E_F, sorted by value."""
    sorted_rows = sorted(rows, key=lambda r: r["shc_ef"]
                         if not np.isnan(r["shc_ef"]) else 0)
    formulas = [r["formula"] for r in sorted_rows]
    values   = [r["shc_ef"]  if not np.isnan(r["shc_ef"]) else 0
                for r in sorted_rows]

    colors = ["steelblue" if v >= 0 else "tomato" for v in values]

    fig, ax = plt.subplots(figsize=(max(7, len(formulas) * 0.7 + 2), 5))
    bars = ax.bar(formulas, values, color=colors, edgecolor="black",
                  linewidth=0.6)

    # Value labels on bars
    for bar, val in zip(bars, values):
        va  = "bottom" if val >= 0 else "top"
        off = 30 if val >= 0 else -30
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + off * np.sign(val) * 0.05,
                f"{val:.0f}", ha="center", va=va, fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(
        r"$\sigma_{xy}^z$  $[(\hbar/e)\,(\Omega\,\mathrm{cm})^{-1}]$",
        fontsize=12
    )
    ax.set_title("SHC at $E_F$ — Material Comparison", fontsize=13)
    ax.set_xlabel("Element", fontsize=12)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Bar chart saved:    {out_path}")


# ------------------------------------------------------------------
# CSV
# ------------------------------------------------------------------

def write_csv(rows, out_path):
    header = "Formula,E_F_eV,SHC_at_EF,SHC_max_1eV,SHC_min_1eV"
    lines  = [header]
    for r in rows:
        ef  = f"{r['efermi']:.4f}"   if r['efermi']  is not None else ""
        shc = f"{r['shc_ef']:.4f}"   if not np.isnan(r['shc_ef'])  else ""
        mx  = f"{r['shc_max']:.4f}"  if not np.isnan(r['shc_max']) else ""
        mn  = f"{r['shc_min']:.4f}"  if not np.isnan(r['shc_min']) else ""
        lines.append(f"{r['formula']},{ef},{shc},{mx},{mn}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"CSV saved:          {out_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    # Determine target formulas
    if len(sys.argv) > 1:
        formulas = sys.argv[1:]
    else:
        formulas = find_completed_formulas()
        if not formulas:
            sys.exit(
                "No completed SHC data found.\n"
                "Run 'python run_shc.py <Formula>' for each material first,\n"
                "or specify formulas explicitly: python compare_shc.py Pt W Ta"
            )
        print(f"Auto-detected {len(formulas)} material(s): {', '.join(formulas)}")

    rows     = []
    datasets = []
    missing  = []

    for formula in formulas:
        try:
            energy, shc = read_shc_dat(formula)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping {formula}")
            missing.append(formula)
            continue

        efermi  = read_fermi_energy(formula)
        shc_ef  = shc_at_fermi(energy, shc, efermi) if efermi is not None else np.nan
        shc_max, shc_min = (
            shc_near_fermi(energy, shc, efermi) if efermi is not None
            else (np.nan, np.nan)
        )

        rows.append({
            "formula" : formula,
            "efermi"  : efermi,
            "shc_ef"  : shc_ef,
            "shc_max" : shc_max,
            "shc_min" : shc_min,
        })
        datasets.append((formula, energy, shc, efermi))

    if not rows:
        sys.exit("No valid SHC data loaded.")

    # ---- Table ----
    table_str = make_table(rows)
    print()
    print(table_str)

    txt_path = os.path.join(PLOTS_DIR, "shc_comparison_table.txt")
    with open(txt_path, "w") as f:
        f.write(f"Spin Hall Conductivity Comparison\n")
        f.write(f"Units: {UNIT}\n")
        f.write(f"sigma_xy^z (SHCqiao, WannierBerri)\n\n")
        f.write(table_str + "\n")
    print(f"\nTable saved:        {txt_path}")

    # ---- CSV ----
    csv_path = os.path.join(PLOTS_DIR, "shc_comparison.csv")
    write_csv(rows, csv_path)

    # ---- Plots ----
    if len(datasets) >= 1:
        plot_spectra(datasets, os.path.join(PLOTS_DIR, "shc_spectra.png"))
    if len(rows) >= 1:
        plot_bar(rows, os.path.join(PLOTS_DIR, "shc_bar.png"))

    if missing:
        print(f"\nSkipped (no data): {', '.join(missing)}")
    print(f"\nDone. All outputs in: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()