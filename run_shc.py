"""
run_shc.py  --  Spin Hall Conductivity (SHC) calculation
                WannierBerri v26.x compatible

Usage:
    python run_shc.py Pt

Output (written to calc/Pt/04_shc/):
    shc_Pt.dat   -- Energy[eV] vs SHC[(hbar/e)(Ohm*cm)^-1]
    shc_Pt.png   -- SHC spectrum plot

Required files in calc/Pt/03_wannier/:
    pt.chk, pt.mmn, pt.eig, pt.spn   (needed for SHCqiao)

Unit note:
    WannierBerri outputs SHC in SI units [S/m].
    To convert to the conventional (hbar/e)(Ohm*cm)^-1 used in literature:
        SHC [(hbar/e)(Ohm*cm)^-1] = SHC [S/m] / 100
    Reference: WannierBerri GitHub issue #274
"""

import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wannierberri as wberri
import ray


# ------------------------------------------------------------------
# Unit conversion: WannierBerri outputs in S/m (SI).
# Literature uses (hbar/e)(Omega*cm)^-1 = (hbar/e) S/cm.
# 1 S/m = 0.01 S/cm  =>  divide by 100 to convert.
# ------------------------------------------------------------------
WBERRI_TO_HBAR_E_SCMINV = 1.0 / 100.0   # S/m  ->  (hbar/e)(Omega cm)^-1


def read_fermi_energy(scf_out_path: str) -> float:
    """
    Parse the Fermi energy from a Quantum ESPRESSO scf.out file.
    Looks for lines like:
        the Fermi energy is    16.9986 ev
    Returns the last matching value (eV).
    Raises FileNotFoundError or RuntimeError if not found.
    """
    if not os.path.exists(scf_out_path):
        raise FileNotFoundError(f"scf.out not found: {scf_out_path}")

    pattern = re.compile(
        r"the\s+Fermi\s+energy\s+is\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )
    efermi = None
    with open(scf_out_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                efermi = float(m.group(1))   # keep last occurrence

    if efermi is None:
        raise RuntimeError(
            f"Could not find 'the Fermi energy is ...' in {scf_out_path}\n"
            "  Run the SCF step first, or check the output file."
        )
    return efermi


def main():
    # ---------------------------------------------
    # 0. Paths
    # ---------------------------------------------
    if len(sys.argv) < 2:
        sys.exit("Usage: python run_shc.py <Formula>  (e.g. Pt, W)")

    FORMULA  = sys.argv[1]
    SEEDNAME = FORMULA.lower()

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    SCF_DIR  = os.path.join(ROOT_DIR, "calc", FORMULA, "01_scf")
    WAN_DIR  = os.path.join(ROOT_DIR, "calc", FORMULA, "03_wannier")
    OUT_DIR  = os.path.join(ROOT_DIR, "calc", FORMULA, "04_shc")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Required files for SHCqiao
    required_files = {
        "chk": f"{SEEDNAME}.chk",
        "mmn": f"{SEEDNAME}.mmn",
        "eig": f"{SEEDNAME}.eig",
        "spn": f"{SEEDNAME}.spn",
    }
    missing = [
        fname for fname in required_files.values()
        if not os.path.exists(os.path.join(WAN_DIR, fname))
    ]
    if missing:
        sys.exit(
            f"ERROR: missing files: {missing}\n"
            f"  Directory: {WAN_DIR}\n"
            f"  Run run_pipeline.py first."
        )

    OUT_DAT = os.path.join(OUT_DIR, f"shc_{FORMULA.upper()}.dat")
    OUT_PNG = os.path.join(OUT_DIR, f"shc_{FORMULA.upper()}.png")

    # ---------------------------------------------
    # 1. Read Fermi energy automatically from scf.out
    # ---------------------------------------------
    scf_out = os.path.join(SCF_DIR, "scf.out")
    print(f"Reading Fermi energy from: {scf_out}", flush=True)
    Efermi_center = read_fermi_energy(scf_out)
    print(f"  Fermi energy: {Efermi_center:.4f} eV", flush=True)

    # ---------------------------------------------
    # 2. Load WannierData
    # ---------------------------------------------
    print(f"Loading WannierData: {WAN_DIR}/{SEEDNAME}.*", flush=True)
    wandata = wberri.WannierData.from_w90_files(
        seedname = os.path.join(WAN_DIR, SEEDNAME),
        files    = ["mmn", "eig", "chk", "spn"],
    )

    # ---------------------------------------------
    # 3. Build System_R (SHCqiao mode)
    # ---------------------------------------------
    print("Building System_R (SHCqiao mode) ...", flush=True)
    system = wberri.System_R.from_wannierdata(
        wandata  = wandata,
        SHCqiao  = True,
        berry    = True,
    )

    # ---------------------------------------------
    # 4. k-point grid
    #    NK=100 is recommended for production; NK=50 for testing.
    #    Berry curvature has sharp features near avoided crossings
    #    that require dense k-sampling to resolve correctly.
    # ---------------------------------------------
    NK    = 100   # increase to 150-200 for publication-quality results
    NKFFT = 10
    print(f"Setting up k-grid: NK={NK}, NKFFT={NKFFT} ...", flush=True)
    grid = wberri.Grid(system, NK=NK, NKFFT=NKFFT)

    # ---------------------------------------------
    # 5. Fermi energy scan
    # ---------------------------------------------
    Efermi_range = 3.0
    Efermi_npts  = 200
    Efermi_array = np.linspace(
        Efermi_center - Efermi_range,
        Efermi_center + Efermi_range,
        Efermi_npts,
    )
    print(f"Fermi energy scan: {Efermi_array[0]:.2f} to {Efermi_array[-1]:.2f} eV "
          f"({Efermi_npts} points)", flush=True)

    # ---------------------------------------------
    # 6. SHC calculator
    #    sigma_{xy}^{z}: spin=z, current=x, field=y
    # ---------------------------------------------
    print("Setting up SHC calculator ...", flush=True)
    shc_calculator = wberri.calculators.static.SHC(
        Efermi         = Efermi_array,
        tetra          = False,
        kwargs_formula = {"spin_current_type": "qiao"},
    )

    # ---------------------------------------------
    # 7. Run BZ integration
    # ---------------------------------------------
    if not ray.is_initialized():
    ray.init(num_cpus=multiprocessing.cpu_count())
    print(f"Ray initialized: {multiprocessing.cpu_count()} CPUs", flush=True)

    print(f"Integrating SHC over {NK}^3 k-points ...", flush=True)
    result = wberri.run(
        system,
        grid          = grid,
        calculators   = {"SHC": shc_calculator},
        adpt_num_iter = 0,
        fout_name     = os.path.join(OUT_DIR, "wberri"),
        restart       = False,
        parallel      = wberri.Parallel(method="ray"),
    )

    # ---------------------------------------------
    # 8. Extract results and apply unit conversion
    #    WannierBerri result is in SI: S/m
    #    Convert to (hbar/e)(Omega*cm)^-1 by dividing by 100
    #
    #    Tensor axes: (energy, spin, current, field)
    #    sigma_{xy}^{z}: spin=z(2), current=x(0), field=y(1)
    # ---------------------------------------------
    shc_result    = result.results["SHC"]
    shc_data_full = shc_result.data       # shape: (nE, 3, 3, 3)  [S/m]

    spin_ax    = 2   # z
    current_ax = 0   # x
    field_ax   = 1   # y
    shc_si = shc_data_full[:, spin_ax, current_ax, field_ax].real   # [S/m]

    # Unit conversion: S/m -> (hbar/e)(Omega*cm)^-1
    shc_data = shc_si * WBERRI_TO_HBAR_E_SCMINV   # [(hbar/e)(Omega*cm)^-1]

    ef_idx = Efermi_npts // 2
    print(f"Integration done. Raw tensor shape: {shc_data_full.shape}", flush=True)
    print(f"  SHC at E_F = {Efermi_center:.4f} eV:", flush=True)
    print(f"    Raw (S/m)                  : {shc_si[ef_idx]:.2f}", flush=True)
    print(f"    Converted [(hbar/e)(Ocm)^-1]: {shc_data[ef_idx]:.2f}", flush=True)
    print(f"    (Expected for Pt ~ +2000)  ", flush=True)

    # ---------------------------------------------
    # 9. Save data
    # ---------------------------------------------
    header = (
        f"# Spin Hall Conductivity of {FORMULA}  --  WannierBerri v{wberri.__version__}\n"
        "# sigma_{xy}^{z}  [(hbar/e)(Ohm*cm)^-1]\n"
        "# SHCqiao method (Qiao et al., PRB 98, 214402 (2018))\n"
        "# Unit conversion: WannierBerri [S/m] / 100 = [(hbar/e)(Ohm*cm)^-1]\n"
        "# Energy[eV]  SHC[(hbar/e)(Ohm*cm)^-1]"
    )
    np.savetxt(
        OUT_DAT,
        np.column_stack([Efermi_array, shc_data]),
        header   = header,
        fmt      = "%.6f\t%.6f",
        comments = "",
    )
    print(f"Data saved: {OUT_DAT}", flush=True)

    # ---------------------------------------------
    # 10. Plot
    # ---------------------------------------------
    print("Plotting ...", flush=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Efermi_array, shc_data,
            color="steelblue", linewidth=1.5, label="SHC (WannierBerri)")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.axvline(Efermi_center, color="red", linewidth=0.8, linestyle=":",
               label=f"$E_F$ = {Efermi_center:.4f} eV")

    ax.set_xlabel("Energy  (eV)", fontsize=13)
    ax.set_ylabel(
        r"$\sigma_{xy}^{z}$  $[(\hbar/e)\,(\Omega\,\mathrm{cm})^{-1}]$",
        fontsize=13)
    ax.set_title(f"Spin Hall Conductivity of {FORMULA}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Plot saved: {OUT_PNG}", flush=True)
    ray.shutdown()
    print("\nDone!", flush=True)
    print(f"  Data : {OUT_DAT}")
    print(f"  Plot : {OUT_PNG}")


if __name__ == "__main__":
    main()