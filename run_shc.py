"""
run_shc.py  --  Spin Hall Conductivity (SHC) calculation
                WannierBerri v26.x compatible

Usage:
    python run_shc.py Pt

Output (written to calc/Pt/04_shc/):
    shc_Pt.dat   -- Energy[eV] vs SHC[(hbar/e)S/cm]
    shc_Pt.png   -- SHC spectrum plot

Required files in calc/Pt/03_wannier/:
    pt.chk, pt.mmn, pt.eig, pt.spn   (needed for SHCqiao)
    Note: pt_hr.dat is NOT needed; System_R.from_wannierdata reads directly from pt.chk

Note:
    Set Efermi_center to the Fermi energy from scf.out:
    grep "Fermi energy" calc/Pt/01_scf/scf.out
"""

import multiprocessing
# Use 'fork' on WSL2; 'forkserver' (Python 3.14 default) crashes WannierBerri's MMN reader
multiprocessing.set_start_method("fork", force=True)

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend for WSL2 (no display)
import matplotlib.pyplot as plt
import wannierberri as wberri


def main():
    # ---------------------------------------------
    # 0. Paths
    # ---------------------------------------------
    if len(sys.argv) < 2:
        sys.exit("Usage: python run_shc.py <Formula>  (e.g. Pt, W)")

    FORMULA  = sys.argv[1]
    SEEDNAME = FORMULA.lower()

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    WAN_DIR  = os.path.join(ROOT_DIR, "calc", FORMULA, "03_wannier")
    OUT_DIR  = os.path.join(ROOT_DIR, "calc", FORMULA, "04_shc")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Required files for SHCqiao: chk, mmn, eig, spn
    # _hr.dat is NOT required by System_R.from_wannierdata
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
    # 1. Load WannierData
    #    v26.x API: WannierData.from_w90_files() -> System_R.from_wannierdata()
    #    Matrices needed for SHCqiao: Ham, AA, SS, SR, SH, SHR
    #    Files required: chk, mmn, eig, spn
    # ---------------------------------------------
    print(f"Loading WannierData: {WAN_DIR}/{SEEDNAME}.*", flush=True)

    wandata = wberri.WannierData.from_w90_files(
        seedname = os.path.join(WAN_DIR, SEEDNAME),
        files    = ["mmn", "eig", "chk", "spn"],
    )

    # ---------------------------------------------
    # 2. Build System_R (SHCqiao mode)
    #    SHCqiao=True: Qiao-type SHC; requires only .spn (no .sHu/.sIu)
    #    SHCryoo=True: Ryoo-type SHC; more accurate but needs .sHu and .sIu
    #                  To enable Ryoo: add write_sHu=.true. write_sIu=.true.
    #                  to pw2wan.in, re-run pw2wannier90.x, then set SHCryoo=True
    # ---------------------------------------------
    print("Building System_R (SHCqiao mode) ...", flush=True)

    system = wberri.System_R.from_wannierdata(
        wandata  = wandata,
        SHCqiao  = True,   # computes SS, SR, SH, SHR matrices from .spn
        berry    = True,   # computes AA matrix (Berry connection)
    )

    # ---------------------------------------------
    # 3. k-point grid
    #    Production run: NK >= 100 (Berry curvature has sharp features)
    #    Test run:       NK = 16
    # ---------------------------------------------
    NK =50   # increase to 50+ for production
    print(f"Setting up {NK}x{NK}x{NK} k-grid ...", flush=True)

    grid = wberri.Grid(system, NK=NK, NKFFT=NK)

    # ---------------------------------------------
    # 4. Fermi energy range
    #    *** Set Efermi_center to the value from scf.out ***
    #    grep "Fermi energy" calc/Pt/01_scf/scf.out
    # ---------------------------------------------
    Efermi_center = 16.9986   # eV  -- from "the Fermi energy is" in scf.out
    Efermi_range  = 3.0       # eV  -- scan from center-range to center+range
    Efermi_npts   = 200

    Efermi_array = np.linspace(
        Efermi_center - Efermi_range,
        Efermi_center + Efermi_range,
        Efermi_npts,
    )

    print(f"Fermi energy: {Efermi_center} eV  "
          f"(scan: {Efermi_array[0]:.2f} to {Efermi_array[-1]:.2f} eV)",
          flush=True)

    # ---------------------------------------------
    # 5. SHC calculator
    #    sigma_{xy}^{z}: spin polarization z, current along x, field along y
    #    Tensor component indices are selected when extracting results (step 7)
    # ---------------------------------------------
    print("Setting up SHC calculator ...", flush=True)

    shc_calculator = wberri.calculators.static.SHC(
        Efermi         = Efermi_array,
        tetra          = False,   # False: Methfessel-Paxton smearing
                                  # True:  tetrahedron method (more accurate, slower)
        kwargs_formula = {"spin_current_type": "qiao"},  # must match SHCqiao=True above
    )

    # ---------------------------------------------
    # 6. Run BZ integration
    # ---------------------------------------------
    print(f"Integrating SHC over {NK}^3 k-points ...", flush=True)

    result = wberri.run(
        system,
        grid          = grid,
        calculators   = {"SHC": shc_calculator},
        adpt_num_iter = 0,      # no adaptive refinement (increase for higher accuracy)
        fout_name     = os.path.join(OUT_DIR, "wberri"),
        restart       = False,
    )

    # ---------------------------------------------
    # 7. Extract results
    #    result.results["SHC"].data shape: (nE, 3, 3, 3)
    #    Axes: (energy, spin, current, field)
    #          spin/current/field indices: 0=x, 1=y, 2=z
    # ---------------------------------------------
    shc_result    = result.results["SHC"]
    shc_data_full = shc_result.data   # shape: (nE, 3, 3, 3)

    # Extract sigma_{xy}^{z}: spin=z(2), current=x(0), field=y(1)
    spin_ax    = 2
    current_ax = 0
    field_ax   = 1
    shc_data = shc_data_full[:, spin_ax, current_ax, field_ax]  # shape: (nE,)

    print(f"Integration done. Tensor shape: {shc_data_full.shape} -> "
          f"sigma_xy^z shape: {shc_data.shape}", flush=True)
    print(f"  SHC at E_F ({Efermi_center:.4f} eV): "
          f"{shc_data[Efermi_npts // 2].real:.2f} (hbar/e) S/cm", flush=True)

    # ---------------------------------------------
    # 8. Save data
    # ---------------------------------------------
    header = (
        f"# Spin Hall Conductivity of {FORMULA}  --  WannierBerri v{wberri.__version__}\n"
        "# sigma_{xy}^{z}  [(hbar/e) S/cm]\n"
        "# SHCqiao method (Qiao et al., PRB 98, 214402 (2018))\n"
        "# Energy[eV]  SHC[(hbar/e)S/cm]"
    )
    np.savetxt(
        OUT_DAT,
        np.column_stack([Efermi_array, shc_data.real]),
        header   = header,
        fmt      = "%.6f\t%.6f",
        comments = "",
    )
    print(f"Data saved: {OUT_DAT}", flush=True)

    # ---------------------------------------------
    # 9. Plot
    # ---------------------------------------------
    print("Plotting ...", flush=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(Efermi_array, shc_data.real,
            color="steelblue", linewidth=1.5, label="SHC (WannierBerri)")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.axvline(Efermi_center, color="red", linewidth=0.8, linestyle=":",
               label=f"$E_F$ = {Efermi_center:.4f} eV")

    ax.set_xlabel("Energy  (eV)", fontsize=13)
    ax.set_ylabel(
        r"$\sigma_{xy}^{z}$  $[(\hbar/e)\,\mathrm{S/cm}]$", fontsize=13)
    ax.set_title(f"Spin Hall Conductivity of {FORMULA}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Plot saved: {OUT_PNG}", flush=True)

    print("\nDone!", flush=True)
    print(f"  Data : {OUT_DAT}")
    print(f"  Plot : {OUT_PNG}")


if __name__ == "__main__":
    main()