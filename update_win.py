"""
update_win.py  --  Update Wannier90 .win energy windows after SCF

Usage:
    python update_win.py <Formula> [--froz-offset FLOAT] [--win-margin FLOAT]

This script reads the Fermi energy from 01_scf/scf.out and the highest
band energy from 02_nscf/nscf.out (or falls back to a conservative estimate),
then patches dis_froz_max and dis_win_max in 03_wannier/<formula>.win.

Parameters
----------
--froz-offset : float, default 2.0
    dis_froz_max = E_Fermi + froz_offset  (eV)
--win-margin  : float, default 2.0
    dis_win_max  = E_band_max - win_margin (eV)
    where E_band_max is the energy of the highest band across all k-points
    found in nscf.out.  If nscf.out is absent, dis_win_max is left unchanged.

Example
-------
    # Run SCF first, then:
    python update_win.py Pt
    python update_win.py Pt --froz-offset 3.0 --win-margin 1.0
"""

import os
import re
import sys
import shutil
import argparse


# =======================================================================
# Parsers
# =======================================================================

def read_fermi_energy(scf_out_path: str) -> float:
    """
    Parse the Fermi energy from QE scf.out.
    QE prints: '    the Fermi energy is    12.3456 ev'
    Returns the last occurrence (final SCF iteration).
    """
    if not os.path.exists(scf_out_path):
        raise FileNotFoundError(f"scf.out not found: {scf_out_path}")

    pattern = re.compile(
        r"the\s+Fermi\s+energy\s+is\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )
    efermi = None
    with open(scf_out_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                efermi = float(m.group(1))   # keep updating to get the last value

    if efermi is None:
        raise RuntimeError(
            f"Fermi energy not found in {scf_out_path}.\n"
            "Make sure the SCF run completed successfully."
        )
    return efermi


def read_highest_band_energy(nscf_out_path: str) -> float | None:
    """
    Parse the highest Kohn-Sham eigenvalue across all k-points from nscf.out.

    QE prints eigenvalue blocks like:
          k = 0.0000 0.0000 0.0000 ( ... PWs)
          bands (ev):
             -5.123   1.234  ...  42.567

    Returns the maximum energy found, or None if the file is absent or
    no eigenvalue lines are detected.
    """
    if not os.path.exists(nscf_out_path):
        return None

    max_energy = None
    in_bands_block = False

    with open(nscf_out_path) as f:
        for line in f:
            if re.search(r"bands\s*\(ev\)", line, re.IGNORECASE):
                in_bands_block = True
                continue
            if in_bands_block:
                stripped = line.strip()
                if not stripped:
                    in_bands_block = False
                    continue
                try:
                    vals = [float(x) for x in stripped.split()]
                    candidate = max(vals)
                    if max_energy is None or candidate > max_energy:
                        max_energy = candidate
                except ValueError:
                    in_bands_block = False

    return max_energy


# =======================================================================
# .win patcher
# =======================================================================

def patch_win(win_path: str, dis_froz_max: float, dis_win_max: float) -> None:
    """
    Overwrite dis_froz_max and dis_win_max in an existing .win file.
    Creates a backup at <win_path>.bak before modifying.
    """
    if not os.path.exists(win_path):
        raise FileNotFoundError(f".win file not found: {win_path}")

    # Backup
    bak_path = win_path + ".bak"
    shutil.copy2(win_path, bak_path)
    print(f"  Backup written: {bak_path}")

    with open(win_path) as f:
        lines = f.readlines()

    patched = []
    for line in lines:
        if re.match(r"^\s*dis_froz_max\s*=", line, re.IGNORECASE):
            patched.append(f"dis_froz_max  = {dis_froz_max:.4f}\n")
        elif re.match(r"^\s*dis_win_max\s*=", line, re.IGNORECASE):
            patched.append(f"dis_win_max   = {dis_win_max:.4f}\n")
        else:
            patched.append(line)

    with open(win_path, "w") as f:
        f.writelines(patched)


# =======================================================================
# Main
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Update Wannier90 .win energy windows after QE SCF."
    )
    parser.add_argument("formula", help="Chemical formula (e.g. Pt, GaAs, Bi2Se3)")
    parser.add_argument(
        "--froz-offset", type=float, default=2.0,
        help="dis_froz_max = E_Fermi + froz_offset  [eV] (default: 2.0)"
    )
    parser.add_argument(
        "--win-margin", type=float, default=2.0,
        help="dis_win_max = E_band_max - win_margin  [eV] (default: 2.0)"
    )
    args = parser.parse_args()

    formula = args.formula
    base_dir  = os.path.abspath(f"./calc/{formula}")
    scf_out   = os.path.join(base_dir, "01_scf",    "scf.out")
    nscf_out  = os.path.join(base_dir, "02_nscf",   "nscf.out")
    win_path  = os.path.join(base_dir, "03_wannier", f"{formula.lower()}.win")

    print(f"\n Updating energy windows for {formula}")
    print(f"  .win file : {win_path}")

    # ---- Fermi energy ----
    efermi = read_fermi_energy(scf_out)
    print(f"  E_Fermi          = {efermi:.4f} eV  (from scf.out)")

    dis_froz_max = efermi + args.froz_offset
    print(f"  dis_froz_max     = {efermi:.4f} + {args.froz_offset} = {dis_froz_max:.4f} eV")

    # ---- Highest band energy ----
    e_band_max = read_highest_band_energy(nscf_out)
    if e_band_max is not None:
        dis_win_max = e_band_max - args.win_margin
        print(f"  E_band_max       = {e_band_max:.4f} eV  (from nscf.out)")
        print(f"  dis_win_max      = {e_band_max:.4f} - {args.win_margin} = {dis_win_max:.4f} eV")
    else:
        # nscf.out not yet available; read the current value from .win and keep it
        dis_win_max = None
        print(f"  nscf.out not found — dis_win_max will not be changed.")
        print(f"  Re-run after NSCF completes to set dis_win_max automatically.")

    # ---- Sanity check ----
    if dis_win_max is not None and dis_win_max <= dis_froz_max:
        print(f"\n  WARNING: dis_win_max ({dis_win_max:.4f}) <= dis_froz_max ({dis_froz_max:.4f}).")
        print(f"  Increasing dis_win_max to dis_froz_max + 5.0 eV as a safety margin.")
        dis_win_max = dis_froz_max + 5.0

    # ---- Read current dis_win_max from .win if not updating it ----
    if dis_win_max is None:
        with open(win_path) as f:
            for line in f:
                m = re.match(r"^\s*dis_win_max\s*=\s*([-+]?\d+\.?\d*)", line, re.IGNORECASE)
                if m:
                    dis_win_max = float(m.group(1))
                    break
        if dis_win_max is None:
            dis_win_max = dis_froz_max + 10.0
            print(f"  Could not read existing dis_win_max; defaulting to {dis_win_max:.4f} eV.")

    # ---- Patch .win ----
    patch_win(win_path, dis_froz_max, dis_win_max)
    print(f"\n  Patched {win_path}")
    print(f"    dis_froz_max = {dis_froz_max:.4f} eV")
    print(f"    dis_win_max  = {dis_win_max:.4f} eV")
    print(f"\n  Next step: run Wannier90 in 03_wannier/")
    print(f"    cd calc/{formula}/03_wannier")
    print(f"    wannier90.x -pp {formula.lower()}")
    print(f"    pw.x < ../02_nscf/nscf.in  (if not done)")
    print(f"    pw2wannier90.x < pw2wan.in")
    print(f"    wannier90.x {formula.lower()}")


if __name__ == "__main__":
    main()