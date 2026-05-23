import os
import subprocess
import sys
import shutil

def run_command(cmd, log_file, shell=False):
    print(f"Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    try:
        with open(log_file, "w") as f:
            subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT,
                text=True, check=True, shell=shell
            )
        print(f"Success! Log: {log_file}\n", flush=True)
    except subprocess.CalledProcessError:
        print(f"ERROR. Check: {log_file}", flush=True)
        try:
            with open(log_file) as f:
                lines = f.readlines()
            print("".join(lines[-50:]), flush=True)
        except Exception:
            pass
        sys.exit(1)

def print_snapshot(label):
    print(f"\n{'='*60}\n[SNAP] {label}")
    print(f"  CWD: {os.getcwd()}")
    for f in sorted(os.listdir(".")):
        print(f"    {f} ({os.path.getsize(f)} B)")
    print("="*60 + "\n")

def dump_file(filename):
    print(f"\n[DUMP] {filename}")
    if os.path.exists(filename):
        with open(filename) as f:
            print(f.read())
    else:
        print(f"  {filename} does not exist.\n")

def main(formula: str):
    seedname = formula.lower()

    ROOT_DIR   = os.path.abspath(os.path.dirname(__file__))
    QE_BIN_DIR = os.path.join(ROOT_DIR, "q-e", "bin")

    PW_EX      = os.path.join(QE_BIN_DIR, "pw.x")
    PP_EX      = os.path.join(QE_BIN_DIR, "pp.x")
    BANDS_EX   = os.path.join(QE_BIN_DIR, "bands.x")
    PW2WAN_EX  = os.path.join(QE_BIN_DIR, "pw2wannier90.x")
    W90_EX     = os.path.join(QE_BIN_DIR, "wannier90.x")

    base_dir  = os.path.join(ROOT_DIR, "calc", formula)
    scf_dir   = os.path.join(base_dir, "01_scf")
    nscf_dir  = os.path.join(base_dir, "02_nscf")
    wan_dir   = os.path.join(base_dir, "03_wannier")
    bands_dir = os.path.join(base_dir, "04_bands")

    # Check required base directories
    for d, label in [(scf_dir, "01_scf"), (nscf_dir, "02_nscf"), (wan_dir, "03_wannier")]:
        if not os.path.isdir(d):
            sys.exit(f"Directory not found: {d}\n"
                     f"   Please run first: python setup_material.py {formula}")

    # Check required input files
    for path, label in [
        (os.path.join(scf_dir,  "scf.in"),          "scf.in"),
        (os.path.join(nscf_dir, "nscf.in"),         "nscf.in"),
        (os.path.join(wan_dir,  f"{seedname}.win"), f"{seedname}.win"),
        (os.path.join(wan_dir,  "pw2wan.in"),       "pw2wan.in"),
    ]:
        if not os.path.exists(path):
            sys.exit(f"Input file not found: {path}\n"
                     f"   Please run first: python setup_material.py {formula}")

    print("=" * 60, flush=True)
    print(f"SHC Pipeline: {formula}", flush=True)
    print("=" * 60 + "\n", flush=True)

    # ------------------------------------------------------------------
    # Step 1: SCF
    # ------------------------------------------------------------------
    print("--- [Step 1/8] SCF ---", flush=True)
    os.chdir(scf_dir)
    os.makedirs("tmp", exist_ok=True)
    run_command(f"mpirun -np 4 {PW_EX} < scf.in", "scf.out", shell=True)

    # ------------------------------------------------------------------
    # Step 2: Update .win energy windows from SCF Fermi energy
    #         Sets dis_froz_max = E_Fermi + 2.0 eV
    # ------------------------------------------------------------------
    print("--- [Step 2/8] Update .win (dis_froz_max from scf.out) ---", flush=True)
    os.chdir(ROOT_DIR)
    run_command(
        [sys.executable,
         os.path.join(ROOT_DIR, "update_win.py"), formula],
        os.path.join(wan_dir, "update_win_scf.log")
    )

    # ------------------------------------------------------------------
    # Step 3: NSCF
    # ------------------------------------------------------------------
    print("--- [Step 3/8] NSCF ---", flush=True)
    os.chdir(nscf_dir)
    run_command(f"mpirun -np 4 {PW_EX} < nscf.in", "nscf.out", shell=True)

    # ------------------------------------------------------------------
    # Step 4: Refine .win energy windows from NSCF band energies
    #         Sets dis_win_max = E_band_max - 2.0 eV
    # ------------------------------------------------------------------
    print("--- [Step 4/8] Update .win (dis_win_max from nscf.out) ---", flush=True)
    os.chdir(ROOT_DIR)
    run_command(
        [sys.executable,
         os.path.join(ROOT_DIR, "update_win.py"), formula],
        os.path.join(wan_dir, "update_win_nscf.log")
    )

    # ------------------------------------------------------------------
    # Step 5: Wannier90 preprocessing (-pp)
    # ------------------------------------------------------------------
    print("--- [Step 5/8] Wannier90 -pp ---", flush=True)
    os.chdir(wan_dir)

    for trash in [f"{seedname}.werr", f"{seedname}.wout", "CRASH",
                  f"{seedname}.nnkp", "wannier90_pp.out"]:
        if os.path.exists(trash):
            os.remove(trash)

    print_snapshot("BEFORE wannier90 -pp")

    print(f"Running: {W90_EX} -pp {seedname}", flush=True)
    try:
        with open("wannier90_pp.out", "w") as f:
            subprocess.run(
                [W90_EX, "-pp", seedname],
                stdout=f, stderr=subprocess.STDOUT, text=True, check=True
            )
        print("Success!\n", flush=True)
    except subprocess.CalledProcessError:
        print("wannier90 -pp failed", flush=True)
        print_snapshot("AFTER wannier90 -pp (FAILED)")
        dump_file(f"{seedname}.werr")
        dump_file(f"{seedname}.wout")
        dump_file("wannier90_pp.out")
        sys.exit(1)

    print_snapshot("AFTER wannier90 -pp")

    # ------------------------------------------------------------------
    # Step 6: pw2wannier90.x
    # ------------------------------------------------------------------
    print("--- [Step 6/8] pw2wannier90 ---", flush=True)
    os.chdir(wan_dir)
    run_command(f"mpirun -np 4 {PW2WAN_EX} < pw2wan.in", "pw2wan.out", shell=True)

    # ------------------------------------------------------------------
    # Step 7: Wannier90 main calculation
    # ------------------------------------------------------------------
    print("--- [Step 7/8] Wannier90 main ---", flush=True)
    os.chdir(wan_dir)
    run_command([W90_EX, seedname], "wannier90_main.out")

    hr_dat = f"{seedname}_hr.dat"
    if not os.path.exists(hr_dat):
        print(f"{hr_dat} was not generated.", flush=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 8: Band structure (pw.x bands -> bands.x -> plot_bands.py)
    # ------------------------------------------------------------------
    print("--- [Step 8/8] Band structure ---", flush=True)
    os.chdir(bands_dir)

    # pw.x bands calculation
    run_command(
        f"mpirun -np 4 {PW_EX} < bands.in",
        "bands.out", shell=True
    )

    # bands.x post-processing (extracts eigenvalues + k-distances -> bands.dat.gnu)
    run_command(
        f"mpirun -np 4 {BANDS_EX} < bands.pp.in",
        "bands_pp.out", shell=True
    )

    # Visualise bands (Python, no MPI needed)
    run_command(
        [sys.executable,
         os.path.join(ROOT_DIR, "plot_bands.py"), formula],
        os.path.join(bands_dir, "plot_bands.log")
    )

    print(f"\n Pipeline complete!", flush=True)
    print(f"   Band structure : calc/{formula}/04_bands/bands_{formula}.png")
    print(f"   Wannier HR     : calc/{formula}/03_wannier/{hr_dat}")
    print(f"\n   Next: python run_shc.py {formula}\n", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python run_pipeline.py <Formula>  (e.g. Pt, GaAs, W)")
    main(sys.argv[1])