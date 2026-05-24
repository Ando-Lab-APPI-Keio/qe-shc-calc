# Spin Hall Conductivity (SHC) Calculation Project

This project evaluates the intrinsic Spin Hall Conductivity (SHC) of a target material from first principles using Quantum ESPRESSO, Wannier90, and WannierBerri.

---

## 1. Computational Environment

| Component | Version |
|---|---|
| OS | Windows 11 (WSL2 Ubuntu) |
| Quantum ESPRESSO | v7.4.1 (`pw.x`, `bands.x`, `pw2wannier90.x`) |
| Wannier90 | v3.1.0 (`wannier90.x`) |
| WannierBerri | >=v26.4.6|

---

## 2. Prerequisites & Installation

### Step 2.1: Install compilers and standard libraries

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential gfortran libopenblas-dev libfftw3-dev libopenmpi-dev openmpi-bin wget git python3-pip -y
```

### Step 2.2: Download and compile Quantum ESPRESSO & Wannier90

```bash
git clone https://gitlab.com/QEF/q-e.git
cd q-e
git checkout qe-7.4.1
./configure

make w90        # Wannier90 must be built first
make pw pp pwcond

cd ../
```

All binaries (`pw.x`, `wannier90.x`, etc.) are placed in `q-e/bin/`.

### Step 2.3: Set up Python environment with uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
```

---

## 3. Project Structure

```
.
├── README.md
├── pyproject.toml          # Python dependencies
├── uv.lock
├── .python-version
├── .gitignore
├── .venv/
│
├── setup_material.py       # Generate all QE/Wannier90 input files for a material
├── update_win.py           # Auto-update .win energy windows after SCF / NSCF
├── run_pipeline.py         # End-to-end pipeline (SCF → Wannier → bands → SHC)
├── run_shc.py              # SHC calculation via WannierBerri
├── plot_bands.py           # Band structure plot from bands.x output
│
├── 00_plots/
│   └── compare_shc.py      # Overlay and compare SHC spectra across materials
│
└── calc/
    └── Pt/                 # One subdirectory per material
        ├── 01_scf/         # SCF ground state
        │   ├── scf.in
        │   ├── scf.out
        │   ├── *.UPF
        │   └── tmp/
        ├── 02_nscf/        # Non-SCF on dense k-mesh
        │   ├── nscf.in
        │   └── nscf.out
        ├── 03_wannier/     # Wannierization
        │   ├── pt.win
        │   ├── pw2wan.in
        │   ├── pt_hr.dat
        │   └── update_win_scf.log / update_win_nscf.log
        ├── 04_bands/       # Band structure
        │   ├── bands.in
        │   ├── bands.pp.in
        │   ├── bands.dat.gnu
        │   └── bands_Pt.png
        └── 05_shc/         # SHC output (auto-created by run_shc.py)
            ├── shc_Pt.dat
            └── shc_Pt.png
```

---

## 4. Scripts

### `setup_material.py`

Fetches the experimental crystal structure from the AFLOW/ICSD database via the Aflux API, converts it to QE format with `cif2cell`, downloads the required fully-relativistic pseudopotentials, and writes all input files.

```bash
uv run setup_material.py Pt
```

Generated files:

| File | Purpose |
|---|---|
| `01_scf/scf.in` | SCF input |
| `02_nscf/nscf.in` | NSCF input on 8×8×8 k-mesh |
| `03_wannier/pt.win` | Wannier90 master file |
| `03_wannier/pw2wan.in` | pw2wannier90 input |
| `04_bands/bands.in` | Band structure pw.x input |
| `04_bands/bands.pp.in` | bands.x post-processing input |

Wannier parameters are estimated automatically:

- `num_wann`: sum of projected orbitals × atoms × 2 (spinor)
- `nbnd`: max(num_wann × 1.5, nelec × 1.2), rounded to a multiple of 4
- `dis_win_max` / `dis_froz_max`: conservative placeholders; updated automatically by `update_win.py`

### `update_win.py`

Reads the Fermi energy from `scf.out` and the highest band energy from `nscf.out`, then patches `dis_froz_max` and `dis_win_max` in the `.win` file. A backup (`.win.bak`) is created before every modification.

```bash
# After SCF — sets dis_froz_max = E_Fermi + 2.0 eV
uv run update_win.py Pt

# After NSCF — also sets dis_win_max = E_band_max - 2.0 eV
uv run update_win.py Pt

# Custom offsets
uv run update_win.py Pt --froz-offset 3.0 --win-margin 1.0
```

### `run_pipeline.py`

Runs the full workflow end-to-end in the correct order.

```bash
uv run run_pipeline.py Pt
```

| Step | Action |
|---|---|
| 1/8 | SCF (`pw.x`) |
| 2/8 | `update_win.py` — set `dis_froz_max` from `scf.out` |
| 3/8 | NSCF (`pw.x`) |
| 4/8 | `update_win.py` — refine `dis_win_max` from `nscf.out` |
| 5/8 | Wannier90 preprocessing (`wannier90.x -pp`) |
| 6/8 | `pw2wannier90.x` |
| 7/8 | Wannier90 main (`wannier90.x`) |
| 8/8 | Band structure (`pw.x` → `bands.x` → `plot_bands.py`) |

Each step writes a log file to the relevant calculation directory. On failure the last 50 lines of the log are printed and the pipeline stops.

### `run_shc.py`

Computes the SHC by integrating the Berry curvature over a dense k-mesh using WannierBerri. Run after `run_pipeline.py` has completed successfully.

```bash
uv run run_shc.py Pt
```

Output is written to `calc/Pt/05_shc/`.

### `plot_bands.py`

Plots the band structure from `bands.dat.gnu` produced by `bands.x`. Called automatically by `run_pipeline.py`; can also be run standalone.

```bash
uv run plot_bands.py Pt
```

Output: `calc/Pt/04_bands/bands_Pt.png`

---

## 5. Calculation Workflow (manual)

If you prefer to run each step by hand instead of using `run_pipeline.py`:

```bash
# 1. Generate input files
uv run setup_material.py Pt

# 2. SCF
cd calc/Pt/01_scf && mkdir -p tmp
mpirun -np 4 ../../../q-e/bin/pw.x < scf.in > scf.out
cd ../../..

# 3. Update energy windows from SCF Fermi energy
uv run update_win.py Pt

# 4. NSCF
cd calc/Pt/02_nscf
mpirun -np 4 ../../../q-e/bin/pw.x < nscf.in > nscf.out
cd ../../..

# 5. Refine energy windows from NSCF band energies
uv run update_win.py Pt

# 6. Wannier90 preprocessing
cd calc/Pt/03_wannier
../../../q-e/bin/wannier90.x -pp pt

# 7. pw2wannier90
mpirun -np 4 ../../../q-e/bin/pw2wannier90.x < pw2wan.in > pw2wan.out

# 8. Wannier90 main
../../../q-e/bin/wannier90.x pt
cd ../../..

# 9. Band structure
cd calc/Pt/04_bands
mpirun -np 4 ../../../q-e/bin/pw.x < bands.in > bands.out
mpirun -np 4 ../../../q-e/bin/bands.x < bands.pp.in > bands_pp.out
cd ../../..
python plot_bands.py Pt

# 10. SHC
uv run run_shc.py Pt
```

---

## 6. Adding a New Material

```bash
uv run setup_material.py GaAs
uv run run_pipeline.py GaAs
uv run run_shc.py GaAs
```

Supported elements with built-in pseudopotential metadata: Pt, W, Au, Pd, Ir, Ga, As, Bi, Se, Fe, Co, Ni, Mo, Ta, Hf.
For other elements, `setup_material.py` constructs a pseudopotential filename automatically; download the file manually from [quantum-espresso.org](https://pseudopotentials.quantum-espresso.org/upf_files/) if needed.

---

## 7. Notes

**Pseudopotentials**: Always use fully-relativistic (FR) pseudopotentials (filenames contain `rel`). Scalar-relativistic potentials omit SOC and will produce an SHC of exactly zero.

**Energy windows**: `dis_froz_max` and `dis_win_max` are set automatically by `update_win.py`. If Wannier90 reports convergence issues, adjust the offsets:
```bash
uv run update_win.py Pt --froz-offset 3.0 --win-margin 1.0
```
Then re-run from Step 5 (Wannier90 -pp).

**k-point convergence**: Intrinsic SHC requires very dense k-point sampling due to sharp Berry curvature features. Final production runs may need a mesh of 100×100×100 or finer.

**Verifying the Wannier fit**: Compare `calc/Pt/04_bands/bands_Pt.png` (QE band structure) against the Wannier band interpolation from `pt.wout` to confirm the quality of the Wannierization before trusting the SHC result.
