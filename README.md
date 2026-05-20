# Spin Hall Conductivity (SHC) Calculation Project

This project evaluates the intrinsic Spin Hall Conductivity (SHC) of a target material from first principles using Quantum ESPRESSO, Wannier90, and WannierBerri.

## 1. Computational Environment
- **OS**: Windows 11 (WSL2 Ubuntu)
- **Quantum ESPRESSO**: v7.4.1 (`pw.x`, `pw2wan.x`)
- **Wannier90**: v3.1.0 (`wannier90.x`, `postw90.x`)
- **WannierBerri**: Python 3.x (`pip install wannierberri`)

## 2. Prerequisites & Installation

Follow these steps to set up the environment on your WSL2 Ubuntu terminal.

### Step 2.1: Install Compilers and Standard Libraries
Update your system and install the required Fortran/C compilers, MPI for parallel computing, and optimized math libraries (BLAS, LAPACK, FFTW).
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential gfortran libopenblas-dev libfftw3-dev libopenmpi-dev openmpi-bin wget git python3-pip -y
```

### Step 2.2: Download and Compile Quantum ESPRESSO & Wannier90
Download the Quantum ESPRESSO source code (v7.3), configure it to detect your local libraries, and compile the essential binaries including the built-in Wannier90 utility.
``` Bash
# Clone the repository
git clone https://gitlab.com/QEF/q-e.git
cd q-e
# Switch to the latest stable release tag (v7.4.1)
git checkout qe-7.4.1
# Configure the environment
./configure
# Compile PW, Post-Processing, Quantum Transport, and Wannier90
make pw pp pwcond w90

cd ../
```
After a successful compilation, all executable binaries (e.g., pw.x, wannier90.x) will be located in the bin/ directory.
### Step 2.3: Setup Python Environment with uv
Install uv via the official standalone installer, initialize the project environment, and add WannierBerri. This creates an isolated virtual environment (.venv) automatically.
```Bash
# Install uv standalone
curl -LsSf https://astral.sh/uv/install.sh | sh

# (Optional) Restart your terminal or source the environment to activate 'uv' command
source $HOME/.local/bin/env

# Sync the project to automatically create .venv and install all required packages
uv sync
```
## 3. Directory Structure
```Plaintext
.
├── README.md
├── pyproject.toml      # Tracks Python dependencies (Shared across all materials)
├── uv.lock             # Locks exact version overrides
├── .python-version     # Specifies target Python version
├── .gitignore          # Globally ignores large binary data and tmp directories
├── .venv/              # Isolated virtual environment shared by all folders
│
├── 00_plots/           # Centralized folder for data visualization & comparison
│   └── compare_shc.py  # Python script to overlay and compare SHC spectra
│
└── calc/               # Centralized calculation folder
    ├── Pt/             # Folder for Platinum calculation (Example)
    │   ├── 01_scf/     # Self-Consistent Field (SOC required, needs .UPF)
    │   │   └── scf.in
    │   ├── 02_nscf/    # Non-Self-Consistent Field (dense k-mesh, nosym=.true.)
    │   │   └── nscf.in
    │   ├── 03_wannier/ # Wannierization (Wannier90 master file and pw2wan)
    │   │   ├── Pt.win
    │   │   └── pw2wan.in
    │   └── 04_shc/     # Spin Hall Conductivity (WannierBerri execution)
    │       └── run_shc.py
    │
    └── [Other_Material]/ # Easily add other materials (e.g., W, Bi2Se3)
        ├── 01_scf/
        └── ...
```
## 4. Calculation Workflow
### Step 1: SCF Calculation (01_scf/)
Total energy calculation including spin-orbit coupling (SOC) and non-collinear magnetism.
- Crucial Parameters:
    - noncolin = .true.
    - lspinorb = .true.
- Execution Command:
```Bash
mpirun -np 4 /path/to/q-e/bin/pw.x < scf.in > scf.out
```
### Step 2: NSCF Calculation (02_nscf/)
Generate Bloch wavefunctions on a uniform k-point grid for Wannier projection.
- Crucial Parameters:
    - nosym = .true. (Deactivates symmetry operations for compatibility with Wannier90)
    - Execution Command:
```Bash
mpirun -np 4 /path/to/q-e-qe-7.3/bin/pw.x < nscf.in > nscf.out
```
### Step 3: Wannierization (03_wannier/)
1. Construct Maximally Localized Wannier Functions (MLWFs) from the Bloch states. Generate the Wannier90 seed file:
```Bash
/path/to/q-e-qe-7.3/bin/wannier90.x -pp [Material]
```
2. Compute the overlap matrices and projections from QE:
```Bash
mpirun -np 4 /path/to/q-e-qe-7.3/bin/pw2wan.x < pw2wan.in > pw2wan.out
```
3. Minimize the spread to obtain MLWFs:
```Bash
mpirun -np 4 /path/to/q-e-qe-7.3/bin/wannier90.x [Material]
```
- Check [Material].wout to ensure the spreads are well-converged.
### Step 4: Spin Hall Conductivity Calculation (04_shc/)
Integrate the Berry curvature over an extremely dense k-point mesh to compute the SHC ($\sigma_{ij}^k$) using WannierBerri.
- Execution Command:
```Bash
python run_shc.py
```
## 5. Notes & Reminders
- Pseudopotentials: You must use Fully-Relativistic (FR) pseudopotentials (usually containing rel in their filenames). Scalar-relativistic potentials omit SOC, which will result in an SHC of exactly zero.
- k-point Convergence: Intrinsic SHC calculations require very dense k-point sampling due to the sharp features of the Berry curvature. Final production runs may require a mesh as dense as 100x100x100 or finer.
