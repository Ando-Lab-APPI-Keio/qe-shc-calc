import os
import sys
import re
import subprocess
import requests
import numpy as np
from urllib.request import urlretrieve

import spglib
import seekpath

# =======================================================================
# 1. AFLOW/Aflux API - Experimental Data Crawler
# =======================================================================

def parse_formula(formula):
    return re.findall(r'[A-Z][a-z]?', formula)

def get_aflow_cif(chemical_formula, output_dir, spacegroup=225):
    """Fetch experimental CIF from Aflux API using ICSD catalog filter."""
    base_url = "http://aflow.org/API/aflux/?"
    elements = parse_formula(chemical_formula)
    if not elements:
        print(f"Error: Failed to parse chemical formula '{chemical_formula}'")
        return False

    species_query = ",".join(elements)
    num_species = len(elements)

    properties = "compound,auid,aurl,spacegroup_relax,Pearson_symbol_relax,energy_atom"
    query = f"species({species_query}),$catalog(ICSD),$nspecies({num_species}),{properties},format(json)"
    url = base_url + query

    print("Fetching data from Aflux API...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data:
            print(f"Error: No materials found in ICSD catalog for '{chemical_formula}'")
            return False

        filtered_entries = [x for x in data if x.get("spacegroup_relax") == spacegroup and x.get("energy_atom") is not None]
        if not filtered_entries:
            filtered_entries = sorted(data, key=lambda x: x.get("energy_atom", 0))

        entry = sorted(filtered_entries, key=lambda x: x["energy_atom"])[0]
        aurl = entry.get("aurl")

        os.makedirs(output_dir, exist_ok=True)

        filename = f"Rank1_{entry['compound']}_SG{entry['spacegroup_relax']}_{entry['Pearson_symbol_relax']}.cif"
        filepath = os.path.join(output_dir, filename)

        clean_path = aurl.replace("aflowlib.duke.edu:", "").lstrip("/")
        folder_url = f"http://aflow.org/{clean_path}/"

        folder_response = requests.get(folder_url, timeout=15)
        folder_response.raise_for_status()

        cif_files = re.findall(r'href="([^"]+\.cif)"', folder_response.text)
        cif_files = list(set([f for f in cif_files if not f.startswith("/") and "http" not in f]))

        cif_download_url = f"{folder_url}{cif_files[0]}"
        cif_res = requests.get(cif_download_url, timeout=15)
        cif_res.raise_for_status()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cif_res.text)

        print(f"CIF saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"API Error: {e}")
        return False


# =======================================================================
# 2. CIF -> QE geometry via cif2cell + seekpath primitive cell reduction
#
# Pipeline:
#   cif2cell  ->  conventional cell (lattice vectors + fractional coords)
#   spglib    ->  standardise and find primitive cell
#   seekpath  ->  standardised primitive cell + automatic k-path
#
# The output geo dict always contains a primitive cell, so num_wann
# is minimised regardless of what cif2cell originally returns.
# =======================================================================

def run_cif2cell_and_extract(cif_path, work_dir):
    """
    Run cif2cell to get a QE-format conventional cell, then parse it
    into raw Python arrays for later primitive-cell reduction by spglib.

    Returns a dict with keys:
        lattice      : np.ndarray (3,3)  row vectors in Angstrom
        positions    : list of [x,y,z]  fractional coordinates
        species      : list of str       element symbol per site
        atomic_species_meta : dict  symbol -> {mass, upf}  (from cif2cell output)
    """
    venv_python_dir = os.path.dirname(sys.executable)
    cif2cell_executable = os.path.join(venv_python_dir, "cif2cell")
    if not os.path.exists(cif2cell_executable):
        cif2cell_executable = "cif2cell"

    tmp_out = os.path.join(work_dir, "cif_trans.in")
    cmd = [cif2cell_executable, "-p", "pwscf", cif_path, "-o", tmp_out]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"cif2cell error:\n{e.stderr}")
        sys.exit(1)

    with open(tmp_out, "r", encoding="utf-8") as f:
        content = f.read()
    if os.path.exists(tmp_out):
        os.remove(tmp_out)

    # ---- Determine if CELL_PARAMETERS are in 'angstrom' (absolute) or 'alat' (scaled) ----
    # cif2cell with -p pwscf emits either:
    #   CELL_PARAMETERS angstrom   -> rows are already in Angstrom; scale factor = 1.0
    #   CELL_PARAMETERS {alat}     -> rows are in units of alat; need to multiply by A
    # We detect the unit tag on the CELL_PARAMETERS line itself.
    cell_params_unit = "angstrom"   # default assumption
    for line in content.splitlines():
        if "CELL_PARAMETERS" in line:
            m = re.search(r'CELL_PARAMETERS\s*\{?(\w+)\}?', line, re.IGNORECASE)
            if m:
                cell_params_unit = m.group(1).lower()
            break

    # Parse the A (alat) value regardless; needed only when unit != angstrom
    a_value = None
    for line in content.splitlines():
        m = re.match(r'^\s*A\s*=\s*([\d.]+)', line)
        if m:
            a_value = float(m.group(1))
            break
    if a_value is None:
        for line in content.splitlines():
            m = re.match(r'^\s*celldm\(1\)\s*=\s*([\d.]+)', line)
            if m:
                a_value = float(m.group(1)) * 0.529177
                break
    if a_value is None:
        a_value = 1.0

    # Scale factor: 1.0 if already in Angstrom, else alat value
    if cell_params_unit in ("angstrom", "a"):
        scale = 1.0
    else:
        scale = a_value if a_value else 1.0

    print(f"  cif2cell CELL_PARAMETERS unit = '{cell_params_unit}', scale = {scale:.6f} A")

    # ---- Block extractor ----
    def extract_block(text, header_keyword, stop_keywords):
        lines = text.splitlines()
        block, recording = [], False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("!") or stripped.startswith("#"):
                continue
            if header_keyword in line:
                recording = True
            if recording:
                if any(kw in line for kw in stop_keywords) and header_keyword not in line:
                    break
                block.append(line)
        return "\n".join(block).strip()

    cell_block = extract_block(content, "CELL_PARAMETERS",
                               ["ATOMIC_SPECIES", "ATOMIC_POSITIONS", "K_POINTS"])
    pos_block  = extract_block(content, "ATOMIC_POSITIONS",
                               ["K_POINTS", "CELL_PARAMETERS", "ATOMIC_SPECIES"])
    sp_block   = extract_block(content, "ATOMIC_SPECIES",
                               ["ATOMIC_POSITIONS", "CELL_PARAMETERS", "K_POINTS"])

    # ---- Parse lattice vectors ----
    raw_vecs = [l for l in cell_block.splitlines()
                if l.strip() and "CELL_PARAMETERS" not in l]
    lattice = np.array([[float(x) * scale for x in l.split()[:3]]
                        for l in raw_vecs[:3]])

    # ---- Parse atomic positions (fractional) ----
    species, positions = [], []
    for line in pos_block.splitlines():
        if not line.strip() or "ATOMIC_POSITIONS" in line:
            continue
        parts = line.split()
        species.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    # ---- Parse ATOMIC_SPECIES for mass/UPF hints ----
    atomic_species_meta = {}
    for line in sp_block.splitlines():
        if not line.strip() or "ATOMIC_SPECIES" in line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            atomic_species_meta[parts[0]] = {"mass": parts[1], "upf": parts[2]}

    return {
        "lattice":             lattice,
        "positions":           positions,
        "species":             species,
        "atomic_species_meta": atomic_species_meta,
    }


def geo_to_primitive(raw):
    """
    Convert a cif2cell conventional-cell dict to a primitive cell using spglib,
    then pass it through seekpath to get the standardised primitive cell and
    the automatic k-path (Hinuma et al. 2017 convention).

    Returns a geo dict ready for QE input generation:
        cell_params_block    : str
        atomic_species_block : str  (element symbol only; UPF filled later)
        atomic_positions_block: str
        nat, ntyp            : int
        unit_cell_lines      : str
        seekpath_result      : dict  (raw seekpath output for k-path extraction)
        spacegroup_number    : int
        bravais_lattice      : str
    """
    lattice    = raw["lattice"]
    positions  = raw["positions"]
    species    = raw["species"]

    # spglib uses integer atomic numbers; map element symbols to numbers
    unique_syms = sorted(set(species))
    sym_to_num  = {s: i + 1 for i, s in enumerate(unique_syms)}
    numbers     = [sym_to_num[s] for s in species]
    num_to_sym  = {v: k for k, v in sym_to_num.items()}

    cell = (lattice.tolist(), positions, numbers)

    # ---- Get standardised primitive cell via spglib ----
    # Use a slightly generous symprec (1e-3 A) to handle AFLOW CIFs where
    # corner atoms are displaced by ~0.01 fractional coords from ideal sites.
    SYMPREC = 1e-3
    prim = spglib.find_primitive(cell, symprec=SYMPREC)
    if prim is None:
        print("  WARNING: spglib could not find primitive cell; using input cell.")
        prim_lattice   = lattice
        prim_positions = positions
        prim_numbers   = numbers
    else:
        prim_lattice, prim_positions, prim_numbers = prim
        prim_lattice   = np.array(prim_lattice)
        prim_positions = [list(p) for p in prim_positions]
        prim_numbers   = list(prim_numbers)

    prim_species = [num_to_sym[n] for n in prim_numbers]
    n_prim = len(prim_species)

    # ---- Spacegroup of the primitive cell ----
    prim_cell = (prim_lattice.tolist(), prim_positions, prim_numbers)
    sg_info   = spglib.get_spacegroup(prim_cell, symprec=SYMPREC)
    sg_number = int(re.search(r'\((\d+)\)', sg_info).group(1)) if sg_info else 0
    print(f"  Primitive cell: {n_prim} atoms, spacegroup {sg_info}")

    # ---- seekpath: standardised primitive cell + k-path ----
    # seekpath expects (lattice_rows, [[x,y,z]...], [atomic_numbers...])
    sp_result = seekpath.get_path(
        (prim_lattice.tolist(), prim_positions, prim_numbers),
        with_time_reversal=True,
        symprec=SYMPREC,
    )

    # Use the seekpath-standardised cell for QE (ensures correct k-path orientation)
    std_lattice   = np.array(sp_result["primitive_lattice"])
    std_positions = sp_result["primitive_positions"]
    std_numbers   = sp_result["primitive_types"]
    std_species   = [num_to_sym[n] for n in std_numbers]

    nat  = len(std_species)
    ntyp = len(set(std_species))

    # ---- Build QE blocks ----
    vec_lines = [
        f"  {v[0]:18.12f}  {v[1]:18.12f}  {v[2]:18.12f}"
        for v in std_lattice
    ]
    cell_params_block = "CELL_PARAMETERS angstrom\n" + "\n".join(vec_lines)
    unit_cell_lines   = "\n".join(vec_lines)

    # ATOMIC_SPECIES: placeholder UPFs filled by patch_atomic_species_block later
    unique_std = sorted(set(std_species))
    sp_lines   = ["ATOMIC_SPECIES"]
    for el in unique_std:
        sp_lines.append(f"  {el}  ???  ???.UPF")
    atomic_species_block = "\n".join(sp_lines)

    pos_lines = ["ATOMIC_POSITIONS crystal"]
    for sym, pos in zip(std_species, std_positions):
        pos_lines.append(f"  {sym}  {pos[0]:.12f}  {pos[1]:.12f}  {pos[2]:.12f}")
    atomic_positions_block = "\n".join(pos_lines)

    return {
        "cell_params_block":      cell_params_block,
        "atomic_species_block":   atomic_species_block,
        "atomic_positions_block": atomic_positions_block,
        "nat":                    nat,
        "ntyp":                   ntyp,
        "unit_cell_lines":        unit_cell_lines,
        "seekpath_result":        sp_result,
        "spacegroup_number":      sg_number,
        "bravais_lattice":        sp_result.get("bravais_lattice", "unknown"),
    }


# =======================================================================
# 3. Pseudopotential metadata & download
# =======================================================================

ELEMENT_META = {
    "Pt": {"mass": "195.08", "upf": "Pt.rel-pbe-n-rrkjus_psl.1.0.0.UPF"},
    "Ga": {"mass": "69.723", "upf": "Ga.rel-pbe-dn-rrkjus_psl.1.0.0.UPF"},
    "As": {"mass": "74.922", "upf": "As.rel-pbe-n-rrkjus_psl.1.0.0.UPF"},
    "W":  {"mass": "183.84", "upf": "W.rel-pbe-spn-rrkjus_psl.1.0.0.UPF"},
    "Bi": {"mass": "208.98", "upf": "Bi.rel-pbe-dn-rrkjus_psl.1.0.0.UPF"},
    "Se": {"mass": "78.971", "upf": "Se.rel-pbe-n-rrkjus_psl.1.0.0.UPF"},
    "Ta": {"mass": "180.948", "upf": "Ta.rel-pbe-spn-rrkjus_psl.1.0.0.UPF"},
    "Au": {"mass": "196.967", "upf": "Au.rel-pbe-n-rrkjus_psl.1.0.0.UPF"},
    "Pd": {"mass": "106.42",  "upf": "Pd.rel-pbe-n-rrkjus_psl.1.0.0.UPF"},
    "Ir": {"mass": "192.217", "upf": "Ir.rel-pbe-n-rrkjus_psl.1.0.0.UPF"},
    "Hf": {"mass": "178.49",  "upf": "Hf.rel-pbe-spn-rrkjus_psl.1.0.0.UPF"},
    "Mo": {"mass": "95.96",   "upf": "Mo.rel-pbe-spn-rrkjus_psl.1.0.0.UPF"},
    "Fe": {"mass": "55.845",  "upf": "Fe.rel-pbe-spn-rrkjus_psl.1.0.0.UPF"},
    "Co": {"mass": "58.933",  "upf": "Co.rel-pbe-spn-rrkjus_psl.1.0.0.UPF"},
    "Ni": {"mass": "58.693",  "upf": "Ni.rel-pbe-n-rrkjus_psl.1.0.0.UPF"},
}
UPF_BASE_URL = "https://pseudopotentials.quantum-espresso.org/upf_files/"


def get_element_meta(element):
    if element in ELEMENT_META:
        return ELEMENT_META[element]
    return {"mass": "100.0", "upf": f"{element}.rel-pbe-n-rrkjus_psl.1.0.0.UPF"}

def ensure_upf(element, pseudo_dir):
    meta = get_element_meta(element)
    upf_filename = meta["upf"]
    upf_path = os.path.join(pseudo_dir, upf_filename)
    if not os.path.exists(upf_path):
        url = UPF_BASE_URL + upf_filename
        print(f"  Downloading pseudopotential: {url}")
        try:
            urlretrieve(url, upf_path)
            print(f"  Saved to: {upf_path}")
        except Exception as e:
            print(f"  WARNING: Failed to download {upf_filename}: {e}")
            print(f"  Please download it manually from {url}")
    return meta

def patch_atomic_species_block(atomic_species_block, unique_elements, pseudo_dir):
    patched_lines = []
    for line in atomic_species_block.splitlines():
        if "ATOMIC_SPECIES" in line:
            patched_lines.append(line)
            continue
        parts = line.split()
        if not parts:
            patched_lines.append(line)
            continue
        symbol = parts[0]
        if symbol in unique_elements:
            meta = ensure_upf(symbol, pseudo_dir)
            patched_lines.append(f"  {symbol}  {meta['mass']}  {meta['upf']}")
        else:
            patched_lines.append(line)
    return "\n".join(patched_lines)


# =======================================================================
# 4. Input file templates
# =======================================================================

SCF_TEMPLATE = """\
&CONTROL
    calculation  = 'scf'
    restart_mode = 'from_scratch'
    prefix       = '{prefix}'
    pseudo_dir   = './'
    outdir       = './tmp/'
/
&SYSTEM
    ibrav       = 0
    nat         = {nat}
    ntyp        = {ntyp}
    ecutwfc     = {ecutwfc}
    ecutrho     = {ecutrho}
    occupations = 'smearing'
    smearing    = 'mv'
    degauss     = 0.02
    noncolin    = .true.
    lspinorb    = .true.
    nbnd        = {nbnd}
/
&ELECTRONS
    mixing_beta = 0.7
    conv_thr    = 1.0e-8
/
{atomic_species_block}

{cell_params_block}

{atomic_positions_block}

K_POINTS automatic
  8 8 8  0 0 0
"""

NSCF_TEMPLATE = """\
&CONTROL
    calculation  = 'nscf'
    restart_mode = 'from_scratch'
    prefix       = '{prefix}'
    pseudo_dir   = '../01_scf/'
    outdir       = '../01_scf/tmp/'
/
&SYSTEM
    ibrav       = 0
    nat         = {nat}
    ntyp        = {ntyp}
    ecutwfc     = {ecutwfc}
    ecutrho     = {ecutrho}
    occupations = 'smearing'
    smearing    = 'mv'
    degauss     = 0.02
    noncolin    = .true.
    lspinorb    = .true.
    nosym       = .true.
    nbnd        = {nbnd}
/
&ELECTRONS
    mixing_beta = 0.7
    conv_thr    = 1.0e-8
/
{atomic_species_block}

{cell_params_block}

{atomic_positions_block}

K_POINTS crystal
  {nk}
{kpoints_list}"""

# Differences from scf.in:
#   - calculation = 'bands'
#   - pseudo_dir / outdir point to 01_scf/ (reuse the SCF charge density)
#   - occupations / smearing / degauss omitted (not meaningful for band runs)
#   - nosym = .true. to preserve all k-points along the path
#   - K_POINTS uses crystal_b format (high-symmetry point list from seekpath)
BANDS_TEMPLATE = """\
&CONTROL
    calculation  = 'bands'
    restart_mode = 'from_scratch'
    prefix       = '{prefix}'
    pseudo_dir   = '../01_scf/'
    outdir       = '../01_scf/tmp/'
/
&SYSTEM
    ibrav       = 0
    nat         = {nat}
    ntyp        = {ntyp}
    ecutwfc     = {ecutwfc}
    ecutrho     = {ecutrho}
    noncolin    = .true.
    lspinorb    = .true.
    nosym       = .true.
    nbnd        = {nbnd}
/
&ELECTRONS
    mixing_beta = 0.7
    conv_thr    = 1.0e-8
/
{atomic_species_block}

{cell_params_block}

{atomic_positions_block}

K_POINTS crystal_b
{kpath_block}"""

# Post-processing input for bands.x.
# filband = 'bands.dat' -> bands.x writes bands.dat.gnu, read by plot_bands.py.
BANDSPP_TEMPLATE = """\
&BANDS
    prefix      = '{prefix}'
    outdir      = '../01_scf/tmp/'
    filband     = 'bands.dat'
    lsym        = .false.
/
"""

PW2WAN_TEMPLATE = """\
&INPUTPP
    outdir       = '../01_scf/tmp/'
    prefix       = '{prefix}'
    seedname     = '{seedname}'
    spin_component = 'none'
    write_mmn    = .true.
    write_amn    = .true.
    write_unk    = .false.
    write_spn    = .true.
/
"""

WAN_TEMPLATE = """\
num_bands       = {num_bands}
num_wann        = {num_wann}

begin unit_cell_cart
Angstrom
{unit_cell_lines}
end unit_cell_cart

begin atoms_frac
{atoms_frac_block}
end atoms_frac

begin projections
{projections_block}
end projections

dis_win_max   = {dis_win_max}
dis_froz_max  = {dis_froz_max}
dis_num_iter  = 1000
num_iter      = 2000
spinors       = .true.
write_hr      = .true.
mp_grid       = 8 8 8
search_shells = 40

begin kpoints
{kpoints_block}end kpoints
"""


# =======================================================================
# 5. Wannier parameter estimation
# =======================================================================

D_METALS = {"Pt", "W", "Au", "Pd", "Ir", "Rh", "Ru", "Os", "Fe", "Co", "Ni", "Mn", "Mo", "Ta", "Hf"}

PROJECTION_ORBITALS = {
    "d_metal": ["s", "p", "d"],   # 1+3+5 = 9 orbitals x 2 (spinor) = 18 per atom
    "main":    ["s", "p"],        # 1+3   = 4 orbitals x 2 (spinor) =  8 per atom
}
ORB_COUNT = {"s": 1, "p": 3, "d": 5}

def get_orbitals_for_element(el):
    return PROJECTION_ORBITALS["d_metal"] if el in D_METALS else PROJECTION_ORBITALS["main"]

def count_wann_per_atom(el):
    return sum(ORB_COUNT[o] for o in get_orbitals_for_element(el)) * 2

def build_projections_block(unique_elements):
    lines = []
    for el in unique_elements:
        orbs = get_orbitals_for_element(el)
        lines.append(f"  {el} : " + " ; ".join(orbs))
    return "\n".join(lines)

def estimate_wannier_params(all_atoms_in_cell, nbnd_input):
    """
    Compute num_wann and nbnd consistent with the projections block.

    num_wann = sum of Wannier functions across all atoms in the cell.
    nbnd     = max(num_wann x 1.5, nbnd_input) rounded up to a multiple of 4.
    """
    num_wann = sum(count_wann_per_atom(el) for el in all_atoms_in_cell)
    nbnd = max(int(num_wann * 1.5), nbnd_input)
    nbnd = int((nbnd + 3) // 4 * 4)
    return num_wann, nbnd


# =======================================================================
# 6. Energy window estimation (initial placeholders; refined by update_win.py)
# =======================================================================

_BAND_WIDTH_EV = {
    "d_metal": {"win_max": 60.0, "froz_max": 20.0},
    "main":    {"win_max": 40.0, "froz_max": 15.0},
}

def estimate_energy_windows(unique_elements):
    """Return conservative initial values for dis_win_max and dis_froz_max."""
    has_d_metal = any(el in D_METALS for el in unique_elements)
    key = "d_metal" if has_d_metal else "main"
    return {
        "dis_win_max":  _BAND_WIDTH_EV[key]["win_max"],
        "dis_froz_max": _BAND_WIDTH_EV[key]["froz_max"],
    }


# =======================================================================
# 7. k-point mesh and k-path generation via seekpath
# =======================================================================

def generate_kpoints(n):
    """Uniform n x n x n k-point mesh (fractional coords + weights)."""
    w = 1.0 / n**3
    lines = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                lines.append(f"  {x/n:12.8f}  {y/n:12.8f}  {z/n:12.8f}  {w:14.10f}")
    return "\n".join(lines) + "\n"


def generate_kpath_block_seekpath(sp_result, npoints_per_segment=40):
    """
    Build the K_POINTS crystal_b body for bands.in from a seekpath result.

    seekpath returns:
        point_coords : dict  label -> [kx, ky, kz]  (in primitive reciprocal coords)
        path         : list of (label_start, label_end) segment tuples

    The crystal_b format requires an ordered list of high-symmetry points
    with the number of interpolation points per segment. Branch discontinuities
    (two consecutive different segments that do not share an endpoint) are
    marked by repeating the endpoint with npoints=1.

    Parameters
    ----------
    sp_result           : dict  seekpath output from seekpath.get_path()
    npoints_per_segment : int   k-points per segment (default 40)

    Returns
    -------
    kpath_block : str   Text block after "K_POINTS crystal_b"
    label_str   : str   Human-readable path, e.g. "GAMMA-X-U|K-GAMMA-L-W-X"
    """
    point_coords = sp_result["point_coords"]
    path         = sp_result["path"]

    # Build ordered list of (label, coords, npoints) for crystal_b
    kpt_list = []
    for i, (start, end) in enumerate(path):
        if i == 0:
            kpt_list.append((start, point_coords[start], npoints_per_segment))
        else:
            prev_end = path[i - 1][1]
            if start != prev_end:
                # Branch discontinuity: insert a terminating point then restart
                kpt_list[-1] = (kpt_list[-1][0], kpt_list[-1][1], 1)
                kpt_list.append((start, point_coords[start], npoints_per_segment))
        kpt_list.append((end, point_coords[end], npoints_per_segment))

    # Final point always has npoints = 1
    if kpt_list:
        last = kpt_list[-1]
        kpt_list[-1] = (last[0], last[1], 1)

    lines = [str(len(kpt_list))]
    for label, coords, npts in kpt_list:
        lines.append(
            f"  {coords[0]:8.5f}  {coords[1]:8.5f}  {coords[2]:8.5f}  {npts:4d}"
            f"  ! {label}"
        )
    kpath_block = "\n".join(lines)

    # Human-readable label string
    label_parts = []
    for i, (label, _, npts) in enumerate(kpt_list):
        if i == 0:
            label_parts.append(label)
        elif npts == 1 and i < len(kpt_list) - 1:
            # This is a branch-end point; next will be branch-start -> use "|"
            label_parts.append(f"|{label}")
        elif label_parts and label_parts[-1].endswith(f"|{label}"):
            pass  # already appended as discontinuity marker
        else:
            label_parts.append(f"-{label}")
    label_str = "".join(label_parts)

    return kpath_block, label_str


# =======================================================================
# 8. Plane-wave cutoff selection
#
# Cutoffs are estimated from the pseudopotential type and the heaviest
# element in the formula, rather than hardcoded per material name.
#
# Norm-conserving fully-relativistic PSLib pseudopotentials generally
# require ecutwfc >= 60 Ry. d-metals with hard cores (Pt, W) benefit
# from ecutwfc = 90 Ry to converge total energies to < 1 meV/atom.
# Light main-group elements (Ga, As, Se, Bi) converge well at 60 Ry.
# =======================================================================

# Elements whose pseudopotentials require higher cutoffs
_HARD_ELEMENTS = {"Pt", "W", "Ir", "Os", "Re", "Au", "Pd", "Rh", "Ru", "Mo", "Ta", "Hf"}

def estimate_ecut(unique_elements):
    """
    Return (ecutwfc, ecutrho) in Ry based on the elements present.

    Hard d-metals: 90 / 1080 Ry  (following Qiao et al. PRB 98, 214402)
    Others       : 60 /  480 Ry
    """
    if any(el in _HARD_ELEMENTS for el in unique_elements):
        return 90.0, 1080.0
    return 60.0, 480.0


# =======================================================================
# 9. Valence electron count estimation
# =======================================================================

ZVAL_MAP = {
    "Pt": 10, "W": 14, "Au": 11, "Pd": 10, "Ir":  9,
    "Ga": 13, "As":  5, "Bi": 15, "Se":  6, "Fe":  8,
    "Co":  9, "Ni": 10, "Mo": 14, "Ta": 13, "Hf": 12,
}

def estimate_nelec(all_atoms_in_cell):
    """Sum valence electrons over all atoms in the primitive cell."""
    return sum(ZVAL_MAP.get(el, 10) for el in all_atoms_in_cell)


# =======================================================================
# 10. Main
# =======================================================================

def main(formula: str):
    elements = parse_formula(formula)
    unique_elements = sorted(set(elements))
    prefix   = f"{formula.lower()}_scf"
    seedname = formula.lower()

    # Directory layout
    base_dir  = os.path.abspath(f"./calc/{formula}")
    scf_dir   = os.path.join(base_dir, "01_scf")
    nscf_dir  = os.path.join(base_dir, "02_nscf")
    wan_dir   = os.path.join(base_dir, "03_wannier")
    bands_dir = os.path.join(base_dir, "04_bands")
    cif_dir   = os.path.join(base_dir, f"{formula}_selected_cifs")

    for d in [scf_dir, nscf_dir, wan_dir, bands_dir]:
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Step A: Fetch CIF
    # ------------------------------------------------------------------
    print(f"\n[1/5] Fetching CIF for {formula} from AFLOW...")
    cif_path = get_aflow_cif(chemical_formula=formula, output_dir=cif_dir, spacegroup=225)
    if not cif_path:
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step B: CIF -> conventional cell via cif2cell
    # ------------------------------------------------------------------
    print(f"\n[2/5] Converting CIF -> primitive cell via cif2cell + spglib + seekpath...")
    raw = run_cif2cell_and_extract(cif_path, base_dir)

    # ------------------------------------------------------------------
    # Step C: Primitive cell reduction + k-path via seekpath
    # ------------------------------------------------------------------
    geo = geo_to_primitive(raw)

    nat  = geo["nat"]
    ntyp = geo["ntyp"]
    brav = geo["bravais_lattice"]
    sg   = geo["spacegroup_number"]
    print(f"  Primitive cell: nat={nat}, ntyp={ntyp}")
    print(f"  Bravais lattice: {brav}  (SG {sg})")

    # Sanity check: for a pure-element FCC material (SG 225) the primitive cell
    # must have exactly 1 atom.  Abort early with a clear message if not.
    if sg == 225 and ntyp == 1 and nat != 1:
        print(
            f"\n  ERROR: SG 225 single-element material should have nat=1 in the\n"
            f"  primitive cell, but got nat={nat}.\n"
            f"  This usually means the AFLOW CIF has displaced corner atoms that\n"
            f"  spglib could not collapse.  Inspect the CIF and check symprec."
        )
        sys.exit(1)

    # Build full atom list from the primitive cell positions block
    all_atoms_in_cell = [
        line.split()[0]
        for line in geo["atomic_positions_block"].splitlines()
        if line.strip() and "ATOMIC_POSITIONS" not in line
    ]
    print(f"  Atoms in primitive cell: {all_atoms_in_cell}")

    # ------------------------------------------------------------------
    # Step D: Pseudopotentials
    # ------------------------------------------------------------------
    print(f"\n[3/5] Resolving pseudopotentials...")
    patched_species = patch_atomic_species_block(
        geo["atomic_species_block"], unique_elements, scf_dir
    )
    import shutil
    for upf_file in os.listdir(scf_dir):
        if upf_file.endswith(".UPF"):
            for dest in [nscf_dir, wan_dir, bands_dir]:
                dest_path = os.path.join(dest, upf_file)
                if not os.path.exists(dest_path):
                    shutil.copy(os.path.join(scf_dir, upf_file), dest_path)

    # ------------------------------------------------------------------
    # Step E: Wannier and cutoff parameter estimation
    # ------------------------------------------------------------------
    nelec    = estimate_nelec(all_atoms_in_cell)
    nbnd_min = int((nelec * 1.2 + 3) // 4 * 4)
    nbnd_min = max(nbnd_min, 20)

    num_wann, nbnd = estimate_wannier_params(all_atoms_in_cell, nbnd_min)
    proj_block     = build_projections_block(unique_elements)
    ewin           = estimate_energy_windows(unique_elements)
    ecutwfc, ecutrho = estimate_ecut(unique_elements)

    print(f"  nelec={nelec}, nbnd_min={nbnd_min}")
    print(f"  num_wann={num_wann}, nbnd={nbnd}")
    print(f"  projections:\n{proj_block}")
    print(f"  dis_win_max={ewin['dis_win_max']}, dis_froz_max={ewin['dis_froz_max']}")
    print(f"  ecutwfc={ecutwfc} Ry, ecutrho={ecutrho} Ry")

    # ------------------------------------------------------------------
    # Step F: Write SCF / NSCF / Wannier input files
    # ------------------------------------------------------------------
    print(f"\n[4/5] Writing SCF / NSCF / Wannier input files...")

    # scf.in
    scf_content = SCF_TEMPLATE.format(
        prefix=prefix, nat=nat, ntyp=ntyp, nbnd=nbnd,
        ecutwfc=ecutwfc, ecutrho=ecutrho,
        atomic_species_block=patched_species,
        cell_params_block=geo["cell_params_block"],
        atomic_positions_block=geo["atomic_positions_block"],
    )
    with open(os.path.join(scf_dir, "scf.in"), "w") as f:
        f.write(scf_content)
    print(f"  Written: {os.path.join(scf_dir, 'scf.in')}")

    # nscf.in
    wan_kmesh = 8
    kpts_raw  = generate_kpoints(wan_kmesh)
    nscf_content = NSCF_TEMPLATE.format(
        prefix=prefix, nat=nat, ntyp=ntyp, nbnd=nbnd,
        ecutwfc=ecutwfc, ecutrho=ecutrho,
        atomic_species_block=patched_species,
        cell_params_block=geo["cell_params_block"],
        atomic_positions_block=geo["atomic_positions_block"],
        nk=wan_kmesh ** 3,
        kpoints_list=kpts_raw,
    )
    with open(os.path.join(nscf_dir, "nscf.in"), "w") as f:
        f.write(nscf_content)
    print(f"  Written: {os.path.join(nscf_dir, 'nscf.in')}")

    # pw2wan.in
    with open(os.path.join(wan_dir, "pw2wan.in"), "w") as f:
        f.write(PW2WAN_TEMPLATE.format(prefix=prefix, seedname=seedname))
    print(f"  Written: {os.path.join(wan_dir, 'pw2wan.in')}")

    # .win (Wannier90)
    atoms_frac_lines = "\n".join(
        f"  {line.split()[0]}  {' '.join(line.split()[1:])}"
        for line in geo["atomic_positions_block"].splitlines()
        if line.strip() and "ATOMIC_POSITIONS" not in line
    )
    win_content = WAN_TEMPLATE.format(
        num_bands=nbnd, num_wann=num_wann,
        unit_cell_lines=geo["unit_cell_lines"],
        atoms_frac_block=atoms_frac_lines,
        projections_block=proj_block,
        dis_win_max=ewin["dis_win_max"],
        dis_froz_max=ewin["dis_froz_max"],
        kpoints_block=generate_kpoints(wan_kmesh),
    )
    with open(os.path.join(wan_dir, f"{seedname}.win"), "w") as f:
        f.write(win_content)
    print(f"  Written: {os.path.join(wan_dir, f'{seedname}.win')}")

    # ------------------------------------------------------------------
    # Step G: Write band-structure input files
    # ------------------------------------------------------------------
    print(f"\n[5/5] Writing band-structure input files (04_bands/)...")

    kpath_block, label_str = generate_kpath_block_seekpath(
        geo["seekpath_result"], npoints_per_segment=40
    )
    print(f"  Bravais lattice : {brav}")
    print(f"  k-path          : {label_str}")

    bands_content = BANDS_TEMPLATE.format(
        prefix=prefix, nat=nat, ntyp=ntyp, nbnd=nbnd,
        ecutwfc=ecutwfc, ecutrho=ecutrho,
        atomic_species_block=patched_species,
        cell_params_block=geo["cell_params_block"],
        atomic_positions_block=geo["atomic_positions_block"],
        kpath_block=kpath_block,
    )
    with open(os.path.join(bands_dir, "bands.in"), "w") as f:
        f.write(bands_content)
    print(f"  Written: {os.path.join(bands_dir, 'bands.in')}")

    with open(os.path.join(bands_dir, "bands.pp.in"), "w") as f:
        f.write(BANDSPP_TEMPLATE.format(prefix=prefix, seedname=seedname))
    print(f"  Written: {os.path.join(bands_dir, 'bands.pp.in')}")

    # ------------------------------------------------------------------
    print(f"\n All input files generated for {formula}")
    print(f"   Next: python run_pipeline.py {formula}")

    margin = nbnd - num_wann
    print(f"\n--- Consistency Check ---")
    print(f"  num_wann  = {num_wann}  ({', '.join(unique_elements)} x orbitals x 2 spinor)")
    print(f"  num_bands = {nbnd}  (margin: +{margin})")
    print(f"  ecutwfc   = {ecutwfc} Ry,  ecutrho = {ecutrho} Ry")
    print(f"  dis_win_max / dis_froz_max : placeholders; auto-updated by update_win.py")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python setup_material.py <Formula>  (e.g. Pt, GaAs, W, Bi2Se3)")
    main(sys.argv[1])