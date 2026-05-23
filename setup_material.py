import os
import sys
import re
import glob
import subprocess
import requests
from urllib.request import urlretrieve

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
# 2. cif2cell Integration Logic
# =======================================================================

def run_cif2cell_and_extract(cif_path, work_dir):
    """
    Run cif2cell and extract geometry blocks.
    Returns CELL_PARAMETERS angstrom (vectors already scaled by A).
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

    # Lattice scale A (Angstroms)
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
        print("WARNING: Could not find lattice parameter A in cif2cell output.")
        print("         Defaulting to A = 1.0 — please check CELL_PARAMETERS manually.")
        a_value = 1.0

    print(f"  Detected lattice parameter A = {a_value:.6f} A")

    def extract_block(text, header_keyword, stop_keywords):
        lines = text.splitlines()
        block = []
        recording = False
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

    atomic_species_block = extract_block(
        content, "ATOMIC_SPECIES",
        ["ATOMIC_POSITIONS", "CELL_PARAMETERS", "K_POINTS"]
    )
    atomic_positions_block = extract_block(
        content, "ATOMIC_POSITIONS",
        ["K_POINTS", "CELL_PARAMETERS", "ATOMIC_SPECIES"]
    )

    raw_cell_block = extract_block(
        content, "CELL_PARAMETERS",
        ["ATOMIC_SPECIES", "ATOMIC_POSITIONS", "K_POINTS"]
    )

    raw_vector_lines = [
        l for l in raw_cell_block.splitlines()
        if l.strip() and "CELL_PARAMETERS" not in l
    ]

    converted_vectors = []
    for vline in raw_vector_lines[:3]:
        nums = vline.split()
        if len(nums) >= 3:
            scaled = [float(x) * a_value for x in nums[:3]]
            converted_vectors.append(
                f"  {scaled[0]:18.12f}  {scaled[1]:18.12f}  {scaled[2]:18.12f}"
            )

    cell_params_block = "CELL_PARAMETERS angstrom\n" + "\n".join(converted_vectors)
    unit_cell_lines   = "\n".join(converted_vectors)

    nat = sum(
        1 for line in atomic_positions_block.splitlines()
        if line.strip() and "ATOMIC_POSITIONS" not in line
    )
    species_lines = [
        l for l in atomic_species_block.splitlines()
        if l.strip() and "ATOMIC_SPECIES" not in l
    ]
    ntyp = len(species_lines)

    return {
        "cell_params_block":       cell_params_block,
        "atomic_species_block":    atomic_species_block,
        "atomic_positions_block":  atomic_positions_block,
        "nat":                     nat,
        "ntyp":                    ntyp,
        "unit_cell_lines":         unit_cell_lines,
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
}
UPF_BASE_URL = "https://pseudopotentials.quantum-espresso.org/upf_files/"

KNOWN_STRUCTURES = {
    "Pt": {
        "cell_params_block": (
            "CELL_PARAMETERS angstrom\n"
            "  3.924200  0.000000  0.000000\n"
            "  0.000000  3.924200  0.000000\n"
            "  0.000000  0.000000  3.924200"
        ),
        "atomic_species_block": (
            "ATOMIC_SPECIES\n"
            "  Pt  195.08  Pt.rel-pbe-n-rrkjus_psl.1.0.0.UPF"
        ),
        "atomic_positions_block": (
            "ATOMIC_POSITIONS crystal\n"
            "  Pt  0.00  0.00  0.00\n"
            "  Pt  0.50  0.50  0.00\n"
            "  Pt  0.50  0.00  0.50\n"
            "  Pt  0.00  0.50  0.50"
        ),
        "nat": 4,
        "ntyp": 1,
        "unit_cell_lines": (
            "  3.924200  0.000000  0.000000\n"
            "  0.000000  3.924200  0.000000\n"
            "  0.000000  0.000000  3.924200"
        ),
    },
    "W": {
        "cell_params_block": (
            "CELL_PARAMETERS angstrom\n"
            "  3.165000  0.000000  0.000000\n"
            "  0.000000  3.165000  0.000000\n"
            "  0.000000  0.000000  3.165000"
        ),
        "atomic_species_block": (
            "ATOMIC_SPECIES\n"
            "  W  183.84  W.rel-pbe-spn-rrkjus_psl.1.0.0.UPF"
        ),
        "atomic_positions_block": (
            "ATOMIC_POSITIONS crystal\n"
            "  W  0.00  0.00  0.00\n"
            "  W  0.50  0.50  0.50"
        ),
        "nat": 2,
        "ntyp": 1,
        "unit_cell_lines": (
            "  3.165000  0.000000  0.000000\n"
            "  0.000000  3.165000  0.000000\n"
            "  0.000000  0.000000  3.165000"
        ),
    },
}

def validate_geo(geo):
    pos_lines = [
        l for l in geo["atomic_positions_block"].splitlines()
        if l.strip() and "ATOMIC_POSITIONS" not in l
    ]
    if len(pos_lines) != geo["nat"]:
        return False, f"nat={geo['nat']} but {len(pos_lines)} position lines"
    for line in pos_lines:
        coords = line.split()[1:]
        for c in coords:
            try:
                v = float(c)
                if 0.95 < abs(v) < 1.0:
                    return False, f"suspicious coordinate {c} in: {line.strip()}"
            except ValueError:
                pass
    return True, "OK"


def get_element_meta(element):
    if element in ELEMENT_META:
        return ELEMENT_META[element]
    return {
        "mass": "100.0",
        "upf": f"{element}.rel-pbe-n-rrkjus_psl.1.0.0.UPF"
    }

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
    ecutwfc     = 60.0
    ecutrho     = 480.0
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
    ecutwfc     = 60.0
    ecutrho     = 480.0
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
#   - K_POINTS uses crystal_b format (high-symmetry point list)
#
# Output files written to 04_bands/ to match plot_bands.py expectations:
#   bands.dat      -- binary eigenvalue data
#   bands.dat.gnu  -- plain-text k-distance vs energy, parsed by plot_bands.py
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
    ecutwfc     = 60.0
    ecutrho     = 480.0
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
# filband base name must be 'bands.dat' so bands.x writes bands.dat.gnu,
# which is the file expected by plot_bands.py.
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
dis_num_iter  = 200
num_iter      = 500
spinors       = .true.
write_hr      = .true.
mp_grid       = 8 8 8
search_shells = 40

begin kpoints
{kpoints_block}end kpoints
"""


# =======================================================================
# 5. Wannier parameter estimation
#    Projections and num_wann are derived per-atom to stay consistent.
# =======================================================================

# d-transition metals (projected onto s + p + d orbitals)
D_METALS = {"Pt", "W", "Au", "Pd", "Ir", "Rh", "Ru", "Os", "Fe", "Co", "Ni", "Mn", "Mo", "Ta", "Hf"}

# Projection orbital sets per element type
PROJECTION_ORBITALS = {
    "d_metal": ["s", "p", "d"],   # 1+3+5 = 9 orbitals x 2 (spinor) = 18 per atom
    "main":    ["s", "p"],        # 1+3   = 4 orbitals x 2 (spinor) =  8 per atom
}
ORB_COUNT = {"s": 1, "p": 3, "d": 5}

def get_orbitals_for_element(el):
    """Return the projection orbital list for an element (d-metal: s,p,d / others: s,p)."""
    if el in D_METALS:
        return PROJECTION_ORBITALS["d_metal"]
    return PROJECTION_ORBITALS["main"]

def count_wann_per_atom(el):
    """Return the number of Wannier functions per atom (x2 for SOC spinors)."""
    orbs = get_orbitals_for_element(el)
    return sum(ORB_COUNT[o] for o in orbs) * 2

def build_projections_block(unique_elements):
    """
    Build the Wannier90 projections block.
    One line per unique element; Wannier90 applies it to all atoms of that species.
    """
    lines = []
    for el in unique_elements:
        orbs = get_orbitals_for_element(el)
        lines.append(f"  {el} : " + " ; ".join(orbs))
    return "\n".join(lines)

def estimate_wannier_params(all_atoms_in_cell, nbnd_input):
    """
    Compute num_wann and nbnd consistent with the projections block.

    num_wann = sum of Wannier functions across all atoms in the cell,
               matching projections_block exactly.
    nbnd     = max(num_wann x 1.5, nbnd_input) rounded up to a multiple of 4.

    Using num_wann x 1.5 (rather than num_wann + 4) ensures that the number
    of bands inside the disentanglement window stays >= num_wann at all k-points.

    Parameters
    ----------
    all_atoms_in_cell : list of str
        All atoms in the unit cell (e.g. ['Pt', 'Pt', 'Pt', 'Pt']).
    nbnd_input : int
        Lower bound on the number of bands from nelec x 1.2.

    Returns
    -------
    num_wann : int   Total number of Wannier functions from projections.
    nbnd     : int   Number of bands sufficient to contain num_wann (multiple of 4).
    """
    num_wann = sum(count_wann_per_atom(el) for el in all_atoms_in_cell)

    # nbnd >= num_wann x 1.5 and >= nbnd_input, rounded up to multiple of 4
    nbnd = max(int(num_wann * 1.5), nbnd_input)
    nbnd = int((nbnd + 3) // 4 * 4)   # round up to nearest multiple of 4

    return num_wann, nbnd


# =======================================================================
# 6. Frozen / outer energy window estimation
#    dis_win_max is set conservatively large so all nbnd bands are included.
#    Adjust manually after SCF using the actual band energies.
# =======================================================================

# Empirical bandwidth above the Fermi level (eV) by element type.
# d-metals need a wider window due to broad d-bands and SOC splitting.
_BAND_WIDTH_EV = {
    "d_metal": {"win_max": 60.0, "froz_max": 20.0},
    "main":    {"win_max": 40.0, "froz_max": 15.0},
}

def estimate_energy_windows(unique_elements):
    """
    Return initial values for dis_win_max and dis_froz_max.

    dis_win_max is set large (d-metal: 60 eV, main-group: 40 eV) to prevent
    the 'number of bands in window < num_wann' error during disentanglement.
    These are absolute energies as output by QE (not relative to the Fermi level).
    After SCF, check the Fermi energy in scf.out and set:
        dis_froz_max = E_Fermi + 2-5 eV
        dis_win_max  <= energy of the nbnd-th band at all k-points
    """
    has_d_metal = any(el in D_METALS for el in unique_elements)
    key = "d_metal" if has_d_metal else "main"
    return {
        "dis_win_max":  _BAND_WIDTH_EV[key]["win_max"],
        "dis_froz_max": _BAND_WIDTH_EV[key]["froz_max"],
    }


# =======================================================================
# 7. k-point mesh and k-path generation
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


# High-symmetry k-point coordinates in crystal (fractional) coordinates.
# Paths follow the conventions of Setyawan & Curtarolo, CMS 49 (2010).

_KPOINTS_FCC = [
    # Gamma - X - U|K - Gamma - L - W - X
    ("Gamma", [0.000,  0.000,  0.000]),
    ("X",     [0.500,  0.000,  0.500]),
    ("U",     [0.625,  0.250,  0.625]),
    ("K",     [0.375,  0.375,  0.750]),
    ("Gamma", [0.000,  0.000,  0.000]),
    ("L",     [0.500,  0.500,  0.500]),
    ("W",     [0.500,  0.250,  0.750]),
    ("X",     [0.500,  0.000,  0.500]),
]

_KPOINTS_BCC = [
    # Gamma - H - N - Gamma - P - H
    ("Gamma", [0.000,  0.000,  0.000]),
    ("H",     [0.500, -0.500,  0.500]),
    ("N",     [0.000,  0.000,  0.500]),
    ("Gamma", [0.000,  0.000,  0.000]),
    ("P",     [0.250,  0.250,  0.250]),
    ("H",     [0.500, -0.500,  0.500]),
]

_KPOINTS_HCP = [
    # Gamma - M - K - Gamma - A - L - H - A
    ("Gamma", [0.000,  0.000,  0.000]),
    ("M",     [0.500,  0.000,  0.000]),
    ("K",     [0.333,  0.333,  0.000]),
    ("Gamma", [0.000,  0.000,  0.000]),
    ("A",     [0.000,  0.000,  0.500]),
    ("L",     [0.500,  0.000,  0.500]),
    ("H",     [0.333,  0.333,  0.500]),
    ("A",     [0.000,  0.000,  0.500]),
]

_KPOINTS_SC = [
    # Gamma - X - M - Gamma - R - X  (simple cubic fallback)
    ("Gamma", [0.000,  0.000,  0.000]),
    ("X",     [0.500,  0.000,  0.000]),
    ("M",     [0.500,  0.500,  0.000]),
    ("Gamma", [0.000,  0.000,  0.000]),
    ("R",     [0.500,  0.500,  0.500]),
    ("X",     [0.500,  0.000,  0.000]),
]

# Space groups belonging to the BCC Bravais lattice (cubic body-centred)
_BCC_SPACEGROUPS = {197, 199, 204, 206, 211, 214, 217, 220, 229, 230}

# Space groups belonging to the hexagonal / HCP family
_HEX_SPACEGROUPS = {168, 169, 170, 171, 172, 173, 174, 175, 176,
                    177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
                    187, 188, 189, 190, 191, 192, 193, 194}

def _kpath_for_spacegroup(spacegroup):
    """
    Select the k-path and Bravais-lattice label for a given space group number.

    Returns
    -------
    lattice_key : str   One of "FCC", "BCC", "HCP", "SC"
    kpts        : list  List of (label, [kx, ky, kz]) tuples
    """
    if spacegroup in _HEX_SPACEGROUPS:
        return "HCP", _KPOINTS_HCP
    if 195 <= spacegroup <= 230:          # cubic family
        if spacegroup in _BCC_SPACEGROUPS:
            return "BCC", _KPOINTS_BCC
        return "FCC", _KPOINTS_FCC       # FCC covers cF and cP not in BCC set
    return "SC", _KPOINTS_SC             # generic fallback


def generate_kpath_block(spacegroup=225, npoints_per_segment=40):
    """
    Build the K_POINTS crystal_b body for bands.in.

    crystal_b format (QE convention):
      Line 1 : total number of high-symmetry points
      Lines 2+: kx  ky  kz  npoints
                The final point must have npoints = 1 (no segment after it).

    Parameters
    ----------
    spacegroup          : int   Space group number (default 225 = FCC)
    npoints_per_segment : int   k-points per segment (default 40)

    Returns
    -------
    kpath_block : str   Text block to insert after "K_POINTS crystal_b"
    label_str   : str   Human-readable path string, e.g. "G-X-U|K-G-L-W-X"
    lattice_key : str   Bravais lattice identifier ("FCC", "BCC", "HCP", "SC")
    """
    lattice_key, kpts = _kpath_for_spacegroup(spacegroup)
    n_pts = len(kpts)

    lines = [str(n_pts)]
    for i, (label, coords) in enumerate(kpts):
        # Final point terminates the path with npoints = 1
        np_seg = 1 if i == n_pts - 1 else npoints_per_segment
        lines.append(
            f"  {coords[0]:8.5f}  {coords[1]:8.5f}  {coords[2]:8.5f}  {np_seg:4d}"
            f"  ! {label}"
        )

    kpath_block = "\n".join(lines)

    # Build a human-readable label string; mark branch discontinuities with "|"
    labels = [lbl for lbl, _ in kpts]
    label_parts = [labels[0]]
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            label_parts.append("|" + labels[i])
        else:
            label_parts.append("-" + labels[i])
    label_str = "".join(label_parts).replace("Gamma", "G")

    return kpath_block, label_str, lattice_key


# =======================================================================
# 8. Valence electron count estimation
# =======================================================================

ZVAL_MAP = {
    "Pt": 10, "W": 14, "Au": 11, "Pd": 10, "Ir":  9,
    "Ga": 13, "As":  5, "Bi": 15, "Se":  6, "Fe":  8,
    "Co":  9, "Ni": 10, "Mo": 14, "Ta": 13, "Hf": 12,
}

def estimate_nelec(all_atoms_in_cell):
    """
    Sum the valence electron count over all atoms in the unit cell.
    Iterates over all_atoms_in_cell directly to avoid double-counting
    that arose from the old unique_elements x (nat / len(elements)) approach.
    """
    return sum(ZVAL_MAP.get(el, 10) for el in all_atoms_in_cell)


# =======================================================================
# 9. Main
# =======================================================================

def main(formula: str):
    elements = parse_formula(formula)
    unique_elements = sorted(set(elements))
    prefix   = f"{formula.lower()}_scf"
    seedname = formula.lower()

    # Directory layout
    # 01_scf    : SCF ground state
    # 02_nscf   : non-SCF on dense k-mesh for Wannierization
    # 03_wannier : Wannier90 + pw2wannier90
    # 04_bands  : band-structure pw.x run + bands.x post-processing
    #             (numbered 05 to match plot_bands.py and run_all_shc_pipeline.py)
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
    # Step B: CIF -> QE geometry
    # ------------------------------------------------------------------
    print(f"\n[2/5] Converting CIF to QE format via cif2cell...")
    geo = run_cif2cell_and_extract(cif_path, base_dir)

    ok, reason = validate_geo(geo)
    if not ok:
        print(f"  WARNING: cif2cell geometry rejected ({reason})")
        if formula in KNOWN_STRUCTURES:
            print(f"  Using built-in structure for {formula}")
            geo = KNOWN_STRUCTURES[formula]
        else:
            print(f"  No fallback for {formula}. Please check the CIF manually.")
            sys.exit(1)
    else:
        print(f"  Geometry OK: {reason}")

    nat  = geo["nat"]
    ntyp = geo["ntyp"]
    print(f"  nat={nat}, ntyp={ntyp}")

    # ------------------------------------------------------------------
    # Step C: Build full atom list for the unit cell
    # e.g. formula='GaAs' (2 species), nat=8 -> ['Ga','As','Ga','As',...]
    # e.g. formula='Pt'   (1 species), nat=4 -> ['Pt','Pt','Pt','Pt']
    # ------------------------------------------------------------------
    if nat % len(elements) == 0:
        repeat = nat // len(elements)
        all_atoms_in_cell = elements * repeat
    else:
        # For complex structures where nat is not divisible by number of species,
        # extract element labels directly from the atomic positions block.
        all_atoms_in_cell = [
            line.split()[0]
            for line in geo["atomic_positions_block"].splitlines()
            if line.strip() and "ATOMIC_POSITIONS" not in line
        ]

    print(f"  All atoms in cell: {all_atoms_in_cell}")

    # ------------------------------------------------------------------
    # Step D: Pseudopotentials
    # ------------------------------------------------------------------
    print(f"\n[3/5] Resolving pseudopotentials...")
    patched_species = patch_atomic_species_block(
        geo["atomic_species_block"], unique_elements, scf_dir
    )
    for upf_file in os.listdir(scf_dir):
        if upf_file.endswith(".UPF"):
            for dest in [nscf_dir, wan_dir, bands_dir]:
                dest_path = os.path.join(dest, upf_file)
                if not os.path.exists(dest_path):
                    import shutil
                    shutil.copy(os.path.join(scf_dir, upf_file), dest_path)

    # ------------------------------------------------------------------
    # Step E: Wannier parameter estimation
    # ------------------------------------------------------------------
    nelec    = estimate_nelec(all_atoms_in_cell)
    nbnd_min = int((nelec * 1.2 + 3) // 4 * 4)   # rough lower bound (multiple of 4)
    nbnd_min = max(nbnd_min, 20)

    # num_wann is derived from all_atoms_in_cell x projected orbitals,
    # and nbnd is auto-adjusted to be safely larger than num_wann.
    num_wann, nbnd = estimate_wannier_params(all_atoms_in_cell, nbnd_min)

    print(f"  nelec={nelec}, nbnd_min(1.2x)={nbnd_min}")
    print(f"  num_wann={num_wann}  (= sum of atoms x orbitals x 2 spinor)")
    print(f"  nbnd={nbnd}          (>= num_wann, multiple of 4)")

    proj_block = build_projections_block(unique_elements)
    print(f"  projections:\n{proj_block}")

    ewin = estimate_energy_windows(unique_elements)
    print(f"  dis_win_max={ewin['dis_win_max']}, dis_froz_max={ewin['dis_froz_max']}")

    # ------------------------------------------------------------------
    # Step F: Write SCF / NSCF / Wannier input files
    # ------------------------------------------------------------------
    print(f"\n[4/5] Writing SCF / NSCF / Wannier input files...")

    # scf.in
    scf_content = SCF_TEMPLATE.format(
        prefix=prefix, nat=nat, ntyp=ntyp, nbnd=nbnd,
        atomic_species_block=patched_species,
        cell_params_block=geo["cell_params_block"],
        atomic_positions_block=geo["atomic_positions_block"],
    )
    scf_in_path = os.path.join(scf_dir, "scf.in")
    with open(scf_in_path, "w") as f:
        f.write(scf_content)
    print(f"  Written: {scf_in_path}")

    # nscf.in
    wan_kmesh = 8
    kpts_raw  = generate_kpoints(wan_kmesh)
    nscf_content = NSCF_TEMPLATE.format(
        prefix=prefix, nat=nat, ntyp=ntyp, nbnd=nbnd,
        atomic_species_block=patched_species,
        cell_params_block=geo["cell_params_block"],
        atomic_positions_block=geo["atomic_positions_block"],
        nk=wan_kmesh ** 3,
        kpoints_list=kpts_raw,
    )
    nscf_in_path = os.path.join(nscf_dir, "nscf.in")
    with open(nscf_in_path, "w") as f:
        f.write(nscf_content)
    print(f"  Written: {nscf_in_path}")

    # pw2wan.in
    pw2wan_content = PW2WAN_TEMPLATE.format(prefix=prefix, seedname=seedname)
    pw2wan_path = os.path.join(wan_dir, "pw2wan.in")
    with open(pw2wan_path, "w") as f:
        f.write(pw2wan_content)
    print(f"  Written: {pw2wan_path}")

    # .win (Wannier90)
    atoms_frac_lines = "\n".join(
        f"  {line.split()[0]}  {' '.join(line.split()[1:])}"
        for line in geo["atomic_positions_block"].splitlines()
        if line.strip() and "ATOMIC_POSITIONS" not in line
    )
    win_content = WAN_TEMPLATE.format(
        num_bands=nbnd,
        num_wann=num_wann,
        unit_cell_lines=geo["unit_cell_lines"],
        atoms_frac_block=atoms_frac_lines,
        projections_block=proj_block,
        dis_win_max=ewin["dis_win_max"],
        dis_froz_max=ewin["dis_froz_max"],
        kpoints_block=generate_kpoints(wan_kmesh),
    )
    win_path = os.path.join(wan_dir, f"{seedname}.win")
    with open(win_path, "w") as f:
        f.write(win_content)
    print(f"  Written: {win_path}")

    # ------------------------------------------------------------------
    # Step G: Write band-structure input files
    #
    # bands.in     -- pw.x band run; reads charge density from 01_scf/tmp/
    # bands.pp.in  -- bands.x post-processing; produces bands.dat.gnu
    #                 which is read by plot_bands.py
    #
    # Both files are placed in 04_bands/ to match plot_bands.py and
    # run_all_shc_pipeline.py directory conventions.
    # ------------------------------------------------------------------
    print(f"\n[5/5] Writing band-structure input files (04_bands/)...")

    kpath_block, label_str, lattice_key = generate_kpath_block(
        spacegroup=225, npoints_per_segment=40
    )
    print(f"  Bravais lattice : {lattice_key}")
    print(f"  k-path          : {label_str}")

    bands_content = BANDS_TEMPLATE.format(
        prefix=prefix, nat=nat, ntyp=ntyp, nbnd=nbnd,
        atomic_species_block=patched_species,
        cell_params_block=geo["cell_params_block"],
        atomic_positions_block=geo["atomic_positions_block"],
        kpath_block=kpath_block,
    )
    bands_in_path = os.path.join(bands_dir, "bands.in")
    with open(bands_in_path, "w") as f:
        f.write(bands_content)
    print(f"  Written: {bands_in_path}")

    bandspp_content = BANDSPP_TEMPLATE.format(prefix=prefix, seedname=seedname)
    bandspp_path = os.path.join(bands_dir, "bands.pp.in")
    with open(bandspp_path, "w") as f:
        f.write(bandspp_content)
    print(f"  Written: {bandspp_path}")

    # ------------------------------------------------------------------
    print(f"\n All input files generated for {formula}")
    print(f"   Next step: python run_all_shc_pipeline.py {formula}")

    margin = nbnd - num_wann
    print(f"\n--- Consistency Check ---")
    print(f"  projections  -> {num_wann} Wannier functions  (= sum of atoms x orbitals x 2 spinor)")
    print(f"  num_wann     = {num_wann}")
    print(f"  num_bands    = {nbnd}  (margin: +{margin} bands above num_wann)")
    print(f"  dis_win_max  = {ewin['dis_win_max']} eV  [conservative placeholder; auto-updated by update_win.py]")
    print(f"  dis_froz_max = {ewin['dis_froz_max']} eV  [conservative placeholder; auto-updated by update_win.py]")
    print(f"\n--- Band Structure ---")
    print(f"  Bravais lattice : {lattice_key}")
    print(f"  k-path          : {label_str}  (40 points per segment)")
    print(f"  Run order       : pw.x < bands.in  ->  bands.x < bands.pp.in  ->  python plot_bands.py {formula}")
    print(f"\n--- Next Step ---")
    print(f"  uv run run_pipeline.py {formula}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python setup_material.py <Formula>  (e.g. Pt, GaAs, W, Bi2Se3)")
    main(sys.argv[1])