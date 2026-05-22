import os
import subprocess
import sys
import shutil

def run_command(cmd, log_file, shell=False):
    print(f"🏃 Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    try:
        with open(log_file, "w") as f:
            subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT,
                text=True, check=True, shell=shell
            )
        print(f"✅ Success! Log: {log_file}\n", flush=True)
    except subprocess.CalledProcessError:
        print(f"❌ ERROR. Check: {log_file}", flush=True)
        try:
            with open(log_file) as f:
                lines = f.readlines()
            print("".join(lines[-50:]), flush=True)
        except Exception:
            pass
        sys.exit(1)

def print_snapshot(label):
    print(f"\n{'='*60}\n🔍 [SNAP] {label}")
    print(f"  CWD: {os.getcwd()}")
    for f in sorted(os.listdir(".")):
        print(f"    {f} ({os.path.getsize(f)} B)")
    print("="*60 + "\n")

def dump_file(filename):
    print(f"\n📢 [DUMP] {filename}")
    if os.path.exists(filename):
        with open(filename) as f:
            print(f.read())
    else:
        print(f"  ❌ {filename} does not exist.\n")

def main(formula: str):
    # seedname は formula の小文字（wannier90 の命名規則）
    seedname = formula.lower()

    ROOT_DIR   = os.path.abspath(os.path.dirname(__file__))
    QE_BIN_DIR = os.path.join(ROOT_DIR, "q-e", "bin")

    PW_EX     = os.path.join(QE_BIN_DIR, "pw.x")
    PW2WAN_EX = os.path.join(QE_BIN_DIR, "pw2wannier90.x")
    W90_EX    = os.path.join(QE_BIN_DIR, "wannier90.x")

    base_dir = os.path.join(ROOT_DIR, "calc", formula)
    scf_dir  = os.path.join(base_dir, "01_scf")
    nscf_dir = os.path.join(base_dir, "02_nscf")
    wan_dir  = os.path.join(base_dir, "03_wannier")
    shc_dir  = os.path.join(base_dir, "04_shc")

    # 必須ディレクトリの存在チェック
    for d, label in [(scf_dir, "01_scf"), (nscf_dir, "02_nscf"), (wan_dir, "03_wannier")]:
        if not os.path.isdir(d):
            sys.exit(f"❌ ディレクトリが見つかりません: {d}\n"
                     f"   先に: python setup_material.py {formula}")

    # 入力ファイルの存在チェック
    for path, label in [
        (os.path.join(scf_dir,  "scf.in"),            "scf.in"),
        (os.path.join(nscf_dir, "nscf.in"),           "nscf.in"),
        (os.path.join(wan_dir,  f"{seedname}.win"),   f"{seedname}.win"),
        (os.path.join(wan_dir,  "pw2wan.in"),         "pw2wan.in"),
    ]:
        if not os.path.exists(path):
            sys.exit(f"❌ 入力ファイルが見つかりません: {path}\n"
                     f"   先に: python setup_material.py {formula}")

    print("=" * 60, flush=True)
    print(f"🚀 SHC Pipeline: {formula}", flush=True)
    print("=" * 60 + "\n", flush=True)

    # ------------------------------------------------------------------
    # Step 1: SCF
    # ------------------------------------------------------------------
    print("--- [Step 1/5] SCF ---", flush=True)
    os.chdir(scf_dir)
    os.makedirs("tmp", exist_ok=True)
    run_command(f"mpirun -np 4 {PW_EX} < scf.in", "scf.out", shell=True)

    # ------------------------------------------------------------------
    # Step 2: NSCF
    # nscf.in の outdir = '../01_scf/tmp/' を参照
    # ------------------------------------------------------------------
    print("--- [Step 2/5] NSCF ---", flush=True)
    os.chdir(nscf_dir)
    run_command(f"mpirun -np 4 {PW_EX} < nscf.in", "nscf.out", shell=True)

    # ------------------------------------------------------------------
    # Step 3: Wannier90 前処理 (-pp)
    # ------------------------------------------------------------------
    print("--- [Step 3/5] Wannier90 -pp ---", flush=True)
    os.chdir(wan_dir)

    for trash in [f"{seedname}.werr", f"{seedname}.wout", "CRASH",
                  f"{seedname}.nnkp", "wannier90_pp.out"]:
        if os.path.exists(trash):
            os.remove(trash)

    print_snapshot("BEFORE wannier90 -pp")

    print(f"🏃 Running: {W90_EX} -pp {seedname}", flush=True)
    try:
        with open("wannier90_pp.out", "w") as f:
            subprocess.run(
                [W90_EX, "-pp", seedname],
                stdout=f, stderr=subprocess.STDOUT, text=True, check=True
            )
        print("✅ Success!\n", flush=True)
    except subprocess.CalledProcessError:
        print("❌ wannier90 -pp failed", flush=True)
        print_snapshot("AFTER wannier90 -pp (FAILED)")
        dump_file(f"{seedname}.werr")
        dump_file(f"{seedname}.wout")
        dump_file("wannier90_pp.out")
        sys.exit(1)

    print_snapshot("AFTER wannier90 -pp")

    # ------------------------------------------------------------------
    # Step 4: pw2wannier90.x
    # ------------------------------------------------------------------
    print("--- [Step 4/5] pw2wannier90 ---", flush=True)
    os.chdir(wan_dir)
    run_command(f"mpirun -np 4 {PW2WAN_EX} < pw2wan.in", "pw2wan.out", shell=True)

    # ------------------------------------------------------------------
    # Step 5: Wannier90 本計算
    # ------------------------------------------------------------------
    print("--- [Step 5/5] Wannier90 main ---", flush=True)
    os.chdir(wan_dir)
    run_command([W90_EX, seedname], "wannier90_main.out")

    hr_dat = f"{seedname}_hr.dat"
    if not os.path.exists(hr_dat):
        print(f"❌ {hr_dat} が生成されていません", flush=True)
        sys.exit(1)

    print(f"✨ {hr_dat} 生成完了。次は 04_shc/run_shc.py を実行してください。\n",
          flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python run_pipeline.py <Formula>  (例: Pt, GaAs, W)")
    main(sys.argv[1])