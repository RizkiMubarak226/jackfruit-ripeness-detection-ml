# -- coding: utf-8 --
"""
Evaluator all-in-one untuk 8 model SSD MobileNet (V1_FPN 320/640; V2_FPNLite 320/640; baseline/opt).
- Auto eval TFOD jika belum ada event eval
- Ambil mAP, mAP@0.5, mAP@0.75, AR@100 dari event eval
- Ambil TotalLoss + durasi training dari event train
- Simpan CSV ringkasan
Python 3.9 compatible.
"""

import os
import time
import csv
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# =============== KONFIGURASI PATH (EDIT JIKA PERLU) ===============
# Path ke TF Object Detection API -> model_main_tf2.py
TFOD_MODEL_MAIN = Path(r"C:\Object Detection\TensorFlow\models-master\research\object_detection\model_main_tf2.py")

# Root folder semua model (isi subfolder seperti V1_FPN_320_BASELINE, dst.)
ROOT_MODELS = Path(r"E:\NEW REVISI AFTER SIDANG\rgb_runs_model")

# File CSV output
OUT_CSV = Path(r"E:\NEW REVISI AFTER SIDANG\results\eval_models.csv")

# Nama subfolder yang biasanya dipakai TFOD untuk event evaluasi & training
EVAL_DIRNAME_PREFIX = "eval"
TRAIN_DIRNAME = "train"

# Daftar 8 folder model (HARUS sesuai nama subfolder di ROOT_MODELS)
MODEL_FOLDERS = [
    "V1_FPN_320_BASELINE",
    "V1_FPN_320_OPT",
    "V1_FPN_640_BASELINE",
    "V1_FPN_640_OPT",
    "V2_FPNLite_320_BASELINE",
    "V2_FPNLite_320_OPT",
    "V2_FPNLite_640_BASELINE",
    "V2_FPNLite_640_OPT",
]

# Nama scalar yang biasa muncul di event evaluasi TFOD
EVAL_SCALARS = {
    "mAP": "DetectionBoxes_Precision/mAP",
    "mAP@0.5": "DetectionBoxes_Precision/mAP@0.5IOU",
    "mAP@0.75": "DetectionBoxes_Precision/mAP@0.75IOU",
    "AR@100": "DetectionBoxes_Recall/AR@100",
}

# Kandidat nama scalar total loss di event train (TFOD kadang beda label)
TRAIN_TOTAL_LOSS_CANDIDATES = [
    "Loss/total_loss",
    "total_loss",
    "Train/total_loss",
]

# Jika belum ada event eval, mau auto-jalankan evaluasi TFOD?
AUTO_RUN_EVAL = True

# Argumen evaluasi (silakan sesuaikan num_eval_steps kalau perlu)
EVAL_ARGS = [
    "--alsologtostderr",
    "--num_eval_steps=200"
]
# ================================================================


def sanity_paths() -> None:
    print(f"[sanity] TFOD exists: {TFOD_MODEL_MAIN.exists()}")
    print(f"[sanity] ROOT_MODELS exists: {ROOT_MODELS.exists()}")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Not a path issue? stop early
    if not TFOD_MODEL_MAIN.exists():
        print("[!] model_main_tf2.py tidak ditemukan. Periksa TFOD_MODEL_MAIN.")
    if not ROOT_MODELS.exists():
        print("[!] ROOT_MODELS tidak ditemukan. Periksa path-nya.")


def latest_subdir_eval(model_dir: Path) -> Optional[Path]:
    """Ambil subfolder eval* terbaru (kalau ada)."""
    cands = [p for p in model_dir.iterdir() if p.is_dir() and p.name.lower().startswith(EVAL_DIRNAME_PREFIX)]
    if not cands:
        return None
    return sorted(cands)[-1]


def has_event_files(folder: Path) -> bool:
    if not folder or not folder.exists():
        return False
    for f in folder.iterdir():
        if f.is_file() and f.name.startswith("events.out.tfevents"):
            return True
    return False


def first_event_file(folder: Path) -> Optional[Path]:
    if not folder or not folder.exists():
        return None
    files = [f for f in folder.iterdir() if f.is_file() and f.name.startswith("events.out.tfevents")]
    if not files:
        return None
    return sorted(files)[-1]  # ambil yang terbaru


def ensure_eval(model_dir: Path) -> Optional[Path]:
    """
    Pastikan ada event eval untuk model_dir.
    Jika belum ada dan AUTO_RUN_EVAL True -> jalankan evaluasi TFOD.
    Return: path folder eval (baru/eksisting) atau None kalau gagal.
    """
    eval_dir = latest_subdir_eval(model_dir)
    if eval_dir and has_event_files(eval_dir):
        return eval_dir

    if not AUTO_RUN_EVAL:
        print("   [info] AUTO_RUN_EVAL dimatikan. Lewati.")
        return None

    # Jika belum ada, coba panggil evaluasi TFOD
    print("   [eval] Menjalankan evaluasi TFOD (model_main_tf2.py)…")
    # TFOD hanya butuh --model_dir (ke folder model), opsional --checkpoint_dir untuk evaluasi ckpt terakhir
    cmd = [
        "python",
        str(TFOD_MODEL_MAIN),
        f"--model_dir={str(model_dir)}",
        f"--checkpoint_dir={str(model_dir)}",
    ] + EVAL_ARGS

    try:
        # Jalankan blocking, biar beres dulu baru dibaca eventnya
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"   [eval] GAGAL menjalankan evaluasi TFOD: {e}")
        return None

    # Setelah selesai, cari lagi eval dir + event
    time.sleep(2)
    eval_dir = latest_subdir_eval(model_dir)
    if eval_dir and has_event_files(eval_dir):
        print(f"   [eval] OK, event eval tersedia di: {eval_dir.name}")
        return eval_dir

    print("   [eval] Selesai jalan, tapi belum menemukan event eval.")
    return None


def load_events_scalar_series(event_file: Path) -> Dict[str, List[Tuple[float, float]]]:
    """
    Baca seluruh scalar di sebuah event file pakai tensorboard.event_accumulator.
    Return dict: tag -> list[(step, value)]
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        print("   [warn] TensorBoard belum terpasang. Install: pip install tensorboard")
        return {}

    acc = EventAccumulator(str(event_file))
    try:
        acc.Reload()
    except Exception as e:
        print(f"   [warn] Gagal muat event: {event_file.name}: {e}")
        return {}

    out: Dict[str, List[Tuple[float, float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        out[tag] = [(ev.step, ev.value) for ev in events]
    return out


def pick_last(series: List[Tuple[float, float]]) -> Optional[float]:
    if not series:
        return None
    # ambil nilai step terbesar
    series_sorted = sorted(series, key=lambda x: x[0])
    return float(series_sorted[-1][1])


def collect_eval_metrics(eval_dir: Path) -> Dict[str, Optional[float]]:
    """
    Ambil mAP, mAP@0.5, mAP@0.75, AR@100 dari event eval terbaru.
    """
    ret: Dict[str, Optional[float]] = {k: None for k in EVAL_SCALARS.keys()}
    ev = first_event_file(eval_dir)
    if not ev:
        return ret

    all_scalars = load_events_scalar_series(ev)
    if not all_scalars:
        return ret

    for nice_name, tag in EVAL_SCALARS.items():
        val = pick_last(all_scalars.get(tag, []))
        ret[nice_name] = val
    return ret


def collect_train_loss_and_duration(train_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    """
    Ambil total loss terakhir dan durasi training (detik) dari event train terbaru.
    Durasi = last_wall_time - first_wall_time (jika tersedia).
    """
    ev = first_event_file(train_dir)
    if not ev:
        return None, None

    # Pakai EventAccumulator untuk cari scalar; untuk durasi, kita pakai wall_time dari scalars apa pun (fallback)
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        print("   [warn] TensorBoard belum terpasang. Install: pip install tensorboard")
        return None, None

    acc = EventAccumulator(str(ev))
    try:
        acc.Reload()
    except Exception as e:
        print(f"   [warn] Gagal muat event TRAIN: {ev.name}: {e}")
        return None, None

    # Cari kandidat total_loss
    total_loss: Optional[float] = None
    for cand in TRAIN_TOTAL_LOSS_CANDIDATES:
        if cand in acc.Tags().get("scalars", []):
            series = acc.Scalars(cand)
            if series:
                total_loss = float(series[-1].value)
                break

    # Estimasi durasi: pakai wall_time dari scalar mana saja yang paling panjang
    all_scalar_tags = acc.Tags().get("scalars", [])
    first_t: Optional[float] = None
    last_t: Optional[float] = None
    for tag in all_scalar_tags:
        series = acc.Scalars(tag)
        if not series:
            continue
        ft = series[0].wall_time
        lt = series[-1].wall_time
        if first_t is None or ft < first_t:
            first_t = ft
        if last_t is None or lt > last_t:
            last_t = lt

    duration = (last_t - first_t) if (first_t is not None and last_t is not None) else None
    return total_loss, (float(duration) if duration is not None else None)


def evaluate_one_model(model_name: str, model_dir: Path) -> Dict[str, Optional[float]]:
    print(f"[*] Proses: {model_name}")
    # Cek event TRAIN
    train_dir = model_dir / TRAIN_DIRNAME
    has_train = has_event_files(train_dir)
    # Cek / buat event EVAL
    eval_dir = latest_subdir_eval(model_dir)
    has_eval = has_event_files(eval_dir) if eval_dir else False

    print(f"   [debug] EVAL event: {'ada' if has_eval else 'tidak ada'} | TRAIN event: {'ada' if has_train else 'tidak ada'}")

    if not has_eval:
        eval_dir = ensure_eval(model_dir)
        has_eval = has_event_files(eval_dir) if eval_dir else False
        print(f"   [debug] sesudah ensure_eval -> EVAL: {'ada' if has_eval else 'tidak ada'}")

    # Kumpulkan metrik
    eval_metrics = collect_eval_metrics(eval_dir) if has_eval else {k: None for k in EVAL_SCALARS.keys()}
    totloss, dur = collect_train_loss_and_duration(train_dir) if has_train else (None, None)

    row = {
        "model_name": model_name,
        "mAP": eval_metrics.get("mAP"),
        "mAP@0.5": eval_metrics.get("mAP@0.5"),
        "mAP@0.75": eval_metrics.get("mAP@0.75"),
        "AR@100": eval_metrics.get("AR@100"),
        "TotalLoss": totloss,
        "TrainDuration(s)": dur,
    }
    return row


def fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.4f}"


def main():
    t0 = time.time()
    sanity_paths()

    rows: List[Dict[str, Optional[float]]] = []

    print("\n=== Mulai koleksi metrik ===\n")
    for name in MODEL_FOLDERS:
        model_dir = ROOT_MODELS / name
        if not model_dir.exists():
            print(f"[skip] {name}: folder tidak ditemukan -> {model_dir}")
            continue
        try:
            row = evaluate_one_model(name, model_dir)
            rows.append(row)
        except Exception as e:
            print(f"[err] {name}: {e}")

    # Tampilkan ringkasan
    print("\nRingkasan:\n")
    header = ["model_name", "mAP", "mAP@0.5", "mAP@0.75", "AR@100", "TotalLoss", "TrainDuration(s)"]
    print("{:24} {:>8} {:>8} {:>8} {:>10} {:>12} {:>16}".format(*header))
    for r in rows:
        print("{:24} {:>8} {:>8} {:>8} {:>10} {:>12} {:>16}".format(
            r["model_name"],
            fmt(r["mAP"]),
            fmt(r["mAP@0.5"]),
            fmt(r["mAP@0.75"]),
            fmt(r["AR@100"]),
            fmt(r["TotalLoss"]),
            fmt(r["TrainDuration(s)"]),
        ))

    # Simpan CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                r["model_name"],
                r["mAP"],
                r["mAP@0.5"],
                r["mAP@0.75"],
                r["AR@100"],
                r["TotalLoss"],
                r["TrainDuration(s)"],
            ])

    print(f"\n[OK] Tersimpan: {OUT_CSV}")
    print(f"[Done] Elapsed: {time.time()-t0:0.2f}s")


if __name__ == "__main__":
    main()