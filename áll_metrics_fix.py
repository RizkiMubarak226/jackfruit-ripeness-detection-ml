# savemodel_eval_all_fixed.py
# -- coding: utf-8 --

"""
Evaluasi banyak model SSD MobileNet (TFOD) dengan GT dari anotasi VOC XML.
- Robust ke variasi struktur folder saved_model (pencarian rekursif).
- Output: CSV selalu, XLSX jika openpyxl tersedia.

Metrik:
  Accuracy total
  Precision / Recall / F1 (Matang, Mentah)
  Latency per image (s), FPS
  RAM & VRAM (MB) terpakai saat evaluasi
  Ukuran model (MB)
  Jumlah sampel evaluasi
"""

from __future__ import annotations

import os, sys, time, json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
import psutil

# GPU opsional
try:
    import GPUtil
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

# ======================= KONFIGURASI =======================

# 1) Root folder yang berisi subfolder model (seperti di screenshot kamu)
ROOT_MODELS = Path(r"E:\NEW REVISI AFTER SIDANG\rgb_runs_model")

# 2) Daftar nama subfolder model. Tulis persis seperti di Explorer
MODELS: List[str] = [
    "V1_FPN_320_BASELINE",
    "V1_FPN_320_OPT",
    "V1_FPN_640_BASELINE",
    "V1_FPN_640_OPT",
    "V2_FPNLite_320_BASELINE",
    "V2_FPNLite_320_OPT",
    "V2_FPNLite_640_BASELINE",
    "V2_FPNLite_640_OPT",
]

# 3) Dataset uji (VOC): folder gambar & anotasi XML
TEST_IMAGES_DIR = Path(r"E:\NEW REVISI AFTER SIDANG\dataset_rgb_bydata\Test\images")
TEST_ANN_DIR    = Path(r"E:\NEW REVISI AFTER SIDANG\dataset_rgb_bydata\Test\annotations")  # XML VOC

# 4) (Opsional) label map pbtxt untuk memetakan id->name
LABELMAP_PATH = Path(r"E:\NEW REVISI AFTER SIDANG\annotations\labelmap.pbtxt")  # boleh tidak ada

# Ambang skor deteksi
SCORE_THRESHOLD: float = 0.20

# Nama kelas yang dipakai dalam laporan (urutan ini juga dipakai saat metrik)
CLASSES = ["matang", "mentah"]

# Output hasil
RESULTS_DIR = Path(r"E:\NEW REVISI AFTER SIDANG\results")
CSV_PATH  = RESULTS_DIR / "eval_detection_metrics.csv"
XLSX_PATH = RESULTS_DIR / "eval_detection_metrics.xlsx"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Fallback mapping id->name bila labelmap tidak tersedia
FALLBACK_ID2NAME: Dict[int, str] = {
    1: "nangka_mentah",
    2: "nangka_matang",
}

# ==========================================================

def log(s: str) -> None:
    print(s, flush=True)

# ---------- util: label canonical ----------

def canonize(name: str) -> Optional[str]:
    """
    Ubah berbagai variasi string nama kelas ke 'matang' / 'mentah'.
    Return None jika tidak bisa dikenali.
    """
    if not name:
        return None
    n = name.strip().lower()
    n = n.replace("_", " ").replace("-", " ")
    # contoh varian yang sering muncul:
    candidates = {
        "matang": ["matang", "rip", "ripe", "nangka matang", "nangka_matang"],
        "mentah": ["mentah", "raw", "unripe", "nangka mentah", "nangka_mentah"],
    }
    for k, arr in candidates.items():
        for a in arr:
            if a in n:
                return k
    # fallback: jika string punya kata 'matang' / 'mentah'
    if "matang" in n: return "matang"
    if "mentah" in n: return "mentah"
    return None

# ---------- util: baca VOC ----------

import xml.etree.ElementTree as ET

def load_voc_annotation(xml_path: Path) -> Optional[str]:
    """
    Ambil kelas GT dari 1 file VOC XML (ambil objek pertama).
    Return 'matang'/'mentah' bila dikenali; None bila gagal.
    """
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        obj = root.find("object")
        if obj is None:
            return None
        name = obj.findtext("name", default="").strip()
        return canonize(name)
    except Exception:
        return None

def build_test_pairs(images_dir: Path, ann_dir: Path) -> List[Tuple[Path, str]]:
    """
    Buat daftar (path_gambar, gt_label) memakai pasangan nama file (stem sama).
    Hanya ambil sample yang punya anotasi valid (matang/mentah).
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs: List[Tuple[Path, str]] = []
    if not images_dir.exists() or not ann_dir.exists():
        return pairs

    ann_map: Dict[str, Path] = {p.stem: p for p in ann_dir.glob("*.xml")}
    for img in images_dir.rglob("*"):
        if img.is_file() and img.suffix.lower() in exts:
            xml = ann_map.get(img.stem)
            if not xml:
                continue
            gt = load_voc_annotation(xml)
            if gt in CLASSES:
                pairs.append((img, gt))
    return pairs

# ---------- util: temukan saved_model ----------

def find_saved_model_dir(model_root: Path) -> Optional[Path]:
    """
    Temukan folder yang berisi 'saved_model.pb' (prioritas pada subfolder 'saved_model').
    Robust ke struktur tersarang (rglob).
    """
    cand = model_root / "saved_model"
    if (cand / "saved_model.pb").exists():
        return cand
    for p in model_root.rglob("saved_model.pb"):
        return p.parent
    return None

def model_disk_size_mb(model_dir: Path) -> float:
    total = 0
    for dp, dn, files in os.walk(model_dir):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except Exception:
                pass
    return total / (1024 * 1024)

# ---------- util: labelmap ----------

def parse_labelmap_pbtxt(pb_path: Path) -> Dict[int, str]:
    """
    Parse minimal labelmap.pbtxt → {id:int: name:str}
    Format umum:
      item { id: 1 name: "nangka_mentah" }
    """
    if not pb_path.exists():
        return {}
    text = pb_path.read_text(encoding="utf-8", errors="ignore")
    id2name: Dict[int, str] = {}
    cur_id: Optional[int] = None
    cur_name: Optional[str] = None

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("id:"):
            try:
                cur_id = int(s.split(":")[1].strip())
            except Exception:
                cur_id = None
        elif s.startswith("name:"):
            name = s.split(":", 1)[1].strip().strip('"').strip("'")
            cur_name = name
        elif s == "}" or s == "item {":
            # commit bila lengkap
            if cur_id is not None and cur_name:
                id2name[cur_id] = cur_name
            cur_id, cur_name = None, None

    # commit tail
    if cur_id is not None and cur_name:
        id2name[cur_id] = cur_name
    return id2name

# ---------- prediksi satu gambar ----------

def predict_label(detect_fn, img_bgr: np.ndarray, id2name: Dict[int,str], th: float) -> str:
    """Ambil label final dari satu gambar (top score di atas threshold)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = tf.convert_to_tensor(np.expand_dims(img_rgb, 0), dtype=tf.uint8)
    out = detect_fn(inp)

    scores  = out["detection_scores"][0].numpy()
    classes = out["detection_classes"][0].numpy().astype(np.int32)

    # top index ≥ th
    idx = None
    for i, sc in enumerate(scores):
        if sc >= th:
            idx = i
            break
    if idx is None:
        return "tidak_terdeteksi"

    cid = int(classes[idx])
    raw_name = id2name.get(cid, f"id_{cid}")
    lab = canonize(raw_name)
    return lab if lab in CLASSES else "tidak_terdeteksi"

# ---------- evaluasi satu model ----------

def evaluate_one_model(model_root: Path,
                       test_pairs: List[Tuple[Path, str]],
                       id2name: Dict[int,str]) -> Optional[Dict]:
    sm = find_saved_model_dir(model_root)
    if sm is None:
        log(f"    (!) saved_model.pb tidak ditemukan di bawah: {model_root}")
        return None

    try:
        detect_fn = tf.saved_model.load(str(sm))
    except Exception as e:
        log(f"    [ERR] Gagal load saved_model: {e}")
        return None

    # ukuran model dihitung dari folder induk sm (biar termasuk variables, assets)
    model_mb = model_disk_size_mb(sm.parent)

    proc = psutil.Process(os.getpid())
    ram_before = proc.memory_info().rss / (1024 * 1024)
    vram_before = GPUtil.getGPUs()[0].memoryUsed if (_HAS_GPU and GPUtil.getGPUs()) else None

    y_true: List[str] = []
    y_pred: List[str] = []

    t0 = time.time()
    frames = 0

    for img_path, gt in test_pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        pred = predict_label(detect_fn, img, id2name, SCORE_THRESHOLD)

        # catat metrik hanya jika GT valid (matang/mentah)
        if gt in CLASSES:
            y_true.append(gt)
            y_pred.append(pred if pred in CLASSES else ("matang" if gt == "mentah" else "mentah"))
        frames += 1

    t1 = time.time()

    ram_after = proc.memory_info().rss / (1024 * 1024)
    vram_after = GPUtil.getGPUs()[0].memoryUsed if (_HAS_GPU and GPUtil.getGPUs()) else None

    if len(y_true) == 0:
        acc = 0.0
        pr = rc = f1 = [0.0, 0.0]
    else:
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASSES, zero_division=0
        )
        acc = accuracy_score(y_true, y_pred)

    latency = (t1 - t0) / max(frames, 1)
    fps = (1.0 / latency) if latency > 0 else 0.0
    ram_used = ram_after - ram_before
    vram_used = (vram_after - vram_before) if (vram_before is not None and vram_after is not None) else None

    return {
        "Accuracy": round(float(acc), 2),
        "Precision (Matang)": round(float(pr[CLASSES.index("matang")]), 2) if len(pr) == 2 else 0.0,
        "Recall (Matang)":    round(float(rc[CLASSES.index("matang")]), 2) if len(rc) == 2 else 0.0,
        "F1-Score (Matang)":  round(float(f1[CLASSES.index("matang")]), 2) if len(f1) == 2 else 0.0,
        "Precision (Mentah)": round(float(pr[CLASSES.index("mentah")]), 2) if len(pr) == 2 else 0.0,
        "Recall (Mentah)":    round(float(rc[CLASSES.index("mentah")]), 2) if len(rc) == 2 else 0.0,
        "F1-Score (Mentah)":  round(float(f1[CLASSES.index("mentah")]), 2) if len(f1) == 2 else 0.0,
        "Latency (s)":        round(float(latency), 4),
        "FPS":                round(float(fps), 2),
        "RAM (MB)":           round(float(ram_used), 2),
        "VRAM (MB)":          (round(float(vram_used), 2) if vram_used is not None else "N/A"),
        "Model Size (MB)":    round(float(model_mb), 2),
        "Samples":            int(frames),
    }

# ---------- main ----------

def main() -> None:
    log(f"[INFO] Root models : {ROOT_MODELS}")
    log(f"[INFO] Output dir  : {RESULTS_DIR}")

    # Bangun pasangan (img, gt) dari VOC
    test_pairs = build_test_pairs(TEST_IMAGES_DIR, TEST_ANN_DIR)
    log(f"[INFO] Jumlah sample test (punya anotasi): {len(test_pairs)}")
    if len(test_pairs) == 0:
        log(f"[ERR] Tidak menemukan pasangan gambar+XML di:\n  {TEST_IMAGES_DIR}\n  {TEST_ANN_DIR}")
        sys.exit(1)

    # Mapping id->name: dari labelmap jika ada; fallback jika tidak
    id2name = parse_labelmap_pbtxt(LABELMAP_PATH) if LABELMAP_PATH.exists() else {}
    if not id2name:
        id2name = FALLBACK_ID2NAME.copy()
        log("[WARN] labelmap.pbtxt tidak ditemukan/parsable. Pakai fallback ID2NAME.")
    else:
        log(f"[INFO] labelmap terdeteksi: {id2name}")

    import pandas as pd
    rows: List[Dict] = []

    for name in MODELS:
        model_root = ROOT_MODELS / name
        log(f"\n[+] Evaluasi: {name}")
        if not model_root.exists():
            log(f"    (!) Folder model tidak ada: {model_root}, lewati.")
            continue

        metrics = evaluate_one_model(model_root, test_pairs, id2name)
        if metrics is None:
            log("    Gagal evaluasi (lihat pesan di atas), lewati.")
            continue

        row = {"Model": name}
        row.update(metrics)
        rows.append(row)
        log("    [OK] Selesai.")

    if not rows:
        log("[ERR] Tidak ada hasil evaluasi yang tersimpan.")
        return

    # Susunan kolom
    cols = [
        "Model",
        "Accuracy",
        "Precision (Matang)", "Recall (Matang)", "F1-Score (Matang)",
        "Precision (Mentah)", "Recall (Mentah)", "F1-Score (Mentah)",
        "Latency (s)", "FPS", "RAM (MB)", "VRAM (MB)",
        "Model Size (MB)", "Samples"
    ]
    df = pd.DataFrame(rows)[cols]

    # Simpan CSV
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    log(f"\n[OK] CSV disimpan: {CSV_PATH}")

    # Simpan XLSX (opsional)
    try:
        import openpyxl  # noqa: F401
        df.to_excel(XLSX_PATH, index=False)
        log(f"[OK] Excel disimpan: {XLSX_PATH}")
    except Exception as e:
        log(f"[WARN] Gagal simpan Excel (install openpyxl): {e}")

    # Cetak ringkas ke terminal
    with pd.option_context("display.max_columns", None, "display.width", 160):
        log("\n[INFO] Preview hasil:")
        print(df)

if __name__ == "__main__":
    main()