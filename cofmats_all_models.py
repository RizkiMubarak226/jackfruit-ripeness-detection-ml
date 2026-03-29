# make_confmats_all_models.py
# -- coding: utf-8 --

import os, sys, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd

# =============== KONFIGURASI ===============
ROOT_MODELS = Path(r"E:\NEW REVISI AFTER SIDANG\rgb_runs_model")
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

# Dataset uji
TEST_IMAGES_DIR = Path(r"E:\NEW REVISI AFTER SIDANG\dataset_rgb_bydata\Test\images")
TEST_ANN_DIR    = Path(r"E:\NEW REVISI AFTER SIDANG\dataset_rgb_bydata\Test\annotations")  # VOC XML (opsional)

# Skor minimal deteksi yang dianggap valid
SCORE_THRESHOLD = 0.20

# Mapping ID kelas -> nama (ubah jika labelmap Anda berbeda)
# 1 = mentah, 2 = matang (sesuai setup Anda sebelumnya)
ID2NAME = {1: "nangka_mentah", 2: "nangka_matang"}

CLASSES = ["matang", "mentah"]  # urutan label final
RESULTS_DIR = Path(r"E:\NEW REVISI AFTER SIDANG\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================

def find_saved_model_dir(model_root: Path) -> Optional[Path]:
    """Cari folder yang berisi saved_model.pb."""
    cand = model_root / "saved_model"
    if (cand / "saved_model.pb").exists():
        return cand
    # cari rekursif max 3 level
    for depth in range(1, 4):
        pattern = ("/" + "*/"*depth).strip("/") + "saved_model.pb"
        for p in model_root.glob(pattern):
            return p.parent
    return None

def load_bgr(img_path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(img_path))
    return img

def parse_voc_xml_label(xml_path: Path) -> Optional[str]:
    """Ambil GT dari VOC XML: jika ada 'nangka_matang' → matang, kalau 'nangka_mentah' → mentah."""
    if not xml_path.exists():
        return None
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        names = [obj.findtext("name", default="").strip().lower() for obj in root.findall("object")]
        if any("matang" in n for n in names):
            return "matang"
        if any("mentah" in n for n in names):
            return "mentah"
        return None
    except Exception:
        return None

def list_test_pairs(images_dir: Path, ann_dir: Path) -> List[Tuple[Path,str]]:
    """
    Kembalikan [(path_img, gt_label)].
    Prioritas:
      1) Jika ada subfolder 'matang' / 'mentah' → pakai nama subfolder.
      2) Jika flat folder → baca GT dari VOC XML dengan nama file yang sama.
    """
    pairs: List[Tuple[Path,str]] = []
    if not images_dir.exists():
        return pairs

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    subdirs = [d for d in images_dir.iterdir() if d.is_dir()]

    if subdirs:  # pola Test/images/matang/.jpg, Test/images/mentah/.jpg
        for d in subdirs:
            low = d.name.lower()
            if "matang" in low:
                gt = "matang"
            elif "mentah" in low:
                gt = "mentah"
            else:
                continue
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    pairs.append((p, gt))
    else:  # flat folder: GT dari VOC xml
        for p in images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                xml = (ann_dir / (p.stem + ".xml"))
                gt = parse_voc_xml_label(xml)
                if gt in ("matang", "mentah"):
                    pairs.append((p, gt))
    return pairs

def id_to_final_name(cid: int) -> str:
    name = ID2NAME.get(int(cid), f"id_{cid}").lower()
    if "matang" in name:
        return "matang"
    if "mentah" in name:
        return "mentah"
    return "unknown"

def predict_one(detect_fn, img_bgr: np.ndarray) -> str:
    """Prediksi 1 gambar → 'matang'/'mentah'/'unknown' (bila tak ada deteksi ≥ threshold)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = tf.convert_to_tensor(np.expand_dims(img_rgb, 0), dtype=tf.uint8)
    out = detect_fn(tensor)
    scores  = out["detection_scores"][0].numpy()
    classes = out["detection_classes"][0].numpy().astype(np.int32)

    # ambil deteksi skor tertinggi yang ≥ threshold
    for sc, cid in zip(scores, classes):
        if sc >= SCORE_THRESHOLD:
            return id_to_final_name(int(cid))
    return "unknown"

def plot_and_save_confmat(cm: np.ndarray, labels: List[str], title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
    im = ax.imshow(cm, interpolation="kaiser")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color="black" if cm[i, j] > thresh else "white")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    print(f"[INFO] Cari sample test…")
    pairs = list_test_pairs(TEST_IMAGES_DIR, TEST_ANN_DIR)
    if not pairs:
        print(f"[ERR] Tidak menemukan sample test valid di: {TEST_IMAGES_DIR}")
        if not TEST_ANN_DIR.exists():
            print(f"      (Folder annotations juga tidak ditemukan: {TEST_ANN_DIR})")
        sys.exit(1)
    print(f"[OK] Jumlah sample test: {len(pairs)}")

    all_rows = []

    for name in MODELS:
        mroot = ROOT_MODELS / name
        smdir = find_saved_model_dir(mroot)
        print(f"\n[+] Evaluasi: {name}")
        if smdir is None:
            print(f"    (!) saved_model tidak ditemukan di {mroot} → lewati.")
            continue

        try:
            detect_fn = tf.saved_model.load(str(smdir))
        except Exception as e:
            print(f"    [ERR] Gagal load model: {e} → lewati.")
            continue

        y_true, y_pred = [], []
        for img_path, gt in pairs:
            img = load_bgr(img_path)
            if img is None:
                continue
            pred = predict_one(detect_fn, img)
            # jika unknown, anggap kelas salah (pilih lawannya)
            if pred not in CLASSES:
                pred = "mentah" if gt == "matang" else "matang"
            y_true.append(gt)
            y_pred.append(pred)

        # Confusion matrix berurutan sesuai CLASSES
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)

        # Simpan gambar
        out_png = RESULTS_DIR / f"confmat_{name}.png"
        plot_and_save_confmat(cm, CLASSES, f"Confusion Matrix - {name}", out_png)
        print(f"    [OK] Gambar disimpan: {out_png}")

        # Simpan ringkas ke list → nanti CSV gabungan
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        all_rows.append({"Model": name, "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)})

    # CSV ringkas semua model
    if all_rows:
        df = pd.DataFrame(all_rows, columns=["Model","TP","FP","FN","TN"])
        out_csv = RESULTS_DIR / "confusion_matrices_counts.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n[OK] Rekap counts disimpan: {out_csv}")
    else:
        print("\n[WARN] Tidak ada hasil yang tersimpan. Periksa path model/test.")

if __name__ == "__main__":
    main()