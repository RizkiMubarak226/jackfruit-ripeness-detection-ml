# build_rgb_bydata_and_tfrecord.py
# Pipeline BY-DATA (RGB) + rewrite VOC XML + CSV + (opsional) TFRecord
# Folder sumber: images_original/{Train,Validation,Test} (gambar & XML boleh campur)
# Output dataset: dataset_rgb_bydata/{Train,Validation,Test}/{images,annotations}
# TFRecord (opsional): tfrecords_bydata/{train.record, validation.record}

import os, io, csv, json, glob, shutil, argparse
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import tensorflow as tf  # untuk TFRecord

# ---------- KONFIG DEFAULT (boleh langsung RUN tanpa argumen) ----------
DEF_IMAGES_ROOT = r"E:/NEW REVISI AFTER SIDANG/images_original"
DEF_OUT_ROOT    = r"E:/NEW REVISI AFTER SIDANG/dataset_rgb_bydata"
DEF_LABELMAP    = r"E:/NEW REVISI AFTER SIDANG/annotations/labelmap.pbtxt"
DEF_TFRECORDS   = r"E:/NEW REVISI AFTER SIDANG/tfrecords_bydata"  # kosongkan "" kalau tidak mau bikin TFRecord
CLASS_MATANG    = "Nangka Matang"
CLASS_MENTAH    = "Nangka Mentah"
# ----------------------------------------------------------------------

# ---------- UTIL ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images_and_xml(mixed_dir: Path) -> List[Tuple[Path, Path]]:
    """Cari pasangan (img, xml) di folder campur. Return list tuple; xml bisa None untuk Test."""
    exts = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}
    items = []
    # indekskan xml by stem
    xml_map = {Path(x).stem: Path(x) for x in glob.glob(str(mixed_dir / "*.xml"))}
    for p in mixed_dir.iterdir():
        if p.suffix in exts:
            xml = xml_map.get(p.stem, None)
            items.append((p, xml))
    return items

def read_voc_objects(xml_path: Path):
    root = ET.parse(str(xml_path)).getroot()
    size = root.find("size")
    W = int(size.findtext("width"))
    H = int(size.findtext("height"))
    bboxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name").strip()
        bb   = obj.find("bndbox")
        xmin = int(float(bb.findtext("xmin"))); ymin = int(float(bb.findtext("ymin")))
        xmax = int(float(bb.findtext("xmax"))); ymax = int(float(bb.findtext("ymax")))
        bboxes.append((name, xmin, ymin, xmax, ymax, obj))
    return root, W, H, bboxes

def pil_crop_safe(img: Image.Image, box):
    x1,y1,x2,y2 = box
    x1 = max(0, min(x1, img.width-1))
    x2 = max(0, min(x2, img.width-1))
    y1 = max(0, min(y1, img.height-1))
    y2 = max(0, min(y2, img.height-1))
    if x2 <= x1 or y2 <= y1:
        return None
    return img.crop((x1,y1,x2,y2))

def rgb_mean(pil_img: Image.Image):
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    R = float(arr[...,0].mean()); G = float(arr[...,1].mean()); B = float(arr[...,2].mean())
    return R,G,B

def otsu_threshold(values: np.ndarray) -> float:
    """Otsu 1D untuk values float: kita buat histogram 256 bin pada range min..max."""
    if values.size == 0:
        return 0.0
    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin < 1e-6:
        return vmin
    hist, bin_edges = np.histogram(values, bins=256, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t*omega - mu)**2 / (omega*(1.0-omega) + 1e-12)
    idx = int(np.nanargmax(sigma_b2))
    # threshold = tepi bin berikutnya
    thr = bin_edges[idx+1]
    return thr

def load_labelmap(pbtxt_path: str) -> Dict[str,int]:
    mp = {}
    name=None; idx=None
    with open(pbtxt_path, "r", encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if s.startswith("id:"):
                try: idx=int(s.split("id:")[1].strip())
                except: pass
            elif s.startswith("name:"):
                val=s.split("name:")[1].strip().strip("'\"")
                name=val
            if name is not None and idx is not None:
                mp[name]=idx; name=None; idx=None
    # pastikan dua kelas ada
    for need in (CLASS_MATANG, CLASS_MENTAH):
        if need not in mp:
            raise ValueError(f"Labelmap harus memuat kelas '{need}'. File: {pbtxt_path}")
    return mp

# ---------- LANGKAH 1: BELAJAR AMBANG dari Train ----------
def learn_threshold(train_dir: Path) -> dict:
    print(f"[*] Belajar threshold dari: {train_dir}")
    pairs = list_images_and_xml(train_dir)
    vals_rg = []
    n_sample = 0
    for img_path, xml_path in pairs:
        if xml_path is None: 
            continue
        try:
            img = Image.open(str(img_path)).convert("RGB")
            _, _, _, bboxes = read_voc_objects(xml_path)
            if len(bboxes)==0: 
                continue
            for _, x1,y1,x2,y2,_ in bboxes:
                crop = pil_crop_safe(img, (x1,y1,x2,y2))
                if crop is None: 
                    continue
                R,G,B = rgb_mean(crop)
                vals_rg.append(R-G)
                n_sample += 1
        except Exception:
            continue
    vals = np.array(vals_rg, dtype=np.float32)
    thr = otsu_threshold(vals) if vals.size>0 else 0.0
    stat = {
        "feature": "RminusG",
        "n_samples": int(n_sample),
        "cluster_low_mean": float(vals[vals<=thr].mean()) if vals.size else None,
        "cluster_high_mean": float(vals[vals>thr].mean()) if vals.size else None,
        "threshold": float(thr)
    }
    print(f"[+] Threshold selesai: {stat}")
    return stat

# ---------- LANGKAH 2: REWRITE XML + COPY DATA + CSV ----------
def rewrite_split(mixed_dir: Path, out_split_dir: Path, thr: float) -> dict:
    img_out = out_split_dir / "images"
    ann_out = out_split_dir / "annotations"
    ensure_dir(img_out); ensure_dir(ann_out)
    csv_rows = []
    pairs = list_images_and_xml(mixed_dir)
    kept, skipped = 0, 0
    for img_path, xml_path in pairs:
        if xml_path is None:
            # untuk Test tanpa GT: hanya salin gambar (tanpa XML)
            shutil.copy2(str(img_path), str(img_out / img_path.name))
            continue
        try:
            img = Image.open(str(img_path)).convert("RGB")
            root, W,H, bboxes = read_voc_objects(xml_path)
            if len(bboxes)==0:
                skipped += 1
                continue
            for i,(name,x1,y1,x2,y2,obj_node) in enumerate(bboxes):
                crop = pil_crop_safe(img, (x1,y1,x2,y2))
                if crop is None: 
                    continue
                R,G,B = rgb_mean(crop)
                f = R-G
                pred = CLASS_MATANG if f >= thr else CLASS_MENTAH
                # ganti label
                obj_node.find("name").text = pred
                csv_rows.append([mixed_dir.name, img_path.name, i, R, G, B, f, pred])
            # simpan xml baru
            out_xml = ann_out / xml_path.name
            ET.ElementTree(root).write(str(out_xml), encoding="utf-8")
            # salin gambar
            shutil.copy2(str(img_path), str(img_out / img_path.name))
            kept += 1
        except Exception:
            skipped += 1
    # CSV
    with open(out_split_dir / "rgb_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); 
        w.writerow(["split","filename","bbox_id","meanR","meanG","meanB","RminusG","pred_label"])
        w.writerows(csv_rows)
    return {"kept_xml": kept, "skipped": skipped, "csv_rows": len(csv_rows)}

# ---------- LANGKAH 3 (opsional): TFRecord ----------
def _bytes_feature(v: bytes): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
def _float_list_feature(v: List[float]): return tf.train.Feature(float_list=tf.train.FloatList(value=v))
def _int64_list_feature(v: List[int]): return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def write_tfrecord_for_split(split_dir: Path, out_record: Path, label2id: Dict[str,int]) -> dict:
    imgs = split_dir / "images"; anns = split_dir / "annotations"
    xmls = sorted(glob.glob(str(anns / "*.xml")))
    ensure_dir(out_record.parent)
    w = tf.io.TFRecordWriter(str(out_record))
    total = 0; kept = 0
    for x in xmls:
        total += 1
        try:
            root, W,H, bboxes = read_voc_objects(Path(x))
            img_name = root.findtext("filename") or (Path(x).with_suffix(".jpg").name)
            img_path = imgs / img_name
            if not img_path.exists():
                # coba variasi ekstensi
                stem = Path(img_name).stem
                found = False
                for ext in [".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"]:
                    cand = imgs / f"{stem}{ext}"
                    if cand.exists(): img_path=cand; found=True; break
                if not found: 
                    continue
            with open(img_path, "rb") as f:
                enc = f.read()
            pil = Image.open(io.BytesIO(enc)).convert("RGB")
            W,H = pil.size
            xmins,xmaxs,ymins,ymaxs,cls_text,cls_id = [],[],[],[],[],[]
            for name,x1,y1,x2,y2,_ in bboxes:
                if name not in label2id: 
                    continue
                x1 = np.clip(x1, 0, W-1); x2 = np.clip(x2, 0, W-1)
                y1 = np.clip(y1, 0, H-1); y2 = np.clip(y2, 0, H-1)
                if x2<=x1 or y2<=y1: 
                    continue
                xmins.append(x1/W); xmaxs.append(x2/W)
                ymins.append(y1/H); ymaxs.append(y2/H)
                cls_text.append(name.encode("utf-8"))
                cls_id.append(label2id[name])
            if len(cls_id)==0: 
                continue
            feat = {
                "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[H])),
                "image/width":  tf.train.Feature(int64_list=tf.train.Int64List(value=[W])),
                "image/filename": _bytes_feature(img_path.name.encode("utf-8")),
                "image/source_id": _bytes_feature(img_path.name.encode("utf-8")),
                "image/encoded": _bytes_feature(enc),
                "image/format": _bytes_feature(img_path.suffix.replace(".","").encode("utf-8")),
                "image/object/bbox/xmin": _float_list_feature(xmins),
                "image/object/bbox/xmax": _float_list_feature(xmaxs),
                "image/object/bbox/ymin": _float_list_feature(ymins),
                "image/object/bbox/ymax": _float_list_feature(ymaxs),
                "image/object/class/text": tf.train.Feature(bytes_list=tf.train.BytesList(value=cls_text)),
                "image/object/class/label": _int64_list_feature(cls_id),
            }
            ex = tf.train.Example(features=tf.train.Features(feature=feat))
            w.write(ex.SerializeToString())
            kept += 1
        except Exception:
            continue
    w.close()
    return {"xml_total": total, "examples_written": kept, "record": str(out_record)}

# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser(description="RGB-by-data (Otsu) + XML rewrite + CSV + TFRecord")
    ap.add_argument("--images_root",  default=DEF_IMAGES_ROOT, help="Root sumber, mis: E:/.../images_original")
    ap.add_argument("--out_root",     default=DEF_OUT_ROOT,    help="Root output dataset by-data")
    ap.add_argument("--labelmap",     default=DEF_LABELMAP,    help="labelmap.pbtxt (wajib berisi dua nama kelas)")
    ap.add_argument("--class_matang", default=CLASS_MATANG)
    ap.add_argument("--class_mentah", default=CLASS_MENTAH)
    ap.add_argument("--fixed_thr",    type=float, default=None, help="Jika diisi, pakai threshold ini (skip Otsu)")
    ap.add_argument("--make_tfrecords_to", default=DEF_TFRECORDS, help="Folder TFRecord output (kosongkan untuk skip)")
    args = ap.parse_args()

    images_root = Path(args.images_root)
    out_root    = Path(args.out_root)
    ensure_dir(out_root)

    # validasi split
    splits = []
    for s in ["Train","Validation","Test"]:
        p = images_root / s
        if p.exists():
            splits.append((s, p))
    if not splits:
        raise FileNotFoundError(f"Tidak ditemukan folder split di {images_root} (butuh Train/Validation/Test).")

    # pelajari threshold (kecuali fixed)
    if args.fixed_thr is not None:
        thr_stat = {"feature":"RminusG", "n_samples":0, "threshold": float(args.fixed_thr)}
    else:
        train_dir = images_root / "Train"
        if not train_dir.exists():
            raise FileNotFoundError("Split 'Train' wajib ada untuk belajar threshold Otsu.")
        thr_stat = learn_threshold(train_dir)

    # tulis summary threshold
    with open(out_root / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(thr_stat, f, indent=2)

    # proses tiap split
    all_csv = []
    for name, p in splits:
        print(f"[*] Proses split: {name}")
        out_split = out_root / name
        stats = rewrite_split(p, out_split, thr_stat["threshold"])
        print(f"[+] {name} selesai: {stats}")
        # gabungkan CSV
        csv_path = out_split / "rgb_stats.csv"
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                rdr = csv.reader(f); rows = list(rdr)
                if rows: all_csv.extend(rows[1:])  # skip header

    # simpan CSV gabungan
    if all_csv:
        with open(out_root / "rgb_stats_all.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split","filename","bbox_id","meanR","meanG","meanB","RminusG","pred_label"])
            w.writerows(all_csv)

    # TFRecord (opsional untuk Train & Validation)
    if args.make_tfrecords_to:
        tf_out = Path(args.make_tfrecords_to); ensure_dir(tf_out)
        label2id = load_labelmap(args.labelmap)
        # train
        train_dir = out_root / "Train"
        if train_dir.exists():
            info_tr = write_tfrecord_for_split(train_dir, tf_out / "train.record", label2id)
            print(f"[TFRecord] Train: {info_tr}")
        # validation
        val_dir = out_root / "Validation"
        if val_dir.exists():
            info_va = write_tfrecord_for_split(val_dir, tf_out / "validation.record", label2id)
            print(f"[TFRecord] Validation: {info_va}")
        print(f"[OK] TFRecord disimpan di: {tf_out}")

    print("\n[SELESAI] Semua tahap beres ✔")

if __name__ == "__main__":
    main()
