# -- coding: utf-8 --
"""
Webcam test untuk model TF Object Detection (SavedModel, TF2).
- Baca webcam (default index 0)
- Tampilkan bbox + label + skor
- Tombol:
    [S]  simpan screenshot ke folder output
    [Q]  keluar
    [B]  toggle bbox on/off
    [+/-] naik/turun threshold skor
    [R]  reload model (kalau kamu ganti file di path yang sama)
"""
import os
import time
import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import tensorflow as tf

# === util: labelmap (pakai TFOD utils kalau ada, kalau tidak, parser kecil) ===
def load_labelmap(pbtxt_path: str) -> Dict[int, str]:
    """
    Parser sederhana untuk labelmap.pbtxt:
      item { id: 1 name: 'Nangka Mentah' }
      item { id: 2 name: "Nangka Matang" }
    """
    id2name = {}
    if not os.path.exists(pbtxt_path):
        print(f"[WARN] labelmap tidak ditemukan: {pbtxt_path} -> pakai nama default id_X")
        return id2name

    cur_id = None
    cur_name = None
    with open(pbtxt_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if s.startswith("id:"):
                try:
                    cur_id = int(s.split("id:")[1].strip())
                except:
                    cur_id = None
            elif s.startswith("name:"):
                val = s.split("name:")[1].strip()
                val = val.strip('"').strip("'")
                cur_name = val
            if cur_id is not None and cur_name is not None:
                id2name[cur_id] = cur_name
                cur_id, cur_name = None, None
    return id2name

def draw_overlay(frame, fps, thr, show_box):
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f} | THR: {thr:.2f} | [S]=save  [B]=bbox  [+/-]=thr  [Q]=quit"
    cv2.rectangle(frame, (0, 0), (w, 28), (0,0,0), -1)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    if not show_box:
        cv2.putText(frame, "BBOX OFF", (w-110, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved_model_dir", required=True,
                    help=r'Folder berisi saved_model.pb, contoh: E:\NEW REVISI AFTER SIDANG\rgb_runs_model\V1_FPN_640_BASELINE\saved_model')
    ap.add_argument("--labelmap", default=r"E:\NEW REVISI AFTER SIDANG\tfrecords_bydata\labelmap.pbtxt",
                    help="Path labelmap.pbtxt (boleh kosongi, tapi label jadi id_X).")
    ap.add_argument("--cam_index", type=int, default=0, help="Index webcam (default 0).")
    ap.add_argument("--width", type=int, default=1280, help="Lebar capture (kalau didukung webcam).")
    ap.add_argument("--height", type=int, default=720, help="Tinggi capture (kalau didukung webcam).")
    ap.add_argument("--threshold", type=float, default=0.30, help="Score threshold awal.")
    ap.add_argument("--out_dir", default=r"E:\NEW REVISI AFTER SIDANG\webcam_captures", help="Folder simpan screenshot.")
    args = ap.parse_args()

    saved_model_dir = Path(args.saved_model_dir)
    if not (saved_model_dir / "saved_model.pb").exists():
        print(f"[ERR] saved_model.pb tidak ditemukan di: {saved_model_dir}")
        return

    id2name = load_labelmap(args.labelmap)

    print("[INFO] Load model...")
    detect_fn = tf.saved_model.load(str(saved_model_dir))
    print("[OK] Model loaded.")

    cap = cv2.VideoCapture(args.cam_index, cv2.CAP_DSHOW)
    if args.width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    if args.height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("[ERR] Webcam tidak bisa dibuka.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_thr = float(args.threshold)
    show_box = True
    last_t = time.time()
    fps = 0.0
    snap_idx = 1

    # warmup 1x
    ret, frm = cap.read()
    if ret:
        _ = detect_fn(tf.convert_to_tensor(np.expand_dims(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB), 0), dtype=tf.uint8))

    win_name = "Webcam Nangka Detector (TFOD)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t0 = time.time()
            out = detect_fn(tf.convert_to_tensor(np.expand_dims(rgb, 0), dtype=tf.uint8))
            t1 = time.time()

            # hitung fps sederhana (moving)
            dt = t1 - last_t
            last_t = t1
            cur_fps = 1.0/dt if dt > 0 else fps
            fps = (fps*0.9 + cur_fps*0.1) if fps > 0 else cur_fps

            # ambil hasil
            num = int(out.get("num_detections", [0])[0]) if isinstance(out.get("num_detections", None), tf.Tensor) else None
            scores = out["detection_scores"][0].numpy()
            classes = out["detection_classes"][0].numpy().astype(np.int32)
            boxes  = out["detection_boxes"][0].numpy()

            H, W = frame.shape[:2]
            drawn = frame.copy()

            if show_box:
                N = len(scores) if num is None else min(num, len(scores))
                for i in range(N):
                    sc = float(scores[i])
                    if sc < score_thr:
                        continue
                    cls_id = int(classes[i])
                    name = id2name.get(cls_id, f"id_{cls_id}")
                    # convert y1,x1,y2,x2 (normalized) -> pixel
                    y1, x1, y2, x2 = boxes[i]
                    x1p, y1p = int(x1*W), int(y1*H)
                    x2p, y2p = int(x2*W), int(y2*H)

                    # warna: mentah=biru, matang=kuning-oranye
                    color = (255, 200, 0) if "matang" in name.lower() else (255, 120, 0) if "matang" in name.lower() else (255, 120, 0)
                    # biar konsisten: gunakan hijau untuk 'Mentah', oranye untuk 'Matang'
                    if "mentah" in name.lower():
                        color = (0, 200, 0)   # BGR hijau
                    elif "matang" in name.lower():
                        color = (0, 170, 255) # BGR oranye

                    cv2.rectangle(drawn, (x1p, y1p), (x2p, y2p), color, 2)
                    label = f"{name} {sc:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(drawn, (x1p, max(0, y1p- th - 8)), (x1p+tw+6, y1p), color, -1)
                    cv2.putText(drawn, label, (x1p+3, y1p-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)

            draw_overlay(drawn, fps, score_thr, show_box)
            cv2.imshow(win_name, drawn)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"snap_{ts}_{snap_idx:03d}.jpg"
                cv2.imwrite(str(out_path), drawn)
                print(f"[SAVE] {out_path}")
                snap_idx += 1
            elif key == ord('b'):
                show_box = not show_box
            elif key == ord('+') or key == ord('='):
                score_thr = min(0.99, score_thr+0.05)
            elif key == ord('-') or key == ord('_'):
                score_thr = max(0.05, score_thr-0.05)
            elif key == ord('r'):
                # reload model di path yang sama (kalau kamu overwrite file)
                print("[INFO] Reload model...")
                detect_fn = tf.saved_model.load(str(saved_model_dir))
                print("[OK] Reloaded.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()