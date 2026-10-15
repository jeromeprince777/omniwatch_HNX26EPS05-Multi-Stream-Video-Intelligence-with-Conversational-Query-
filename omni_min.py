#!/usr/bin/env python3
"""
omni_min.py - minimal, debuggable video-search backend for HNX26EPS05.

  python omni_min.py index --cam Gate  --video clips/gate.mp4  --start "2026-10-09 09:00:00"
  python omni_min.py index --cam Lobby --video clips/lobby.mp4 --start "2026-10-09 09:01:00"
  python omni_min.py search "person carrying a bag"
  python omni_min.py search "red car"
  python omni_min.py sheet          # random 30 observations -> out/sheet.jpg (sanity check)

Why this does not give random boxes:
  * box and frame come from the SAME ultralytics result (r.orig_img + r.boxes). No video re-seeking.
  * keyframe (box drawn) and crop are written to disk at detection time; the UI only reads those files.
  * one row per TRACK (its best view). Short junk tracks are dropped.
  * class / bag / colour are computed deterministically. CLIP only RE-RANKS the filtered candidates.
  * if nothing passes the filters it says so (or flags results as unverified) instead of faking a match.
"""
import argparse
import datetime as dt
import os
import random
import re
import sqlite3
import sys
from collections import Counter

import numpy as np

DB = os.environ.get("OMNI_DB", "omni.db")
DATA = os.environ.get("OMNI_DATA", "data")
OUT = "out"
YOLO_W = os.environ.get("OMNI_YOLO", "models/yolov8s.pt" if os.path.exists("models/yolov8s.pt") else "yolov8s.pt")
CLIP_NAME = os.environ.get("OMNI_CLIP", "ViT-L-14")
CLIP_TAG = os.environ.get("OMNI_CLIP_TAG", "laion2b_s32b_b82k")  # check open_clip.list_pretrained()
CONF = float(os.environ.get("OMNI_CONF", "0.45"))
MIN_TRACK = int(os.environ.get("OMNI_MIN_TRACK", "4"))  # min sampled detections per track
TARGET_FPS = 5

VEH = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
BAGS = {24: "backpack", 26: "handbag", 28: "suitcase"}
KEEP = [0] + list(VEH) + list(BAGS)

CLS_WORDS = {
    "person": ["person", "people", "man", "men", "woman", "women", "guy", "girl", "boy", "pedestrian"],
    "car": ["car", "cars", "sedan", "suv", "taxi"],
    "truck": ["truck", "lorry"],
    "bus": ["bus"],
    "motorcycle": ["motorcycle", "motorbike", "scooter"],
    "bicycle": ["bicycle", "bike", "cycle"],
}
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black", "gray", "grey"]
BAG_WORDS = ["bag", "bags", "backpack", "handbag", "suitcase", "luggage", "purse"]


# ----------------------------------------------------------------------------- helpers
def color_name(bgr):
    import cv2
    if bgr is None or bgr.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    chroma = (s > 70) & (v > 60)
    if chroma.mean() < 0.25:
        m = float(v.mean())
        return "black" if m < 70 else ("white" if m > 170 else "gray")
    hist = np.bincount(h[chroma].ravel(), minlength=180)
    bins = {
        "red": list(range(0, 8)) + list(range(170, 180)), "orange": list(range(8, 20)),
        "yellow": list(range(20, 35)), "green": list(range(35, 85)), "blue": list(range(85, 130)),
        "purple": list(range(130, 160)), "pink": list(range(160, 170)),
    }
    return max(bins, key=lambda k: int(hist[bins[k]].sum()))


def color_region(core, cls):
    h, w = core.shape[:2]
    if cls == "person":  # torso
        return core[int(h * .20):int(h * .55), int(w * .20):int(w * .80)]
    return core[int(h * .25):int(h * .85), int(w * .15):int(w * .85)]  # vehicle body


def bag_on_person(p, b):
    """True if >=50% of the bag box lies inside the (slightly expanded) person box."""
    pw, ph = p[2] - p[0], p[3] - p[1]
    px1, py1, px2, py2 = p[0] - .15 * pw, p[1] - .05 * ph, p[2] + .15 * pw, p[3] + .05 * ph
    ix = max(0.0, min(px2, b[2]) - max(px1, b[0]))
    iy = max(0.0, min(py2, b[3]) - max(py1, b[1]))
    return ix * iy / max(1e-6, (b[2] - b[0]) * (b[3] - b[1])) >= 0.5


def parse(q):
    toks = set(re.findall(r"[a-z]+", q.lower()))
    cls = {n for n, words in CLS_WORDS.items() if toks & set(words)}
    if toks & {"vehicle", "vehicles"}:
        cls |= {"car", "truck", "bus", "motorcycle"}
    color = next((c for c in COLORS if c in toks), None)
    color = "gray" if color == "grey" else color
    bag = bool(toks & set(BAG_WORDS))
    if bag and not cls:
        cls = {"person"}
    return cls, color, bag


def db():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS obs(
        id INTEGER PRIMARY KEY, cam TEXT, tid INT, cls TEXT, color TEXT, has_bag INT, bag_type TEXT,
        det_conf REAL, t_first REAL, t_last REAL, t_best REAL, frame_idx INT,
        x1 INT, y1 INT, x2 INT, y2 INT, fw INT, fh INT, keyframe TEXT, crop TEXT, emb BLOB)""")
    return c


# ----------------------------------------------------------------------------- CLIP
_clip = None


def clip():
    global _clip
    if _clip is None:
        import torch
        import open_clip
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            m, _, pre = open_clip.create_model_and_transforms(CLIP_NAME, pretrained=CLIP_TAG)
            tok = open_clip.get_tokenizer(CLIP_NAME)
        except Exception as e:
            # Fallback to local cached ViT-B-32 if ViT-L-14 is not downloaded or offline
            print(f"[warning] Could not load {CLIP_NAME} ({CLIP_TAG}): {e}")
            print(f"[warning] Falling back to cached ViT-B-32 (laion2b_s34b_b79k)...")
            m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
            tok = open_clip.get_tokenizer("ViT-B-32")
        m = m.to(dev).eval()
        if dev == "cuda":
            m = m.half()
        _clip = (m, pre, tok, dev)
    return _clip


def embed_images(paths):
    import torch
    from PIL import Image
    m, pre, _, dev = clip()
    dtype = next(m.parameters()).dtype
    out = []
    for i in range(0, len(paths), 32):
        x = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + 32]]).to(dev, dtype)
        with torch.no_grad():
            f = m.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


SURVEILLANCE_PROMPTS = [
    "a CCTV security camera view of {q}",
    "a surveillance camera photo of {q}",
    "a security camera video capture of {q}",
    "an overhead surveillance view of {q}",
    "a photo of {q}",
]


def embed_text(q):
    import torch
    m, _, tok, dev = clip()
    prompts = [tmpl.format(q=q) for tmpl in SURVEILLANCE_PROMPTS]
    with torch.no_grad():
        tokens = tok(prompts).to(dev)
        feats = m.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        # Prompt ensembling: mean vector across surveillance templates
        mean_feat = feats.mean(dim=0, keepdim=True)
        mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
    return mean_feat.float().cpu().numpy()[0]


# ----------------------------------------------------------------------------- index
def cmd_index(a):
    import cv2
    import torch
    from ultralytics import YOLO

    fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / TARGET_FPS))
    t0 = dt.datetime.strptime(a.start, "%Y-%m-%d %H:%M:%S").timestamp()
    kd, cd = f"{DATA}/keyframes/{a.cam}", f"{DATA}/crops/{a.cam}"
    os.makedirs(kd, exist_ok=True)
    os.makedirs(cd, exist_ok=True)

    model = YOLO(YOLO_W)  # fresh model per video -> fresh tracker state
    gen = model.track(source=a.video, stream=True, tracker="bytetrack.yaml", conf=CONF, classes=KEEP,
                      vid_stride=stride, device=0 if torch.cuda.is_available() else "cpu", verbose=False)
    T = {}
    for i, r in enumerate(gen):
        fi = i * stride
        b = r.boxes
        if b is None or len(b) == 0 or b.id is None:
            continue
        img = r.orig_img  # <-- the exact frame these boxes were detected on
        H, W = img.shape[:2]
        xy = b.xyxy.cpu().numpy()
        cl = b.cls.cpu().numpy().astype(int)
        cf = b.conf.cpu().numpy()
        ids = b.id.cpu().numpy().astype(int)
        bags = [(xy[k], BAGS[cl[k]]) for k in range(len(cl)) if cl[k] in BAGS]
        for k in range(len(cl)):
            c = int(cl[k])
            if c in BAGS:
                continue
            x1, y1, x2, y2 = [int(v) for v in xy[k]]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if (y2 - y1) < 48 or (x2 - x1) < 24:
                continue
            name = "person" if c == 0 else VEH[c]
            tid = int(ids[k])
            t = T.setdefault(tid, dict(first=fi, n=0, bag=Counter(), best=0.0, cls=name))
            t["last"] = fi
            t["n"] += 1
            if name == "person":
                for bx, bn in bags:
                    if bag_on_person(xy[k], bx):
                        t["bag"][bn] += 1
            score = float(cf[k]) * ((x2 - x1) * (y2 - y1)) ** 0.5
            if score > t["best"] * 1.15:  # new best view of this track -> save it NOW
                kf, cp = f"{kd}/{tid}.jpg", f"{cd}/{tid}.jpg"
                ann = img.copy()
                cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.imwrite(kf, ann)
                pw, ph = int((x2 - x1) * .25), int((y2 - y1) * .25)
                cv2.imwrite(cp, img[max(0, y1 - ph):min(H, y2 + ph), max(0, x1 - pw):min(W, x2 + pw)])
                t.update(best=score, fi=fi, box=(x1, y1, x2, y2), conf=float(cf[k]), kf=kf, cp=cp,
                         color=color_name(color_region(img[y1:y2, x1:x2], name)), wh=(W, H))

    keep = {tid: t for tid, t in T.items() if t["n"] >= MIN_TRACK and "kf" in t}
    print(f"[index] {a.cam}: {len(T)} raw tracks -> {len(keep)} kept (>= {MIN_TRACK} sampled detections)")
    if not keep:
        return
    embs = embed_images([t["cp"] for t in keep.values()])
    con = db()
    con.execute("DELETE FROM obs WHERE cam=?", (a.cam,))
    sql = "INSERT INTO obs(cam,tid,cls,color,has_bag,bag_type,det_conf,t_first,t_last,t_best,frame_idx," \
          "x1,y1,x2,y2,fw,fh,keyframe,crop,emb) VALUES (" + ",".join("?" * 20) + ")"
    for (tid, t), e in zip(keep.items(), embs):
        has_bag = int(sum(t["bag"].values()) >= 2)
        bt = t["bag"].most_common(1)[0][0] if has_bag else None
        con.execute(sql, (a.cam, tid, t["cls"], t["color"], has_bag, bt, t["conf"],
                          t0 + t["first"] / fps, t0 + t["last"] / fps, t0 + t["fi"] / fps, t["fi"],
                          *t["box"], *t["wh"], t["kf"], t["cp"], e.astype(np.float32).tobytes()))
    con.commit()
    print("[index] done")


# ----------------------------------------------------------------------------- search / sheet
def make_sheet(rows, path, cols=3, w=420):
    import cv2
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tiles = []
    for r in rows:
        im = cv2.imread(r["keyframe"])
        if im is None:
            continue
        im = cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])))
        cap = np.zeros((44, w, 3), np.uint8)
        l1 = f'{r["cam"]} #{r["tid"]} {r["color"]} {r["cls"]}' + (f' +{r["bag_type"]}' if r["has_bag"] else "")
        l2 = f'DET {r["det_conf"]:.0%}' + (f' MATCH {r["match"]:.2f}' if "match" in r else "") \
            + (" [UNVERIFIED]" if r.get("unverified") else "")
        cv2.putText(cap, l1, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.putText(cap, l2, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
        tiles.append(np.vstack([im, cap]))
    if not tiles:
        return False
    H = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
    return True


def fmt_t(t):
    return dt.datetime.fromtimestamp(t).strftime("%H:%M:%S")


def search(q, k=6):
    cls, color, bag = parse(q)
    qe = embed_text(q)
    
    # Check if index has rows
    con = db()
    all_cnt = con.execute("SELECT count(*) FROM obs").fetchone()[0]
    if all_cnt == 0:
        return [], "index is empty - run `index` first"

    query_sql = "SELECT * FROM obs"
    params = []
    if cls:
        placeholders = ",".join("?" for _ in cls)
        query_sql += f" WHERE cls IN ({placeholders})"
        params = list(cls)

    rows = [dict(r) for r in con.execute(query_sql, params)]
    if not rows:
        return [], f"parsed: class={sorted(cls) or 'any'} | NO MATCH"

    # Fast vectorized cosine similarity with BLAS
    embs = np.vstack([np.frombuffer(r["emb"], dtype=np.float32) for r in rows])
    sims = embs @ qe
    for r, s in zip(rows, sims):
        r["match"] = float(s)

    strict = [r for r in rows if (not color or r["color"] == color) and (not bag or r["has_bag"])]
    sid = {r["id"] for r in strict}
    loose = [r for r in rows if r["id"] not in sid]
    strict.sort(key=lambda r: -r["match"])
    loose.sort(key=lambda r: -r["match"])
    res = strict[:k]
    note = f"parsed: class={sorted(cls) or 'any'} color={color} bag={bag} | {len(strict)} verified candidates"
    if len(res) < 3:  # be honest: fill with clearly-flagged unverified results
        for r in loose[:3 - len(res)]:
            r["unverified"] = True
            res.append(r)
        if not res:
            note += " | NO MATCH"
    return res, note


# ----------------------------------------------------------------------------- clip rendering
def ensure_clip(cam, tid, pad_sec=2.0, force=False):
    import cv2
    import shutil
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_bin = "ffmpeg"

    clip_dir = os.path.join(DATA, "clips").replace("\\", "/")
    os.makedirs(clip_dir, exist_ok=True)
    out_mp4 = f"{clip_dir}/{cam}_{tid}.mp4"

    # Cache hit: returns cached unless force=True
    if not force and os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0:
        return out_mp4

    con = db()
    row = con.execute("SELECT * FROM obs WHERE cam=? AND tid=?", (cam, int(tid))).fetchone()
    if not row:
        return None

    # Resolve video source path
    vpath = None
    try:
        vrow = con.execute("SELECT video FROM cam_meta WHERE cam=?", (cam,)).fetchone()
        if vrow and os.path.exists(vrow[0]):
            vpath = vrow[0]
    except Exception:
        pass

    if not vpath:
        cam_map = {
            "Yokohama": "data/public_cctv/yokohama_cctv.mp4",
            "Tours": "data/public_cctv/surveillance_tours.mp4",
            "cam1": "data/campus/cam1.mp4",
            "cam2": "data/campus/cam2.mp4",
        }
        vpath = cam_map.get(cam)

    if not vpath or not os.path.exists(vpath):
        try:
            import sqlite3
            c2 = sqlite3.connect("data/omniwatch.db")
            r2 = c2.execute("SELECT video_path FROM cameras WHERE camera_id=?", (cam,)).fetchone()
            if r2 and os.path.exists(r2[0]):
                vpath = r2[0]
        except Exception:
            pass

    if not vpath or not os.path.exists(vpath):
        print(f"[ensure_clip] Video path for camera {cam} not found")
        return None

    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fi = int(row["frame_idx"])
    pad_frames = int(pad_sec * fps)
    start_f = max(0, fi - pad_frames)
    end_f = min(total_frames - 1, fi + pad_frames + int(fps * 1.5))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    cur_f = start_f

    raw_frames = []
    while cur_f <= end_f:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
        cur_f += 1
    cap.release()

    if not raw_frames:
        return None

    # Run YOLO tracker across the extracted clip frames so the box tracks in real-time on EVERY frame
    from ultralytics import YOLO
    import torch

    dev = 0 if torch.cuda.is_available() else "cpu"
    half = torch.cuda.is_available()
    tracker_model = YOLO(YOLO_W)

    k_idx = max(0, min(len(raw_frames) - 1, fi - start_f))
    ref_box = [int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])]
    cls_name = row["cls"]

    trail = []
    active_box = ref_box
    target_tid = None

    try:
        gen = tracker_model.track(
            source=raw_frames,
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.18,
            classes=KEEP,
            device=dev,
            half=half,
            verbose=False,
        )

        tracked_boxes = []
        for idx, r in enumerate(gen):
            b = r.boxes
            found_box = None
            if b is not None and len(b) and b.id is not None:
                xy = b.xyxy.cpu().numpy()
                ids = b.id.cpu().numpy().astype(int)

                # Lock onto target tracklet with highest IoU to ref_box around keyframe
                if target_tid is None:
                    if idx >= k_idx or idx == len(raw_frames) - 1:
                        best_iou = -1.0
                        best_id = None
                        for j in range(len(xy)):
                            x1, y1, x2, y2 = xy[j]
                            ix1 = max(ref_box[0], x1)
                            iy1 = max(ref_box[1], y1)
                            ix2 = min(ref_box[2], x2)
                            iy2 = min(ref_box[3], y2)
                            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                            union = ((ref_box[2] - ref_box[0]) * (ref_box[3] - ref_box[1]) +
                                     (x2 - x1) * (y2 - y1) - inter)
                            iou = inter / max(1e-5, union)
                            if iou > best_iou:
                                best_iou = iou
                                best_id = ids[j]
                        if best_iou > 0.12:
                            target_tid = best_id

                if target_tid is not None:
                    for j in range(len(ids)):
                        if ids[j] == target_tid:
                            found_box = [int(v) for v in xy[j]]
                            break

                # Spatial continuity fallback: match closest candidate box
                if found_box is None and active_box is not None:
                    best_d = 1e9
                    for j in range(len(xy)):
                        cx = (xy[j][0] + xy[j][2]) / 2
                        cy = (xy[j][1] + xy[j][3]) / 2
                        acx = (active_box[0] + active_box[2]) / 2
                        acy = (active_box[1] + active_box[3]) / 2
                        dist = (cx - acx) ** 2 + (cy - acy) ** 2
                        if dist < (0.20 * max(width, height)) ** 2 and dist < best_d:
                            best_d = dist
                            found_box = [int(v) for v in xy[j]]
                            if target_tid is None:
                                target_tid = ids[j]

            if found_box is not None:
                active_box = found_box
            tracked_boxes.append(active_box)
    except Exception as exc:
        print(f"[ensure_clip] Tracker error, using keyframe reference: {exc}")
        tracked_boxes = [ref_box] * len(raw_frames)

    temp_mp4 = f"{clip_dir}/temp_{cam}_{tid}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_mp4, fourcc, fps, (width, height))

    col = (0, 255, 180)  # tactical cyan/green
    for idx, (frame, box) in enumerate(zip(raw_frames, tracked_boxes)):
        if box is not None:
            x1, y1, x2, y2 = box
            trail.append(((x1 + x2) // 2, y2))
            if len(trail) > 40:
                trail.pop(0)

            # Motion trail
            if len(trail) > 1:
                pts = np.array(trail, np.int32)
                cv2.polylines(frame, [pts], False, (0, 230, 80), 2)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)

            # Tactical corner reticles
            clen = min(18, (x2 - x1) // 4, (y2 - y1) // 4)
            for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                dx = clen if cx == x1 else -clen
                dy = clen if cy == y1 else -clen
                cv2.line(frame, (cx, cy), (cx + dx, cy), (0, 255, 255), 4)
                cv2.line(frame, (cx, cy), (cx, cy + dy), (0, 255, 255), 4)

            # Bounding box label
            label = f"#{tid} {cls_name.upper()}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            lx, ly = x1, max(0, y1 - th - 10)
            cv2.rectangle(frame, (lx, ly), (lx + tw + 10, ly + th + 10), col, -1)
            cv2.putText(frame, label, (lx + 5, ly + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Top HUD frame telemetry
        f_hud = f"EVIDENCE CLIP | CAM: {cam} | FRAME {start_f + idx}/{total_frames} | REAL-TIME TRACKING"
        cv2.rectangle(frame, (0, 0), (min(width, 12 + 10 * len(f_hud)), 26), (0, 0, 0), -1)
        cv2.putText(frame, f_hud, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 230), 1)

        writer.write(frame)

    writer.release()

    # Transcode to faststart H.264 MP4 for HTML5 web video playback
    try:
        cmd = [
            ffmpeg_bin, "-y",
            "-i", temp_mp4,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-movflags", "+faststart",
            "-an",
            out_mp4
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        if os.path.exists(temp_mp4):
            os.remove(temp_mp4)
    except Exception as exc:
        print(f"[ensure_clip] ffmpeg transcode error: {exc}")
        if os.path.exists(temp_mp4):
            shutil.move(temp_mp4, out_mp4)

    return out_mp4


def cmd_search(a):
    res, note = search(a.query)
    print(note)
    for i, r in enumerate(res, 1):
        print(f'{i}. {r["cam"]} #{r["tid"]} {r["color"]} {r["cls"]} bag={bool(r["has_bag"])} '
              f'DET {r["det_conf"]:.0%} MATCH {r["match"]:.2f} {fmt_t(r["t_first"])}-{fmt_t(r["t_last"])} '
              f'{"[UNVERIFIED]" if r.get("unverified") else ""}\n   {r["keyframe"]}')
    if res and make_sheet(res, f"{OUT}/search.jpg"):
        print(f"contact sheet -> {OUT}/search.jpg  (OPEN IT and check the boxes)")


def cmd_sheet(a):
    rows = [dict(r) for r in db().execute("SELECT * FROM obs")]
    rows = random.sample(rows, min(30, len(rows)))
    if make_sheet(rows, f"{OUT}/sheet.jpg", cols=5, w=300):
        print(f"{len(rows)} random observations -> {OUT}/sheet.jpg  (boxes must sit on real objects)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("index")
    p.add_argument("--cam", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--start", default="2026-10-09 09:00:00")
    p.set_defaults(fn=cmd_index)
    p = sp.add_parser("search")
    p.add_argument("query")
    p.set_defaults(fn=cmd_search)
    p = sp.add_parser("sheet")
    p.set_defaults(fn=cmd_sheet)
    p = sp.add_parser("clip")
    p.add_argument("--cam", required=True)
    p.add_argument("--tid", type=int, required=True)
    p.set_defaults(fn=lambda a: print(ensure_clip(a.cam, a.tid)))
    args = ap.parse_args()
    args.fn(args)
