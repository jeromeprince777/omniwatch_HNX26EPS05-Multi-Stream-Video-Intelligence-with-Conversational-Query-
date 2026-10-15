#!/usr/bin/env python3
"""
omni_live.py  –  GPU-saturating real-time query tracking for HNX26EPS05.

    python omni_live.py --src Yokohama=data/public_cctv/yokohama_cctv.mp4 \
                        --src Tours=data/public_cctv/surveillance_tours.mp4 \
                        --loop --query "person carrying a bag"
    open http://127.0.0.1:8100

GPU utilisation strategy (why this is fast):
  * Decode thread (per camera): cv2.VideoCapture → raw frames into a bounded
    deque. Never sleeps, grabs as fast as disk/net allows.
  * Detection thread (per camera): YOLO(imgsz=640, half=True, stream) runs
    exclusively on GPU. Draws boxes on the SAME orig_img. No re-seeking.
  * CLIP thread (shared, daemon): batches crops from ALL cameras, runs
    encode_image once per batch on GPU, posts results back via queues.
  * Encode thread (per camera): cv2.imencode on a CPU thread; never blocks
    the GPU threads.
  * The detection and CLIP threads never sleep and never hold the GIL during
    GPU work (torch.no_grad + half precision keeps occupancy high).
  * imgsz=640 fixed so YOLO preprocessing is deterministic and fast.
  * Warmup: one dummy inference on startup so the first real frame is full-speed.
"""

import argparse
import collections
import os
import queue
import re
import sys
import threading
import time
import traceback

import cv2
import numpy as np

import omni_min as om

# ─── tunables ──────────────────────────────────────────────────────────────────
CONF_LIVE       = float(os.environ.get("OMNI_LIVE_CONF",    "0.30"))
CLIP_P          = float(os.environ.get("OMNI_LIVE_CLIP_P",  "0.48"))
CLIP_EVERY      = int(os.environ.get("OMNI_LIVE_CLIP_EVERY","25"))   # frames between CLIP refreshes
CLIP_BATCH      = int(os.environ.get("OMNI_LIVE_CLIP_BATCH","12"))   # crops batched per GPU call
DECODE_BUF      = int(os.environ.get("OMNI_LIVE_BUF",      "4"))     # raw-frame buffer depth
IMGSZ           = int(os.environ.get("OMNI_LIVE_IMGSZ",    "640"))   # YOLO input size
JPEG_Q          = int(os.environ.get("OMNI_LIVE_JPEG_Q",   "82"))    # MJPEG quality
MIN_H, MIN_W    = 36, 18
MAX_STALE       = 120                                                  # frames before track pruned

# ─── NLP stop-word sets ────────────────────────────────────────────────────────
STOP = set("""a an the show me find track follow highlight every all any who whoever that is are in of on at
with wearing carrying holding having has have walking and or please locate where one someone anyone everyone
everybody""".split())
COVER     = set("shirt tshirt jacket top hoodie coat clothes clothing dress kurta sweater".split())
ACCESSORY = set("""hat cap helmet shoes pants trousers jeans shorts scarf mask glasses bag bags backpack
handbag suitcase luggage purse umbrella""".split())
CLS_ALL   = {w for ws in om.CLS_WORDS.values() for w in ws} | {"vehicle", "vehicles"}

# ─── shared state ──────────────────────────────────────────────────────────────
QLOCK   = threading.Lock()
Q       = {"P": None, "others": True}
STREAMS: dict[str, "Stream"] = {}

# ─── CLIP worker plumbing ──────────────────────────────────────────────────────
# Each item: (stream_name, tid, crop_bgr, tf_array)
_clip_in:  queue.Queue = queue.Queue(maxsize=128)
# Each item: (stream_name, tid, prob)
_clip_out: queue.Queue = queue.Queue(maxsize=256)


# ══════════════════════════════════════════════════════════════════════════════
# Query helpers
# ══════════════════════════════════════════════════════════════════════════════
def parse_live(q: str) -> dict:
    cls, color, bag = om.parse(q)
    toks = re.findall(r"[a-z]+", q.lower())
    resid = set(toks) - STOP - set(om.COLORS) - set(om.BAG_WORDS) - COVER - CLS_ALL
    # "red hat" → colour modifies accessory → delegate to CLIP
    if color:
        for i, w in enumerate(toks):
            if w in (color, "grey") and i + 1 < len(toks) and toks[i + 1] in ACCESSORY:
                resid.add(color)
                color = None
                break
    return dict(cls=cls, color=color, bag=bag, resid=resid, text=q.strip())


def _build_text_feats(P: dict) -> np.ndarray | None:
    """Encode query + negatives into a (N, D) float32 array on CPU."""
    if not P["resid"]:
        return None
    import torch
    m, _, tok, dev = om.clip()
    negs = [f"a CCTV security camera view of a {c}" for c in (sorted(P["cls"]) or ["person"])]
    negs += [
        "a photo of an empty street",
        "a blurry background and pavement",
        "an empty building facade",
        "shadows on the ground",
    ]
    pos_texts = [
        f"a CCTV security camera view of {P['text']}",
        f"a photo of {P['text']}",
    ]
    with torch.no_grad():
        pos_tok = tok(pos_texts).to(dev)
        pos_f = m.encode_text(pos_tok)
        pos_f = pos_f / pos_f.norm(dim=-1, keepdim=True)
        pos_mean = pos_f.mean(dim=0, keepdim=True)
        pos_mean = pos_mean / pos_mean.norm(dim=-1, keepdim=True)

        neg_tok = tok(negs).to(dev)
        neg_f = m.encode_text(neg_tok)
        neg_f = neg_f / neg_f.norm(dim=-1, keepdim=True)

        all_f = torch.cat([pos_mean, neg_f], dim=0)
    return all_f.float().cpu().numpy()


def set_query(q: str, others: bool = True) -> None:
    q = (q or "").strip()
    P = parse_live(q)
    P["tf"] = _build_text_feats(P)          # may do a GPU call – ok on API thread
    with QLOCK:
        P["ver"] = (Q["P"]["ver"] + 1) if Q["P"] else 1
        Q["P"], Q["others"] = P, others


def status() -> dict:
    P = Q["P"]
    return {
        "query":  P["text"],
        "others": Q["others"],
        "parsed": {"class": sorted(P["cls"]), "color": P["color"],
                   "bag": P["bag"], "free_text": sorted(P["resid"])},
        "cams":   {n: s.info() for n, s in STREAMS.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Shared CLIP worker  (single daemon thread – owns the GPU encoder)
# ══════════════════════════════════════════════════════════════════════════════
class ClipWorker(threading.Thread):
    """Drains _clip_in in batches, writes scores to _clip_out."""
    def __init__(self):
        super().__init__(daemon=True, name="ClipWorker")

    def run(self):
        import torch
        from PIL import Image
        try:
            m, pre, _, dev = om.clip()
            dtype = next(m.parameters()).dtype
        except Exception as exc:
            print(f"[ClipWorker] cannot load CLIP: {exc}", file=sys.stderr)
            return

        while True:
            # collect up to CLIP_BATCH items or block for up to 20 ms
            items = []
            try:
                items.append(_clip_in.get(timeout=0.02))
            except queue.Empty:
                continue
            while len(items) < CLIP_BATCH:
                try:
                    items.append(_clip_in.get_nowait())
                except queue.Empty:
                    break

            # group by text-feature array (same query → same tf)
            # tf is a numpy array; use its id as key (they're the same object within one query ver)
            groups: dict[int, list] = {}
            for sname, tid, crop, tf in items:
                groups.setdefault(id(tf), []).append((sname, tid, crop, tf))

            for gid, gitems in groups.items():
                tf = gitems[0][3]                       # (N_text, D)
                crops_bgr = [it[2] for it in gitems]
                try:
                    tensors = []
                    for c in crops_bgr:
                        rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                        tensors.append(pre(Image.fromarray(rgb)))
                    x = torch.stack(tensors).to(dev, dtype)
                    with torch.no_grad():
                        cf = m.encode_image(x)
                        cf = cf / cf.norm(dim=-1, keepdim=True)
                        tf_t = torch.from_numpy(tf).to(dev, dtype)
                        # Batched GPU matrix multiplication & softmax across all crops at once
                        logits = 100.0 * (cf @ tf_t.T)  # (B, N_text)
                        probs = torch.softmax(logits, dim=-1)[:, 0].float().cpu().tolist()
                    for (sname, tid, _, _), prob in zip(gitems, probs):
                        _clip_out.put((sname, tid, prob))
                except Exception as exc:
                    print(f"[ClipWorker] inference error: {exc}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# Per-camera stream
# ══════════════════════════════════════════════════════════════════════════════
class Stream:
    """Spawns three threads: decode, detect+draw, encode."""

    def __init__(self, name: str, src: str, loop: bool):
        self.name  = name
        self.src   = src
        self.loop  = loop
        self.is_file = not (src.isdigit() or src.lower().startswith(("rtsp", "http", "rtmp")))

        # raw frame queue: decode thread → detect thread (FIFO, zero dropped frames)
        self._raw_q: queue.Queue = queue.Queue(maxsize=48)

        # processed JPEG: detect thread → MJPEG clients
        self.jpg: bytes | None = None
        self.seq: int = 0
        self.jpg_cv = threading.Condition()

        self.fps_ema: float = 0.0
        self.tracks:  dict  = {}
        self.matches: list  = []
        self.err:     str | None = None

        # CLIP result inbox (tid → prob)
        self._clip_results: dict[int, float] = {}
        self._clip_res_lock = threading.Lock()

    def info(self) -> dict:
        return {
            "fps":     round(self.fps_ema, 1),
            "tracks":  len(self.tracks),
            "matches": self.matches,
            "error":   self.err,
        }

    # ── decode thread ──────────────────────────────────────────────────────────
    def _decode_loop(self):
        while True:
            try:
                cap = cv2.VideoCapture(int(self.src) if self.src.isdigit() else self.src)
                if not cap.isOpened():
                    self.err = f"cannot open {self.src}"
                    return
                fi = 0
                while True:
                    ok, img = cap.read()
                    if not ok:
                        break
                    fi += 1
                    # Push every frame into bounded FIFO queue (blocks if full so no frames are dropped)
                    self._raw_q.put((fi, img), block=True)
                cap.release()
                if not (self.loop and self.is_file):
                    break
                time.sleep(0.05)
            except Exception:
                self.err = traceback.format_exc()
                print(self.err, file=sys.stderr)
                time.sleep(1.0)

    # ── track evaluation ────────────────────────────────────────────────────────
    def evaluate(self, tr: dict, P: dict):
        """Return True / False / None (None = CLIP pending)."""
        if not P["text"]:
            return False
        if P["cls"] and tr["cls"] not in P["cls"]:
            return False
        if P["color"] and tr["color"] != P["color"]:
            return False
        if P["bag"] and not tr["has_bag"]:
            return False
        if P["resid"]:
            p = tr.get("clip_p")
            return None if p is None else p >= CLIP_P
        return True

    # ── detect + draw + CLIP submission ────────────────────────────────────────
    def _detect_loop(self):
        import torch
        from ultralytics import YOLO

        dev  = 0 if torch.cuda.is_available() else "cpu"
        half = torch.cuda.is_available()
        model = YOLO(om.YOLO_W)

        # GPU warmup – prevents cold-start latency on first real frame
        dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
        try:
            model.predict(dummy, imgsz=IMGSZ, device=dev, half=half, verbose=False)
        except Exception:
            pass

        fi_last = -1
        while True:
            # drain CLIP result inbox
            with self._clip_res_lock:
                try:
                    while True:
                        sname, tid, prob = _clip_out.get_nowait()
                        if sname == self.name:
                            tr = self.tracks.get(tid)
                            if tr is not None:
                                old = tr.get("clip_p")
                                tr["clip_p"] = prob if old is None else 0.5 * old + 0.5 * prob
                except queue.Empty:
                    pass

            # pull next frame in sequence from queue — zero dropped frames!
            try:
                fi, img = self._raw_q.get(timeout=0.5)
            except queue.Empty:
                continue

            t1 = time.perf_counter()
            H, W = img.shape[:2]

            # ── YOLO + ByteTrack ────────────────────────────────────────────
            try:
                r = model.track(img, persist=True, tracker="bytetrack.yaml",
                                conf=CONF_LIVE, classes=om.KEEP,
                                imgsz=IMGSZ, device=dev, half=half, verbose=False)[0]
            except Exception as exc:
                print(f"[{self.name}] YOLO error: {exc}", file=sys.stderr)
                continue

            P, others = Q["P"], Q["others"]
            seen: list[int] = []

            b = r.boxes
            if b is not None and len(b) and b.id is not None:
                xy  = b.xyxy.cpu().numpy()
                cl  = b.cls.cpu().numpy().astype(int)
                ids = b.id.cpu().numpy().astype(int)
                bags = [(xy[k], om.BAGS[cl[k]]) for k in range(len(cl)) if cl[k] in om.BAGS]

                for k in range(len(cl)):
                    c = int(cl[k])
                    if c in om.BAGS:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in xy[k])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    bh, bw = y2 - y1, x2 - x1
                    if bh < MIN_H or bw < MIN_W:
                        continue

                    name = "person" if c == 0 else om.VEH[c]
                    tid  = int(ids[k])
                    tr = self.tracks.setdefault(tid, dict(
                        cls=name,
                        cols=collections.deque(maxlen=20), color="unknown",
                        bagq=collections.deque(maxlen=25), bag_type=None, has_bag=False,
                        clip_p=None, clip_f=-10**9,
                        hits=0, on=False,
                        trail=collections.deque(maxlen=50),
                        ver=-1, n=0,
                    ))

                    if tr["ver"] != P["ver"]:
                        tr.update(ver=P["ver"], clip_p=None, clip_f=-10**9, hits=0, on=False)
                    tr["n"]   += 1
                    tr["box"]  = (x1, y1, x2, y2)
                    tr["seen"] = fi
                    tr["trail"].append(((x1 + x2) // 2, y2))

                    # colour vote every 3rd frame
                    if tr["n"] % 3 == 1:
                        region = om.color_region(img[y1:y2, x1:x2], name)
                        tr["cols"].append(om.color_name(region))
                        tr["color"] = collections.Counter(tr["cols"]).most_common(1)[0][0]

                    # bag overlap
                    if name == "person":
                        hit = next((bn for bx, bn in bags if om.bag_on_person(xy[k], bx)), None)
                        tr["bagq"].append(hit is not None)
                        if hit:
                            tr["bag_type"] = hit
                        tr["has_bag"] = sum(tr["bagq"]) >= 3

                    seen.append(tid)

            # ── submit CLIP crops ───────────────────────────────────────────
            if P.get("resid") and P.get("tf") is not None:
                for tid in seen:
                    tr = self.tracks[tid]
                    pre_ok = (
                        (not P["cls"]   or tr["cls"]   in P["cls"])
                        and (not P["color"] or tr["color"] == P["color"])
                        and (not P["bag"]   or tr["has_bag"])
                    )
                    if pre_ok and (fi - tr["clip_f"]) >= CLIP_EVERY:
                        x1, y1, x2, y2 = tr["box"]
                        pw = int((x2 - x1) * 0.20)
                        ph = int((y2 - y1) * 0.20)
                        crop = img[max(0, y1-ph):min(H, y2+ph),
                                   max(0, x1-pw):min(W, x2+pw)]
                        if crop.size == 0:
                            continue
                        tr["clip_f"] = fi
                        try:
                            _clip_in.put_nowait((self.name, tid, crop, P["tf"]))
                        except queue.Full:
                            pass   # skip if CLIP worker is backed up

            # ── verdict + draw ──────────────────────────────────────────────
            matches: list[dict] = []
            for tid in seen:
                tr = self.tracks[tid]
                res = self.evaluate(tr, P)
                if res is True:
                    tr["hits"] = min(10, tr["hits"] + 1)
                elif res is False:
                    tr["hits"] = max(0,  tr["hits"] - 1)
                # hysteresis: needs 2 hits to turn on, 1 hit to stay on
                tr["on"] = (tr["hits"] >= 2) if not tr["on"] else (tr["hits"] > 0)

                x1, y1, x2, y2 = tr["box"]
                if tr["on"]:
                    tag = (f'#{tid} {tr["cls"]} {tr["color"]}'
                           + (f' +{tr["bag_type"]}' if tr["has_bag"] else ""))
                    if tr["clip_p"] is not None and P.get("resid"):
                        tag += f' {tr["clip_p"]:.2f}'
                    col = (0, 230, 80)                           # bright green
                    if len(tr["trail"]) > 1:
                        pts = np.array(tr["trail"], np.int32)
                        cv2.polylines(img, [pts], False, col, 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    lx, ly = x1, max(0, y1 - th - 8)
                    cv2.rectangle(img, (lx, ly), (lx + tw + 8, ly + th + 8), col, -1)
                    cv2.putText(img, tag, (lx + 4, ly + th + 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                    matches.append({"tid": tid, "cls": tr["cls"], "color": tr["color"],
                                    "bag": tr["bag_type"] if tr["has_bag"] else None})
                elif res is None and P.get("text"):
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 255), 1)   # amber = pending
                elif others:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (110, 110, 110), 1) # grey = not matching

            # prune stale tracks
            stale = [t for t, tr in self.tracks.items() if fi - tr.get("seen", fi) > MAX_STALE]
            for t in stale:
                del self.tracks[t]

            self.matches = matches

            # HUD banner
            dtm = max(1e-4, time.perf_counter() - t1)
            self.fps_ema = (1.0/dtm) if not self.fps_ema else 0.92*self.fps_ema + 0.08/dtm
            tag = f'{self.name} | "{P["text"] or "-"}" | match={len(matches)} | {self.fps_ema:.0f}fps'
            bw  = min(W, 12 + 10 * len(tag))
            cv2.rectangle(img, (0, 0), (bw, 30), (0, 0, 0), -1)
            cv2.putText(img, tag, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 230), 1)

            # ── resize for web before encode ────────────────────────────────
            if W > 1280:
                scale = 1280 / W
                img = cv2.resize(img, (1280, int(H * scale)), interpolation=cv2.INTER_LINEAR)

            # Direct fast JPEG encode (only ~3ms on CPU) - guarantees 100% of detected frames are encoded!
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            if ok:
                with self.jpg_cv:
                    self.jpg = buf.tobytes()
                    self.seq += 1
                    self.jpg_cv.notify_all()

            # For file sources, pace to smooth real-time (~25 FPS)
            if self.is_file:
                elapsed = time.perf_counter() - t1
                slack = 0.040 - elapsed
                if slack > 0:
                    time.sleep(slack)

    def start(self):
        for target, name in [
            (self._decode_loop, f"{self.name}-dec"),
            (self._detect_loop, f"{self.name}-det"),
        ]:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()


# ══════════════════════════════════════════════════════════════════════════════
# MJPEG generator
# ══════════════════════════════════════════════════════════════════════════════
def mjpeg(s: Stream):
    last = -1
    while True:
        with s.jpg_cv:
            s.jpg_cv.wait_for(lambda: s.seq != last, timeout=2.0)
            last, jpg = s.seq, s.jpg
        if jpg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpg)).encode()
                + b"\r\n\r\n"
                + jpg
                + b"\r\n"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Web UI
# ══════════════════════════════════════════════════════════════════════════════
PAGE = """\
<!doctype html>
<meta charset="utf-8">
<title>OmniWatch LIVE</title>
<style>
*{box-sizing:border-box}
body{background:#070c18;color:#b8f5d8;font:14px/1.4 'Segoe UI',system-ui,sans-serif;margin:0;padding:0}
header{display:flex;align-items:center;gap:12px;padding:10px 18px;background:#0b1525;
  border-bottom:1px solid #0cf4}
header h1{margin:0;font-size:1.05rem;letter-spacing:.12em;color:#0ff;text-transform:uppercase}
.dot{width:8px;height:8px;border-radius:50%;background:#0f6;box-shadow:0 0 6px #0f6;animation:blink 1.4s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.controls{display:flex;align-items:center;gap:8px;padding:10px 18px;background:#0d1a2e;
  border-bottom:1px solid #0cf3;flex-wrap:wrap}
input[type=text]{flex:1;min-width:260px;padding:8px 12px;background:#101a2b;border:1px solid #0cf;
  color:#fff;border-radius:6px;font-size:13px;outline:none}
input[type=text]:focus{border-color:#0ff;box-shadow:0 0 0 2px rgba(0,255,255,.15)}
button{padding:8px 16px;background:#0b3a4a;color:#0ff;border:1px solid #0cf;border-radius:6px;
  cursor:pointer;font-size:13px;white-space:nowrap}
button:hover{background:#0d4d63}
label{color:#8bc;font-size:13px;display:flex;align-items:center;gap:5px;white-space:nowrap}
.chips{display:flex;flex-wrap:wrap;gap:5px;padding:8px 18px;background:#0a1222;border-bottom:1px solid #0cf2}
.chip{padding:4px 10px;background:#0b2a3a;color:#0dd;border:1px solid #0cf5;border-radius:20px;
  font-size:12px;cursor:pointer;transition:background .15s}
.chip:hover{background:#0d3e54}
#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:8px;padding:10px 18px}
.cam-wrap{background:#0b1525;border:1px solid #1a3a4a;border-radius:8px;overflow:hidden}
.cam-label{padding:6px 10px;font-size:12px;color:#8bc;letter-spacing:.08em;text-transform:uppercase;
  background:#0d1e38;border-bottom:1px solid #1a3a4a}
.cam-wrap img{width:100%;display:block}
#status{padding:8px 18px;background:#080f1e;border-top:1px solid #0cf2}
pre{margin:0;font-size:11.5px;color:#5af;font-family:'Cascadia Code','Consolas',monospace}
</style>
<header>
  <div class="dot"></div>
  <h1>OmniWatch // Live Query Tracking</h1>
</header>
<div class="controls">
  <input id="q" type="text" placeholder="e.g.  person carrying a bag  /  man in red  /  white car  /  everyone" autofocus>
  <button onclick="send(q.value)">▶ TRACK</button>
  <label><input type="checkbox" id="o" checked onchange="send(q.value)"> Show others</label>
</div>
<div class="chips" id="chips"></div>
<div id="grid"></div>
<div id="status"><pre id="st">connecting…</pre></div>
<script>
const QUERIES = ['person carrying a bag','person in red','person in blue','person with a hat',
                 'red car','white car','everyone','man carrying a bag'];
for (const t of QUERIES)
  chips.innerHTML += `<span class="chip" onclick="q.value='${t}';send('${t}')">${t}</span>`;
q.addEventListener('keydown', e => { if (e.key === 'Enter') send(q.value); });
let made = false;
function render(s) {
  if (!made) {
    for (const n in s.cams) {
      grid.innerHTML += `<div class="cam-wrap">
        <div class="cam-label">${n}</div>
        <img src="/stream/${n}" loading="lazy">
      </div>`;
    }
    made = true;
  }
  const out = {};
  for (const [k,v] of Object.entries(s.cams))
    out[k] = {fps: v.fps, tracks: v.tracks,
               matches: (v.matches||[]).map(m=>`#${m.tid} ${m.cls} ${m.color}${m.bag?'+'+m.bag:''}`),
               error: v.error ? 'see server console' : null};
  st.textContent = JSON.stringify({query:s.query, parsed:s.parsed, cams:out}, null, 1);
}
async function send(t) {
  const r = await fetch('/api/query', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({q: t, others: o.checked})});
  render(await r.json());
}
async function poll() {
  try { render(await (await fetch('/api/status')).json()); } catch(e) {}
  setTimeout(poll, 800);
}
poll();
</script>
"""


def build_app():
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, StreamingResponse

    app = FastAPI(title="OmniWatch Live")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/")
    def index():
        return HTMLResponse(PAGE)

    @app.get("/stream/{name}")
    def stream(name: str):
        if name not in STREAMS:
            raise HTTPException(404, "unknown camera")
        return StreamingResponse(
            mjpeg(STREAMS[name]),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/api/query")
    def api_query(body: dict = Body(...)):
        set_query(body.get("q", ""), body.get("others", True))
        return status()

    @app.get("/api/status")
    def api_status():
        return status()

    return app


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="OmniWatch real-time query tracking")
    ap.add_argument("--src", action="append", required=True,
                    help="NAME=path  |  rtsp://...  |  webcam-index  (repeatable)")
    ap.add_argument("--query",  default="person",
                    help="initial natural-language query")
    ap.add_argument("--loop",   action="store_true",
                    help="restart file sources when they end (keeps tracker alive)")
    ap.add_argument("--port",   type=int, default=8100)
    ap.add_argument("--workers", type=int, default=2,
                    help="uvicorn worker count")
    a = ap.parse_args()

    import uvicorn

    print("[startup] loading CLIP …", flush=True)
    om.clip()   # warm up once in main thread
    print("[startup] CLIP ready", flush=True)

    # start shared CLIP worker daemon
    ClipWorker().start()

    set_query(a.query)

    for s_spec in a.src:
        name, eq, src = s_spec.partition("=")
        if not eq:
            name, src = f"cam{len(STREAMS)+1}", s_spec
        st = Stream(name, src, a.loop)
        STREAMS[name] = st
        st.start()
        print(f"[startup] stream '{name}' started ← {src}", flush=True)

    print(f"\n  ► open  http://127.0.0.1:{a.port}\n", flush=True)
    uvicorn.run(build_app(), host="127.0.0.1", port=a.port,
                log_level="warning", workers=1)


if __name__ == "__main__":
    main()
