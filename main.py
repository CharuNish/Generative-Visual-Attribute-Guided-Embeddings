import os
import time
from typing import List, Dict, Any, Set

import cv2
import numpy as np
from PIL import Image

import torch
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from torchreid.utils import FeatureExtractor

from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util

from moondream_wrapper import MoondreamExtractor

# ----------- CONFIG -----------
YOLO_WEIGHTS = ""
MOONDREAM_WEIGHTS = ""
INPUT_PATH = ""
OUTPUT_PATH = ""

DETECT_INTERVAL = 4.0
LOITER_SECONDS = 5.0
TRACK_TTL = 75.0
DRAW_PADDING = 0.3
MAX_EMB_HISTORY = 5

USE_WORDNET = True

# similarity weights
FUZZY_RATIO_W = 0.3
EMB_RATIO_W = 0.7

ATTR_W = 0.4
REID_W = 0.5
HIST_W = 0.1
IOU_W = 0.0

ASSIGN_THRESHOLD = 0.5
SMOOTH_ALPHA = 0.45

# synonyms map
EXT_SYNS = {
    "backpack": ["bookbag", "knapsack", "rucksack"],
    "bag": ["handbag", "tote", "purse"],
    "hoodie": ["hooded_sweatshirt", "hoody"],
    "jacket": ["coat", "windbreaker"],
    "jeans": ["denim", "denims"],
    "sneakers": ["trainers", "running_shoes"],
    "boots": ["work_boots"],
    "hat": ["cap", "beanie"],
    "shirt": ["tshirt", "tee"]
}

# ----------------- attribute helpers -----------------
def _get_wordnet_syns(word: str) -> Set[str]:
    # lazy import to avoid unnecessary startup cost if unused
    import nltk
    from nltk.corpus import wordnet
    result = set()
    for s in wordnet.synsets(word):
        for lem in s.lemmas():
            result.add(lem.name().lower().replace("_", " "))
    return result

def expand_terms(attrs: List[str], synonyms: Dict[str, List[str]], use_wordnet: bool=False) -> Set[str]:
    out = set()
    for a in attrs:
        out.add(a)
        if a in synonyms:
            out.update(synonyms[a])
        if use_wordnet:
            out.update(_get_wordnet_syns(a))
    return out

def fuzzy_avg(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    total = 0.0
    cnt = 0
    for x in a:
        for y in b:
            total += fuzz.ratio(x, y)
            cnt += 1
    return (total / cnt) if cnt else 0.0

def embed_text_list(model: SentenceTransformer, attrs: Set[str]):
    if not attrs:
        return model.encode("", convert_to_tensor=True)
    text = " ".join(sorted(list(attrs)))
    return model.encode(text, convert_to_tensor=True)

def semantic_sim(a: Set[str], b: Set[str], sbert_model: SentenceTransformer) -> float:
    emb1 = embed_text_list(sbert_model, a)
    emb2 = embed_text_list(sbert_model, b)
    return util.cos_sim(emb1, emb2).item()

def compute_attr_similarity(a_attrs: List[str], b_attrs: List[str],
                            synonyms: Dict[str, List[str]], sbert_model: SentenceTransformer,
                            use_wordnet: bool=False) -> float:
    A = expand_terms(a_attrs, synonyms, use_wordnet)
    B = expand_terms(b_attrs, synonyms, use_wordnet)
    fuzzy_sc = fuzzy_avg(A, B) / 100.0
    sem_sc = semantic_sim(A, B, sbert_model)
    combined = FUZZY_RATIO_W * fuzzy_sc + EMB_RATIO_W * sem_sc
    return min(combined, 1.0)

# ----------------- visual helpers -----------------
def iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interA = interW * interH
    aA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    denom = aA + aB - interA
    return (interA / denom) if denom > 0 else 0.0

def color_histogram(bgr_patch):
    hist = cv2.calcHist([bgr_patch],[0,1,2],[None],[8,8,8],[0,256,0,256,0,256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def cos_sim_vec(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = float(np.dot(v1, v2))
    na = np.linalg.norm(v1); nb = np.linalg.norm(v2)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot/(na*nb)

def combined_score(track: Dict[str,Any], det: Dict[str,Any], synonyms, sbert_model, use_wordnet=False) -> float:
    attr_sc = compute_attr_similarity(track['attrs'], det['attrs'], synonyms, sbert_model, use_wordnet)
    iou_sc = iou(track['bbox'], det['bbox']) if IOU_W > 0 else 0.0
    hist_sc = 0.0
    if track.get('hist') is not None and det.get('hist') is not None:
        hist_sc = cv2.compareHist(track['hist'], det['hist'], cv2.HISTCMP_CORREL)
    reid_sc = 0.0
    if track.get('emb') is not None and det.get('emb') is not None:
        reid_sc = cos_sim_vec(track['emb'], det['emb'])
    return ATTR_W*attr_sc + REID_W*reid_sc + HIST_W*hist_sc + IOU_W*iou_sc

# ----------------- Track Database  -----------------
class TrackStore:
    def __init__(self, loiter_thresh=LOITER_SECONDS, ttl=TRACK_TTL, max_emb=MAX_EMB_HISTORY):
        self.data = {}
        self._next = 1
        self.loiter_thresh = loiter_thresh
        self.ttl = ttl
        self.max_emb = max_emb

    def step_assign(self, detections: List[Dict], ts: float, synonyms, sbert_model, use_wordnet=False):
        if not self.data:
            for det in detections:
                self._spawn(det, ts)
            return

        ids = list(self.data.keys())
        T = len(ids); D = len(detections)
        cost = np.ones((T, D), dtype=np.float32)

        for i, tid in enumerate(ids):
            for j, det in enumerate(detections):
                sim = combined_score(self.data[tid], det, synonyms, sbert_model, use_wordnet)
                cost[i,j] = 1.0 - sim

        rows, cols = linear_sum_assignment(cost)
        matched_tracks = set(); matched_dets = set()
        for r,c in zip(rows, cols):
            sim = 1.0 - cost[r,c]
            if sim >= ASSIGN_THRESHOLD:
                tid = ids[r]
                self._update(tid, detections[c], ts)
                matched_tracks.add(tid)
                matched_dets.add(c)

        # new detections
        for idx, det in enumerate(detections):
            if idx not in matched_dets:
                self._spawn(det, ts)

        # mark unmatched
        for tid in ids:
            if tid not in matched_tracks:
                self.data[tid]['matched'] = False

    def _spawn(self, det, ts, custom=None):
        tid = custom if custom else str(self._next)
        if not custom:
            self._next += 1
        emb = det.get('emb')
        emb_list = [emb] if emb is not None else []
        self.data[tid] = {
            'id': tid, 'bbox': det['bbox'], 'prev_bbox': det['bbox'],
            'attrs': list(set(det['attrs'])),
            'hist': det.get('hist'),
            'emb': emb,
            'emb_history': emb_list,
            'first_seen': ts, 'last_seen': ts, 'present_time': 0.0,
            'color': (int(np.random.randint(0,255)), int(np.random.randint(0,255)), int(np.random.randint(0,255))),
            'matched': True,
            'last_cam': det.get('cam'), 'prev_cam': None,
            'switch': False, 'switch_count': 0
        }

    def _update(self, tid, det, ts):
        rec = self.data[tid]
        delta = ts - rec['last_seen']
        rec['present_time'] += delta
        rec['last_seen'] = ts
        rec['prev_bbox'] = rec['bbox']
        rec['bbox'] = det['bbox']
        rec['attrs'] = list(set(rec['attrs']).union(set(det['attrs'])))
        rec['hist'] = det.get('hist')
        new_emb = det.get('emb')
        if new_emb is not None:
            rec['emb_history'].append(new_emb)
            if len(rec['emb_history']) > self.max_emb:
                rec['emb_history'].pop(0)
            rec['emb'] = np.mean(rec['emb_history'], axis=0)
        rec['matched'] = True
        new_cam = det.get('cam'); old_cam = rec.get('last_cam')
        if new_cam and old_cam and new_cam != old_cam:
            rec['switch'] = True
            rec['switch_count'] = 5
        rec['prev_cam'] = old_cam; rec['last_cam'] = new_cam

    def purge(self, ts):
        keep = {}
        for k,v in self.data.items():
            if (ts - v['last_seen']) <= self.ttl:
                keep[k] = v
        self.data = keep

    def get_loiterers(self):
        return [v for v in self.data.values() if v['present_time'] >= self.loiter_thresh]

# ----------------- Main processing -----------------
def main():
    start = time.time()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    yolo = YOLO(YOLO_WEIGHTS)
    yolo.classes = [0]  # assuming person class

    moondream = MoondreamExtractor(MOONDREAM_WEIGHTS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reid_name = 'osnet_ain_x1_0'
    reid_input = (256,128)
    reid_extractor = FeatureExtractor(model_name=reid_name, image_size=reid_input, device=device)

    sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("Cannot open:", INPUT_PATH); return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w,h))

    store = TrackStore(loiter_thresh=LOITER_SECONDS, ttl=TRACK_TTL, max_emb=MAX_EMB_HISTORY)

    last_run = -DETECT_INTERVAL
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0

        if (ts - last_run) >= DETECT_INTERVAL:
            last_run = ts
            # predict (yolo) - low conf to get candidates
            predictions = yolo.predict(frame, conf=0.1, classes=[1])
            boxes_np = predictions[0].boxes.data.cpu().numpy()

            detections = []
            for b in boxes_np:
                x1,y1,x2,y2,conf,cls = b
                x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                hist = color_histogram(crop)
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                text = moondream.get_attributes(pil, PROMPT := (
                    "Analyze the person and enumerate clothing, accessories, colors, and distinct features."
                ))
                attrs = moondream.parse_attributes(text)

                rgb_crop = np.asarray(pil)
                emb_batch = reid_extractor([rgb_crop])
                emb_vec = emb_batch[0].cpu().numpy()

                detections.append({
                    'bbox': [x1,y1,x2,y2],
                    'attrs': attrs,
                    'hist': hist,
                    'emb': emb_vec,
                    'cam': None
                })

            store.step_assign(detections, ts, EXT_SYNS, sbert, use_wordnet=USE_WORDNET)
            store.purge(ts)

        # draw
        for tid, rec in store.data.items():
            delta = ts - rec['last_seen']
            if rec['matched'] or delta <= DRAW_PADDING:
                x1,y1,x2,y2 = rec['bbox']
                color = (0,0,255) if rec['present_time'] >= LOITER_SECONDS else rec['color']
                label = f"ID:{tid} time:{rec['present_time']:.1f}s"
                cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                cv2.putText(frame, label, (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        loiterers = store.get_loiterers()
        if loiterers:
            ids = [l['id'][:4] for l in loiterers]
            cv2.putText(frame, "Loiterers: " + ",".join(ids), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        else:
            cv2.putText(frame, "Loiterers: None", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

        writer.write(frame)

    cap.release(); writer.release(); cv2.destroyAllWindows()
    print("Elapsed:", time.time()-start)

if __name__ == "__main__":
    main()
