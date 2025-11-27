import os
import sys
import cv2
import json
import time
import random
import torch
import numpy as np
import platform
import subprocess
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, VitPoseForPoseEstimation

# ---- SpinePose
from spinepose import SpinePoseEstimator
from spinepose.tools.smoothing import KeypointSmoothing

# ===========================
# Determinism utilities
# ===========================
def set_deterministic(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN determinism (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_git_commit_short() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def get_script_name() -> str:
    try:
        return Path(__file__).name
    except Exception:
        return "unknown"

# ===========================
# COCO / Skeleton constants
# ===========================
LEFT_JOINTS  = {5, 7, 9, 11, 13, 15}
RIGHT_JOINTS = {6, 8,10, 12, 14, 16}

PARTS_DEF = {
    "torso": {"joints": {5, 6, 11, 12}, "edges": {(5, 6), (11, 12), (5, 11), (6, 12)}},
    "hands": {"joints": {5, 6, 7, 8, 9, 10}, "edges": {(5, 7), (7, 9), (6, 8), (8, 10)}},
    "knee":  {"joints": {11, 12, 13, 14}, "edges": {(11, 13), (12, 14)}}
}

SP_NAME_TO_ID = {
    "neck": 18, "hip": 19,
    "spine_01": 26, "spine_02": 27, "spine_03": 28, "spine_04": 29, "spine_05": 30,
    "neck_02": 35, "neck_03": 36
}
SP_NECK_SPINE_ORDER = ["neck_03", "neck_02", "neck", "spine_05", "spine_04", "spine_03", "spine_02", "spine_01", "hip"]
SP_NECK_SPINE_IDS = [SP_NAME_TO_ID[n] for n in SP_NECK_SPINE_ORDER]
SP_NECK_SPINE_EDGES = [(SP_NAME_TO_ID[a], SP_NAME_TO_ID[b]) for a, b in zip(SP_NECK_SPINE_ORDER[:-1], SP_NECK_SPINE_ORDER[1:])]

# ===========================
# Drawing helpers
# ===========================
def draw_text(image, text, pos=(0, 0),
              font=cv2.FONT_HERSHEY_PLAIN, font_scale=1,
              font_thickness=1, text_color=(255, 255, 255),
              text_color_bg=(0, 0, 255)):
    x, y = pos
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, font_thickness)
    th = th + bl
    cv2.rectangle(image, (x, y - th - 4), (x + tw + 8, y + 4), text_color_bg, -1)
    cv2.putText(image, text, (x + 4, y - 4), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

def xywh_center_to_ltrb(x, y, w, h):
    x1 = x - w/2.0; y1 = y - h/2.0
    x2 = x + w/2.0; y2 = y + h/2.0
    return float(x1), float(y1), float(x2), float(y2)

def xywh_center_to_ltrb_only(cx, cy, w, h):
    l = float(cx - w/2.0); t = float(cy - h/2.0)
    r = float(cx + w/2.0); b = float(cy + h/2.0)
    return np.array([[l, t, r, b]], dtype=np.float32)

def np_to_float(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=float).tolist()

# ===========================
# Fused skeleton drawer
# ===========================
def draw_fused_skeleton(image_bgr: np.ndarray,
                        coco_k: np.ndarray, coco_conf: np.ndarray,
                        spine_k: np.ndarray, spine_conf: np.ndarray,
                        side: str,
                        kp_thr: float = 0.30):
    def ok(p, c): return p is not None and c is not None and c >= kp_thr

    # normalize inputs
    if coco_k is None: coco_k = np.zeros((0, 2), float)
    coco_k = np.asarray(coco_k, float)
    if coco_k.ndim == 2 and coco_k.shape[1] >= 3:
        coco_conf = coco_k[:, 2]; coco_k = coco_k[:, :2]
    else:
        coco_conf = np.asarray(coco_conf, float) if coco_conf is not None else np.ones(len(coco_k))

    if spine_k is None: spine_k = np.zeros((0, 2), float)
    spine_k = np.asarray(spine_k, float)
    if spine_k.ndim == 2 and spine_k.shape[1] >= 3:
        spine_conf = spine_k[:, 2]; spine_k = spine_k[:, :2]
    else:
        spine_conf = np.asarray(spine_conf, float) if spine_conf is not None else np.ones(len(spine_k))

    H, W = image_bgr.shape[:2]
    lw = max(2, int(round(W / 800 * 2)))
    rad = max(3, int(round(W / 800 * 3)))

    CO = dict(l_sh=5, r_sh=6, l_el=7, r_el=8, l_wr=9, r_wr=10, l_hip=11, r_hip=12, l_kn=13, r_kn=14)

    def p_coco(i):
        if i is None or i >= len(coco_k): return (None, None)
        return coco_k[i], coco_conf[i]

    def p_sp(name):
        idx = SP_NAME_TO_ID.get(name, None)
        if idx is None or idx >= len(spine_k): return (None, None)
        return spine_k[idx], spine_conf[idx]

    # 1) spine chain (green)
    if len(spine_k) >= 31:
        for nm in SP_NECK_SPINE_ORDER:
            pt, c = p_sp(nm)
            if ok(pt, c):
                cv2.circle(image_bgr, tuple(np.int32(pt)), rad, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        for i, j in SP_NECK_SPINE_EDGES:
            pi, ci = (spine_k[i], spine_conf[i]) if i < len(spine_k) else (None, None)
            pj, cj = (spine_k[j], spine_conf[j]) if j < len(spine_k) else (None, None)
            if ok(pi, ci) and ok(pj, cj):
                cv2.line(image_bgr, tuple(np.int32(pi)), tuple(np.int32(pj)), (0, 255, 0), lw, lineType=cv2.LINE_AA)
    else:
        valid = [(tuple(np.int32(pt)), c) for pt, c in zip(spine_k, spine_conf) if ok(pt, c)]
        for (pt, _) in valid:
            cv2.circle(image_bgr, pt, rad, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        for a, b in zip(valid[:-1], valid[1:]):
            cv2.line(image_bgr, a[0], b[0], (0, 255, 0), lw, lineType=cv2.LINE_AA)

    # 2) bridges (shoulders/hips to neck/hip)
    neck3, c_n3 = p_sp("neck_03")
    neck,  c_n  = p_sp("neck")
    hip_c, c_h  = p_sp("hip")
    lsh,  c_lsh = p_coco(CO["l_sh"])
    rsh,  c_rsh = p_coco(CO["r_sh"])
    lhip, c_lh  = p_coco(CO["l_hip"])
    rhip, c_rh  = p_coco(CO["r_hip"])

    neck_anchor = neck if ok(neck, c_n) else neck3
    if ok(neck_anchor, c_n if neck_anchor is neck else c_n3):
        if side in ("left", "all") and ok(lsh, c_lsh):
            cv2.line(image_bgr, tuple(np.int32(lsh)), tuple(np.int32(neck_anchor)), (0, 255, 0), lw, lineType=cv2.LINE_AA)
        if side in ("right", "all") and ok(rsh, c_rsh):
            cv2.line(image_bgr, tuple(np.int32(rsh)), tuple(np.int32(neck_anchor)), (0, 255, 0), lw, lineType=cv2.LINE_AA)

    if ok(hip_c, c_h):
        if side in ("left", "all") and ok(lhip, c_lh):
            cv2.line(image_bgr, tuple(np.int32(hip_c)), tuple(np.int32(lhip)), (0, 255, 0), lw, lineType=cv2.LINE_AA)
        if side in ("right", "all") and ok(rhip, c_rh):
            cv2.line(image_bgr, tuple(np.int32(hip_c)), tuple(np.int32(rhip)), (0, 255, 0), lw, lineType=cv2.LINE_AA)

    # 3) limbs (arms full; legs hip->knee only)
    def draw_limb(a, b, col):
        pa, ca = p_coco(a); pb, cb = p_coco(b)
        if ok(pa, ca) and ok(pb, cb):
            cv2.line(image_bgr, tuple(np.int32(pa)), tuple(np.int32(pb)), col, lw, lineType=cv2.LINE_AA)

    if side in ("left", "all"):
        draw_limb(CO["l_sh"], CO["l_el"], (255, 0, 255))
        draw_limb(CO["l_el"], CO["l_wr"], (255, 0, 255))
        draw_limb(CO["l_hip"], CO["l_kn"], (255, 255, 0))
    if side in ("right", "all"):
        draw_limb(CO["r_sh"], CO["r_el"], (255, 0, 255))
        draw_limb(CO["r_el"], CO["r_wr"], (255, 0, 255))
        draw_limb(CO["r_hip"], CO["r_kn"], (255, 255, 0))

    # dots for the limbs drawn
    dots = []
    if side in ("left", "all"):
        dots += [5,7,9,11,13]
    if side in ("right", "all"):
        dots += [6,8,10,12,14]
    for idx in dots:
        pt, c = p_coco(idx)
        if ok(pt, c):
            cv2.circle(image_bgr, tuple(np.int32(pt)), max(2, rad-1), (255, 255, 255), -1, lineType=cv2.LINE_AA)

    return image_bgr

# ===========================
# Pose helpers (factor out repeated code)
# ===========================
def _pose_coco_on_box(frame_bgr, box_xywh_center, pose_processor, pose_model, device,
                      pose_model_name: str, do_smooth: bool, filters_dict, target_id):
    """
    Returns:
      coco_xy (Nx2 or None), coco_conf (N or None),
      keypoints_json (list), scores_json (list),
      mean_conf (float or None), elapsed_sec (float)
    """
    if box_xywh_center is None:
        return None, None, [], [], None, 0.0

    t0 = time.perf_counter()
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cx, cy, w, h = box_xywh_center

    pose_outputs = detect_pose(
        pil_img,
        [list([cx, cy, w, h])],
        pose_processor,
        pose_model,
        device,
        pose_model_name
    )
    elapsed = time.perf_counter() - t0

    # --- normalize: we want a single dict with "keypoints","scores" ---
    if pose_outputs is None:
        return None, None, [], [], None, elapsed

    if isinstance(pose_outputs, list):
        # list-of-poses (possibly empty) -> take first if exists
        if len(pose_outputs) == 0:
            return None, None, [], [], None, elapsed
        pose_outputs = pose_outputs[0]

    if not isinstance(pose_outputs, dict):
        # unexpected structure
        return None, None, [], [], None, elapsed

    coco_k = pose_outputs.get("keypoints", None)
    coco_s = pose_outputs.get("scores", None)

    if coco_k is not None and isinstance(coco_k, torch.Tensor):
        coco_k = coco_k.detach().cpu().numpy()

    if coco_k is not None:
        # optional smoothing
        if do_smooth and target_id in filters_dict:
            k = np.asarray(coco_k, dtype=float)
            xy_coco = k[:, :2]
            xy_coco = filters_dict[target_id](xy_coco)
            if k.shape[1] >= 3:
                k[:, :2] = xy_coco
            else:
                k = np.concatenate([xy_coco, np.ones((xy_coco.shape[0], 1), dtype=float)], axis=1)
            coco_k = k

        coco_conf = coco_k[:, 2] if coco_k.shape[1] >= 3 else np.ones(coco_k.shape[0])
        coco_xy   = coco_k[:, :2]
        mean_conf = float(np.mean(coco_conf)) if coco_conf is not None and len(coco_conf) else None

        return (
            coco_xy,
            coco_conf,
            np_to_float(coco_k),
            np_to_float(coco_s) if coco_s is not None else [],
            mean_conf,
            elapsed
        )

    return None, None, [], [], None, elapsed


def _pose_spine_on_box(spine_est: SpinePoseEstimator, frame_bgr, box_xywh_center,
                       do_smooth: bool, filters_dict, target_id,
                       kp_idx_full=SP_NECK_SPINE_IDS):
    """
    Returns:
      sk (Kx2 or None), ss (K or None),
      spine_keypoints_json (list), spine_scores_json (list),
      mean_conf (float or None), elapsed_sec (float)
    """
    if box_xywh_center is None:
        return None, None, [], [], None, 0.0

    bboxes_ltrb = xywh_center_to_ltrb_only(*box_xywh_center)
    t0 = time.perf_counter()
    skps, sscores = spine_est.estimate(frame_bgr, bboxes_ltrb)
    elapsed = time.perf_counter() - t0

    if skps is None or len(skps) == 0:
        return None, None, [], [], None, elapsed

    sk = np.asarray(skps[0], dtype=float)
    if sk.shape[1] >= 3:
        sk = sk[:, :2]
    ss = np.asarray(sscores[0], dtype=float) if isinstance(sscores, (list, np.ndarray)) else None

    if do_smooth:
        if target_id not in filters_dict:
            filters_dict[target_id] = KeypointSmoothing(
                num_keypoints=sk.shape[0],
                freq=filters_dict["_cfg"]["freq"],
                mincutoff=filters_dict["_cfg"]["mincutoff"],
                beta=filters_dict["_cfg"]["beta"],
                dcutoff=filters_dict["_cfg"]["dcutoff"]
            )
        sk = filters_dict[target_id](sk)

    # JSON view (either neck+spine subset if full set, else all)
    if sk.shape[0] >= 31:
        idxs = kp_idx_full
        spine_keypoints = sk[idxs].tolist()
        if ss is not None and len(ss) > max(idxs):
            spine_scores = ss[idxs].tolist()
        else:
            spine_scores = [1.0]*len(idxs)
        mean_conf = float(np.mean(spine_scores)) if len(spine_scores) else None
    else:
        spine_keypoints = sk.tolist()
        spine_scores = ss.tolist() if ss is not None else [1.0]*sk.shape[0]
        mean_conf = float(np.mean(spine_scores)) if len(spine_scores) else None

    return sk, ss, spine_keypoints, spine_scores, mean_conf, elapsed

def _build_payload(video_path: str, frame_idx: int, W: int, H: int, target_id: int,
                   bbox_xywh_center, side: str, parts: set,
                   kps_out, scores_out, spine_keypoints, spine_scores,
                   id_detected: bool, pose_source: str,
                   args,):
    x, y, w, h = bbox_xywh_center if bbox_xywh_center is not None else (None, None, None, None)
    bbox_ltrb = list(xywh_center_to_ltrb(x, y, w, h)) if bbox_xywh_center is not None else []

    return {
        "video": str(video_path),
        "frame_index": int(frame_idx),
        "track_id": int(target_id),
        "image_size": {"width": int(W), "height": int(H)},
        "bbox_xywh_center": [float(x), float(y), float(w), float(h)] if bbox_xywh_center is not None else [],
        "bbox_ltrb": bbox_ltrb,
        "side": side,
        "parts": sorted(list(parts)),
        "keypoints": kps_out,          # ViTPose COCO-17 (list of [x,y,conf])
        "scores": scores_out,
        "spine_keypoints": spine_keypoints,
        "spine_scores": spine_scores,
        "id_detected": bool(id_detected),
        "pose_source": pose_source,
        "smoothing": bool(args.smooth),
        "smoothing_cfg": {
            "freq": args.smooth_fps,
            "mincutoff": args.smooth_mincut,
            "beta": args.smooth_beta,
            "dcutoff": args.smooth_dcut
        } if args.smooth else {}
    }

# ===========================
# Main
# ===========================
import onnxruntime as ort
def pick_spine_device(prefer_cuda=True):
    try:
        providers = ort.get_available_providers()
    except Exception:
        providers = ['CPUExecutionProvider']
    has_cuda_ort = 'CUDAExecutionProvider' in providers
    return 'cuda' if (prefer_cuda and has_cuda_ort) else 'cpu'


def main(args):
    # ---- Determinism
    set_deterministic(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    det_model = YOLO(args.yolo_model)
    pose_processor = AutoProcessor.from_pretrained(args.pose_model)
    pose_model = VitPoseForPoseEstimation.from_pretrained(args.pose_model).to(device).eval()

    spine_device = pick_spine_device(prefer_cuda=True)
    try:
        spine_est = SpinePoseEstimator(device=spine_device)
    except Exception as e:
        print(f"[SpinePose] Init failed on {spine_device}: {e} — falling back to CPU.")
        spine_est = SpinePoseEstimator(device='cpu')

    print("[Devices] torch.cuda.is_available():", torch.cuda.is_available())
    print("[Devices] ONNX providers:", ort.get_available_providers())
    print("[Devices] SpinePose device:", spine_device)
    # Smoothers (store OneEuro config under "_cfg" to build dynamic filters if needed)
    spine_filters = {"_cfg": dict(freq=args.smooth_fps, mincutoff=args.smooth_mincut,
                                  beta=args.smooth_beta, dcutoff=args.smooth_dcut)}
    coco_filters  = {"_cfg": dict(freq=args.smooth_fps, mincutoff=args.smooth_mincut,
                                  beta=args.smooth_beta, dcutoff=args.smooth_dcut)}
    if args.smooth:
        spine_filters[args.target_id] = KeypointSmoothing(
            num_keypoints=37, freq=args.smooth_fps, mincutoff=args.smooth_mincut,
            beta=args.smooth_beta, dcutoff=args.smooth_dcut
        )
        coco_filters[args.target_id] = KeypointSmoothing(
            num_keypoints=17, freq=args.smooth_fps, mincutoff=args.smooth_mincut,
            beta=args.smooth_beta, dcutoff=args.smooth_dcut
        )

    parts = set(args.parts)  # currently not used by the fused-draw, but kept for schema/backward-compat

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = out_dir / f"{Path(args.input).stem}_tracked.avi"
    writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (W, H))

    # one JSON per video — collect all frames here
    all_frames = []

    # profiling aggregates
    t_vit_total = 0.0
    t_sp_total  = 0.0
    n_vit = 0
    n_sp  = 0
    coco_conf_sum = 0.0
    coco_conf_ct  = 0
    spine_conf_sum = 0.0
    spine_conf_ct  = 0

    frame_idx = 0
    last_box = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        annotated = frame.copy()
        results = det_model.track(frame, persist=True, conf=args.conf,
                                  tracker=args.tracker, classes=[0], imgsz=args.imgsz)
        # runtime check
        assert results is not None and len(results) > 0, "YOLO returned empty results list"

        r0 = results[0]
        id_detected = False
        pose_source = "none"
        bbox_used = None

        # try to find target-id
        if r0.boxes is not None and len(r0.boxes) > 0:
            xywh = r0.boxes.xywh.cpu().numpy()
            ids = r0.boxes.id.cpu().numpy() if r0.boxes.id is not None else [-1]*len(xywh)

            for box, tid in zip(xywh, ids):
                if int(tid) == int(args.target_id):
                    id_detected = True
                    bbox_used = box
                    x, y, w, h = box
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    draw_text(annotated, f"ID {tid}", (x1, y1 - 10))
                    pose_source = "detected"
                    last_box = box
                    break

        # pose on chosen box (detected or fallback)
        if not id_detected:
            if last_box is not None:
                x, y, w, h = last_box
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 200), 2)
                draw_text(annotated, f"ID {args.target_id} (last,reposed)", (x1, y1 - 10))
                bbox_used = last_box
                pose_source = "last_box_fallback"
            else:
                bbox_used = None
                pose_source = "no_box"

        # run COCO + Spine (timed)
        coco_xy, coco_conf, kps_out, scores_out, coco_mean, t_vit = _pose_coco_on_box(
            annotated, bbox_used, pose_processor, pose_model, device,
            args.pose_model, args.smooth, coco_filters, args.target_id
        )
        if coco_mean is not None:
            coco_conf_sum += coco_mean
            coco_conf_ct  += 1
        if t_vit > 0:
            t_vit_total += t_vit
            n_vit += 1

        sk, ss, spine_keypoints, spine_scores, spine_mean, t_sp = _pose_spine_on_box(
            spine_est, annotated, bbox_used, args.smooth, spine_filters, args.target_id
        )
        if spine_mean is not None:
            spine_conf_sum += spine_mean
            spine_conf_ct  += 1
        if t_sp > 0:
            t_sp_total += t_sp
            n_sp += 1

        # fused draw
        annotated = draw_fused_skeleton(annotated, coco_xy, coco_conf, sk, ss, side=args.side, kp_thr=args.kp_thr)

        # build payload + collect
        payload = _build_payload(
            video_path=args.input,
            frame_idx=frame_idx, W=W, H=H, target_id=args.target_id,
            bbox_xywh_center=tuple(bbox_used) if bbox_used is not None else None,
            side=args.side, parts=parts,
            kps_out=kps_out, scores_out=scores_out,
            spine_keypoints=spine_keypoints, spine_scores=spine_scores,
            id_detected=id_detected, pose_source=pose_source,
            args=args
        )
        all_frames.append(payload)

        # per-frame logline
        if bbox_used is not None:
            _, _, w, h = bbox_used
            bbox_txt = f"{w:.1f}x{h:.1f}"
        else:
            bbox_txt = "0x0"
        coco_txt  = f"{coco_mean:.3f}" if coco_mean is not None else "nan"
        spine_txt = f"{spine_mean:.3f}" if spine_mean is not None else "nan"
        print(f"[{frame_idx:06d}] {'DETECT' if id_detected else 'FALLBK':6s} "
              f"bbox={bbox_txt:>11s} coco_m={coco_txt:>6s} spine_m={spine_txt:>6s} "
              f"t_vit={t_vit*1000:.1f}ms t_sp={t_sp*1000:.1f}ms")

        # write annotated video frame
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    # ---- ONE JSON per video (meta + all frames) ----
    out_json = out_dir / f"{Path(args.input).stem}_poses.json"
    avg_vit_ms = (t_vit_total / n_vit * 1000.0) if n_vit > 0 else None
    avg_sp_ms  = (t_sp_total  / n_sp  * 1000.0) if n_sp  > 0 else None
    avg_coco_conf  = (coco_conf_sum  / coco_conf_ct)  if coco_conf_ct  > 0 else None
    avg_spine_conf = (spine_conf_sum / spine_conf_ct) if spine_conf_ct > 0 else None

    video_meta = {
        "version": "1.0.0",
        "script": get_script_name(),
        "git_commit": get_git_commit_short(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "env": {
            "torch": torch.__version__,
            "ultralytics": getattr(YOLO, "__module__", "ultralytics"),
            "transformers": AutoProcessor.__module__,
            "spinepose": "spinepose"
        },
        "seed": int(args.seed),
        "video": str(args.input),
        "width": int(W),
        "height": int(H),
        "fps": float(fps),
        "frames_total_reported": int(n_total_frames),
        "frames_processed": int(len(all_frames)),
        "target_id": int(args.target_id),
        "side": args.side,
        "pose_model": args.pose_model,
        "yolo_model": args.yolo_model,
        "tracker": args.tracker,
        "smoothing": bool(args.smooth),
        "smoothing_cfg": {
            "freq": args.smooth_fps,
            "mincutoff": args.smooth_mincut,
            "beta": args.smooth_beta,
            "dcutoff": args.smooth_dcut
        } if args.smooth else {},
        "profiling": {
            "avg_vitpose_ms": avg_vit_ms,
            "avg_spine_ms": avg_sp_ms,
            "avg_coco_conf": avg_coco_conf,
            "avg_spine_conf": avg_spine_conf
        }
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"meta": video_meta, "frames": all_frames}, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved video to {out_video_path}")
    print(f"✅ Saved {len(all_frames)} frames")
    print(f"✅ Saved JSON to {out_json}")


# ===========================
# CLI
# ===========================
def detect_pose(pil_image, person_boxes_center_xywh, pose_processor, pose_model, device, pose_model_name: str):
    if not person_boxes_center_xywh:
        return []
    coco_xywh = []
    for cx, cy, w, h in person_boxes_center_xywh:
        x_min = float(cx - w / 2.0); y_min = float(cy - h / 2.0)
        coco_xywh.append([x_min, y_min, float(w), float(h)])
    inputs = pose_processor(pil_image, boxes=[coco_xywh], return_tensors='pt').to(device)
    with torch.no_grad():
        if "plus" in pose_model_name.lower():
            outputs = pose_model(**inputs, dataset_index=torch.tensor([0], device=device))
        else:
            outputs = pose_model(**inputs)
    pose_results = pose_processor.post_process_pose_estimation(outputs, boxes=[coco_xywh])
    return pose_results[0] if len(pose_results) else []

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--target-id", type=int, default=1)
    parser.add_argument("--pose-model", type=str, default="usyd-community/vitpose-plus-base")
    parser.add_argument("--yolo-model", type=str, default="yolo11x.pt")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--kp-thr", type=float, default=0.30)
    parser.add_argument("--side", type=str, choices=["left", "right", "all"], default="all")
    parser.add_argument("--parts", nargs="+", choices=["torso", "hands", "knee"],
                        default=["torso", "hands", "knee"])
    parser.add_argument("--smooth", action="store_true", default=True)
    parser.add_argument("--smooth-fps", type=float, default=30.0)
    parser.add_argument("--smooth-mincut", type=float, default=0.1)
    parser.add_argument("--smooth-beta", type=float, default=0.1)
    parser.add_argument("--smooth-dcut", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    main(args)