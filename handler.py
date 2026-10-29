"""Recut vision worker (RunPod serverless): open-vocabulary object boxes (Grounding DINO) + exact masks (SAM).

Input  {"image": <base64 JPEG/PNG>, "queries": ["the red sweater", "the man in white"], "threshold": 0.3,
        "masks": false, "boxes": [[x, y, w, h], …] (fractions; SAM on these instead of detecting), "top": 1}
Output {"detections": [{"query": str, "box": [x, y, w, h] (fractions), "score": float, "mask": <base64 PNG L> | null}],
        "size": [w, h], "ms": int}
Models are cached on the network volume (HF_HOME) so a cold start only loads them."""
import base64
import io
import os
import time

import numpy as np
import runpod
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO = os.environ.get("DINO_MODEL", "IDEA-Research/grounding-dino-base")
SAM = os.environ.get("SAM_MODEL", "facebook/sam-vit-base")
_m = {}


def _load():
    if "dino" not in _m:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        _m["dino_p"] = AutoProcessor.from_pretrained(DINO)
        _m["dino"] = AutoModelForZeroShotObjectDetection.from_pretrained(DINO).to(DEVICE).eval()
    return _m


def _load_sam():
    if "sam" not in _m:
        from transformers import SamModel, SamProcessor
        _m["sam_p"] = SamProcessor.from_pretrained(SAM)
        _m["sam"] = SamModel.from_pretrained(SAM).to(DEVICE).eval()
    return _m


def _img(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def detect(img, queries, threshold=0.3, top=1):
    m = _load()
    W, H = img.size
    text = " ".join(q.strip().rstrip(".").lower() + "." for q in queries if q.strip())
    inputs = m["dino_p"](images=img, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = m["dino"](**inputs)
    res = m["dino_p"].post_process_grounded_object_detection(out, inputs.input_ids, threshold=threshold, text_threshold=threshold,
                                                             target_sizes=[(H, W)])[0]
    labels = res.get("text_labels") or res.get("labels") or []
    dets = []
    for box, score, label in zip(res["boxes"], res["scores"], labels):
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        lab = str(label).strip().lower()
        q = next((q for q in queries if lab and (lab in q.lower() or q.lower().rstrip(".") in lab)), queries[0] if len(queries) == 1 else lab)
        dets.append({"query": q, "label": lab, "box": [round(x0 / W, 4), round(y0 / H, 4), round((x1 - x0) / W, 4), round((y1 - y0) / H, 4)],
                     "score": round(float(score), 4)})
    dets.sort(key=lambda d: -d["score"])
    keep, seen = [], {}
    for d in dets:                                   # the best `top` per query
        if seen.get(d["query"], 0) < top:
            keep.append(d); seen[d["query"]] = seen.get(d["query"], 0) + 1
    return keep


def masks_for(img, boxes):
    """boxes: [[x, y, w, h] fractions] -> [base64 PNG (L, 0/255) of the image size]"""
    m = _load_sam()
    W, H = img.size
    px = [[[b[0] * W, b[1] * H, (b[0] + b[2]) * W, (b[1] + b[3]) * H] for b in boxes]]
    inputs = m["sam_p"](img, input_boxes=px, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = m["sam"](**inputs, multimask_output=True)
    masks = m["sam_p"].image_processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
    scores = out.iou_scores.cpu()[0]
    res = []
    for i in range(masks.shape[0]):
        k = int(torch.argmax(scores[i]))
        a = (masks[i, k].numpy().astype(np.uint8) * 255)
        buf = io.BytesIO(); Image.fromarray(a, "L").save(buf, "PNG", optimize=True)
        res.append(base64.b64encode(buf.getvalue()).decode())
    return res


def handler(job):
    t0 = time.time()
    inp = job.get("input") or {}
    if not inp.get("image"):
        return {"error": "no image"}
    img = _img(inp["image"])
    W, H = img.size
    if inp.get("boxes"):
        dets = [{"query": None, "box": [float(v) for v in b], "score": 1.0} for b in inp["boxes"]]
    else:
        qs = [q for q in (inp.get("queries") or []) if isinstance(q, str) and q.strip()]
        if not qs:
            return {"error": "no queries"}
        dets = detect(img, qs, float(inp.get("threshold") or 0.3), int(inp.get("top") or 1))
    if inp.get("masks") and dets:
        for d, mk in zip(dets, masks_for(img, [d["box"] for d in dets])):
            d["mask"] = mk
    return {"detections": dets, "size": [W, H], "ms": int((time.time() - t0) * 1000)}


if __name__ == "__main__":
    if os.environ.get("WARM", "1") == "1":
        _load()
    runpod.serverless.start({"handler": handler})
