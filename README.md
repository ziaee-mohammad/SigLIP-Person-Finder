# 👤 SigLIP-Person-Finder

Open‑set, text‑guided **person search** using **SigLIP** embeddings and **YOLOv8** person detection.  
Given a natural‑language prompt (e.g., *"man with a white shirt and a backpack"*), the system ranks persons in images/videos via **cosine similarity**. Optional **tracking** aggregates scores across frames for robust video search.

---

## ✨ What this repo does
- **Detect persons** in frames with YOLOv8 and crop them
- **Embed** both **text** and **image crops** with **SigLIP**
- **Rank** detections by **cosine similarity** (open‑set retrieval)
- (Optional) **Track** identities over time and aggregate scores
- Provide a clean path from **notebooks → scripts → reproducible inference**

---

## 🗂 Repository Structure (suggested)
```
SigLIP-Person-Finder/
├─ src/
│  ├─ detector.py        # YOLOv8 person detection
│  ├─ embedder.py        # SigLIP text/image embeddings + normalization
│  ├─ search.py          # cosine similarity + ranking
│  ├─ tracker.py         # optional: SORT/ByteTrack integration
│  └─ utils.py           # I/O, drawing, timing
├─ infer.py              # CLI: image/video + prompt → ranked boxes
├─ notebooks/
│  └─ openset_reid_colab.ipynb
├─ requirements.txt
├─ .gitignore
└─ README.md
```

> If you already have a Colab notebook, keep it under `notebooks/` and mirror the steps in `infer.py`.

---

## ⚙️ Installation
```bash
git clone https://github.com/ziaee-mohammad/SigLIP-Person-Finder.git
cd SigLIP-Person-Finder
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt (minimal)**
```
torch
torchvision
transformers
sentencepiece
ultralytics
opencv-python
numpy
scipy
faiss-cpu         # optional: fast retrieval index
```

> Ensure you have a working CUDA/PyTorch install if you plan to use GPU acceleration.

---

## 🚀 Quick Start (CLI)
**Image mode**
```bash
python infer.py   --source assets/street.jpg   --prompt "man with a white shirt and backpack"   --yolo yolov8n.pt   --siglip google/siglip-base-patch16-256   --save outputs/street_annotated.jpg
```

**Video mode (with simple tracking)**
```bash
python infer.py   --source assets/cam01.mp4   --prompt "woman in red dress"   --track   --save outputs/cam01_annotated.mp4
```

**Notes**
- Outputs (annotated frames/video + JSON of boxes/scores) are saved under `outputs/`
- Use a stronger YOLO model (e.g., `yolov8s.pt`) for better detection quality
- For large galleries, consider **FAISS** and cache image embeddings

---

## 🧠 How it works
1. **Detection** — Run YOLOv8 on each frame; keep class `person` only.  
2. **Embedding** — For each crop, encode with SigLIP (L2 normalize). Encode the **text prompt** once.  
3. **Similarity** — Compute **cosine** = dot product of L2‑normalized vectors; rank descending.  
4. **Tracking (optional)** — Associate boxes across frames; aggregate track score by mean/max.  
5. **Output** — Draw top matches with scores; export JSON (boxes, ids, scores, frame index).

---

## 🧪 Benchmark (replace with your real numbers)
| Scenario | Top‑1 Acc | Recall@5 | FPS (1080p) | Notes |
|---|---:|---:|---:|---|
| Single image (indoor) | 0.78 | 0.92 | 18 | A5000, batch=1 |
| Multi‑person video | 0.71 | 0.88 | 15 | YOLOv8s + simple tracker |

> Report dataset/source and hardware. Provide fixed seeds and identical prompts for repeatability.

---

## 🔍 Inference Sketch (core logic)
```python
# core idea used in infer.py
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModel
import torch, cv2, numpy as np
from numpy.linalg import norm

def l2n(x): return x / (norm(x) + 1e-9)

det = YOLO("yolov8n.pt")
proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-256")
sig  = AutoModel.from_pretrained("google/siglip-base-patch16-256")

def txt_emb(prompt: str):
    inp = proc(text=[prompt], return_tensors="pt", padding=True)
    with torch.no_grad():
        z = sig.get_text_features(**inp)[0].cpu().numpy()
    return l2n(z)

def img_emb(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = proc(images=rgb, return_tensors="pt")
    with torch.no_grad():
        z = sig.get_image_features(**inp)[0].cpu().numpy()
    return l2n(z)

def score_frame(frame, prompt_vec):
    res = det(frame, classes=[0])   # 0 = person
    boxes, scores = [], []
    for r in res:
        for xyxy in r.boxes.xyxy.cpu().numpy().astype(int):
            x1,y1,x2,y2 = xyxy
            crop = frame[y1:y2, x1:x2]
            s = float(np.dot(prompt_vec, img_emb(crop)))
            boxes.append((x1,y1,x2,y2)); scores.append(s)
    return boxes, scores
```

---

## 🧩 Tips & Good Practices
- **Normalize** both text & image embeddings (L2) → cosine via dot product.  
- **Prompt engineering** matters; be descriptive (colors, clothing, accessories).  
- Batch crops for speed; cache embeddings if gallery is static.  
- For videos, fuse **detection score × similarity** to filter low‑confidence boxes.  
- Respect **privacy**: do not share identifiable footage without consent.

---

## 🔐 Ethics & Privacy
This project is for **research and educational** use. If applied to real footage, ensure legal compliance and obtain necessary permissions/consents.

---

## 📝 Suggested Description (GitHub About)
> Open‑set, text‑guided **person search** using SigLIP embeddings and YOLOv8 detection, with cosine similarity ranking and optional real‑time tracking for videos.

## 🏷 Suggested Topics
```
computer-vision
multimodal
image-retrieval
open-set
person-search
siglip
yolov8
re-identification
tracking
python
```

---

## 📜 License
MIT — feel free to use and adapt with attribution.

---

## 👤 Author
**Mohammad Ziaee** — Computer Engineer | AI & Data Science  
📧 moha2012zia@gmail.com  
🔗 https://github.com/ziaee-mohammad
👉 Instagram: [@ziaee_mohammad](https://www.instagram.com/ziaee_mohammad/)


