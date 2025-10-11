# 🧠 SigLIP Person Finder

This project implements an *open-set person search system* using *SigLIP* — enabling text-based retrieval of people in *images* or *videos* using natural language descriptions.

---

## 🚀 Overview

The model allows users to *find a person in an image or video* by describing them in text.  
Example:  
> “A man wearing a white shirt and holding a backpack.”

Using *SigLIP embeddings*, the system compares the text prompt and detected people, returning the most visually matching results.

---

## 🧩 Key Features

- 🔍 *Text-based Person Search:* Retrieve people using descriptive sentences.  
- 🧠 *SigLIP Model:* Uses paired image-text embeddings for open-set retrieval.  
- 🧮 *YOLOv8 Detection:* Detects individuals in frames or images.  
- 🧾 *Cosine Similarity Matching:* Measures similarity between text and image embeddings.  
- ⚡ *Tracking Optimization:* Reduces redundant computations for real-time video inference.  
- 🎥 *Multi-view Dataset:* Trained and evaluated on a multi-view re-identification dataset.

---

## 📚 Dataset

*Base Dataset:* Market-1501 (extended version)  
*Enhanced Attributes:*
- Natural language descriptions  
- Multi-view person samples  
- Clothing and posture attributes  
- Consistent labeling across all views  

📦 Dataset Size: ~6,400 samples  
📍 Source: Hugging Face — Multiview ReID Dataset with Descriptions

---

## 🛠 Tech Stack

| Category | Tools & Libraries |
|-----------|------------------|
| Framework | PyTorch, Transformers |
| Detection | YOLOv8 |
| Text & Vision Model | SigLIP |
| Utilities | NumPy, Pillow, Datasets |
| Tracking & Visualization | FiftyOne, OpenCV, Gradio |
| Logging | Weights & Biases (W&B) |

---

## 🧪 Model Training

| Parameter | Details |
|------------|----------|
| Model | google/siglip-base-patch16-224 |
| Epochs | 10 |
| Batch Size | 16 |
| Loss | Symmetric cosine (InfoNCE-style) |
| Optimizer | AdamW with cosine scheduler |
| Metric | Recall@1, Recall@5 |

---

## 🖥 How to Run

1. *Clone the Repository*
   ```bash
   git clone https://github.com/ziaee-mohammad/SigLIP-Person-Finder.git
   cd SigLIP-Person-Finder


---

## ⚠ Limitations & Ethics

- The dataset is *synthetic* and intended for research only.
- Not suitable for *surveillance* or *law enforcement* applications.
- AI-generated attributes may contain occasional *biases* or *errors*.

---

## 📈 Results

| Metric      | Value                         |
|------------|--------------------------------|
| Recall@1   | *0.31*                       |
| Recall@5   | *0.56*                       |
| FPS (video)| ~*18* (optimized with tracking) |

---

## 🧠 Author

*Mohammad Ziaee*  
📧 [moha2012zia@gmail.com](mailto:moha2012zia@gmail.com)  
🌐 [GitHub Profile](https://github.com/ziaee-mohammad)


