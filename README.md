# Uni-SLM: Unified Sign Language Model

> **Unified Sign Language Model for Multi-Task Understanding and Generation**

---

## 🌍 Overview

**Uni-SLM** is a unified framework for **sign language understanding** that integrates multiple core tasks within a single model:

- **Sign Language Recognition (CSLR)**  
- **Sign Language Retrieval (Text↔Video)**  
- **Sign Language Translation (SLT)**  
- **Sign Language Segmentation (SLS)**
- (Future) **Generation**

The framework aims to share a **common multimodal encoder** (RGB, Pose, and Text) and support **task-specific heads** for diverse downstream tasks, following a unified **Adaptive Information Bottleneck (AdaIB)** optimization principle.

---


## ⚙️ Installation

### 1️⃣ Create Environment
```bash
git clone https://github.com/pxl416/Uni-SLM.git
cd uni-slm

conda create -n uni-slm python=3.10
conda activate uni-slm
pip install -r requirements.txt
```

### 2️⃣ Install Optional Components

For translation tasks (mT5 or multilingual support):

Dataset Setup

Uni-SLM currently supports datasets including:

CSL-Daily

CSL-News

BOBSL (partial support)

