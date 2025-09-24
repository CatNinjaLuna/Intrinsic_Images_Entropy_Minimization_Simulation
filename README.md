# Intrinsic Images by Entropy Minimization

This repository explores the algorithms and methods described in the paper:

> **Finlayson, G. D., Drew, M. S., & Lu, C. (2004). Intrinsic Images by Entropy Minimization. In _European Conference on Computer Vision (ECCV)_, LNCS 3023, pp. 582–595. Springer.**

The project focuses on **illumination-invariant image formation**: recovering images that are free from shadows and lighting effects by projecting chromaticity values into a direction that minimizes entropy.

---

## Overview

Illumination changes and shadows often cause problems for segmentation, recognition, and tracking in computer vision. The goal of this work is to derive an **intrinsic image** (reflectance-only representation) that is invariant to lighting conditions.

The key innovation of the paper is that the **invariant projection direction** can be found without camera calibration. Instead, it is the direction in log-chromaticity space that **minimizes entropy** of the resulting greyscale image.

---

## Key Concepts

-  **Intrinsic Image**: Image representation where shading and illumination are factored out, leaving only the inherent reflectance of surfaces.
-  **Log-Chromaticity Space**: Transformation of RGB ratios into log space, where changes in lighting form straight, parallel lines.
-  **Entropy Minimization**: Projection direction is chosen by minimizing entropy of the greyscale image histogram, producing well-separated peaks for each reflectance.

---

## Algorithm Steps:contentReference[oaicite:1]{index=1}

1. **Convert to Log-Chromaticity**

   -  Compute chromaticities using the **geometric mean normalization**:
      \[
      c_k = \frac{R_k}{\sqrt[3]{R \cdot G \cdot B}}
      \]
   -  Take logarithms to map into 2D log-chromaticity space.

2. **Search for Invariant Direction**

   -  For each projection angle θ ∈ [0°, 180°]:
      -  Project log-chromaticities onto a line.
      -  Build histogram of projected values (ignoring outliers).
      -  Compute entropy:
         \[
         \eta = - \sum_i p_i \log(p_i)
         \]

3. **Select Minimum Entropy**

   -  The projection direction with the lowest entropy is the **invariant direction**.
   -  The projected image forms the **shadow-free intrinsic image**.

4. **Reconstruction (Optional)**
   -  Convert greyscale invariant image into **L1-chromaticity** representation.
   -  Re-integrate into a full-color shadow-free image using edge-based reconstruction.

---

---

## 🚀 Usage

1. **Clone the repo**

   ```bash
   git clone https://github.com/CatNinjaLuna/Intrinsic_Images_Illumination_Invariance.git
   cd Intrinsic_Images_Illumination_Invariance

   ```

2. **Run preprocessing & entropy minimization**
   python src/entropy_search.py --input data/example.jpg --output results/invariant.png
