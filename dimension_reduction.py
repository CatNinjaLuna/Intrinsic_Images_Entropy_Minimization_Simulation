import numpy as np
import matplotlib.pyplot as plt

# ---- 1) Create a toy RGB image with multiplicative illumination (shadows) ----
H, W = 120, 180
img = np.zeros((H, W, 3), dtype=np.float32)

# Three vertical materials (reflectances): reddish, greenish, bluish
reflectances = [
    np.array([0.8, 0.2, 0.2]),  # red-ish
    np.array([0.2, 0.8, 0.2]),  # green-ish
    np.array([0.2, 0.3, 0.9])   # blue-ish
]
third = W // 3
img[:, :third, :] = reflectances[0]
img[:, third:2*third, :] = reflectances[1]
img[:, 2*third:, :] = reflectances[2]

# Illumination field: multiplicative shading that varies smoothly left→right
x = np.linspace(0, 1, W, dtype=np.float32)
illum = 0.5 + 0.5 * (0.6 + 0.4*np.cos(2*np.pi*(x*0.8)))  # in [~0.1, 1]
illum = illum[None, :, None]  # shape (1, W, 1)
toy_rgb = img * illum  # apply shading equally to all channels

toy_rgb = np.clip(toy_rgb, 0.0, 1.0)

# ---- 2) Convert to log-chromaticity: [log(B/R), log(G/R)] ----
eps = 1e-8
R = toy_rgb[..., 0] + eps
G = toy_rgb[..., 1] + eps
B = toy_rgb[..., 2] + eps

logBR = np.log(B / R)
logGR = np.log(G / R)
X = np.stack([logBR.ravel(), logGR.ravel()], axis=1)

# ---- 3) Find lighting-change direction (PCA) and project to invariant axis ----
Xc = X - X.mean(axis=0, keepdims=True)
C = (Xc.T @ Xc) / (Xc.shape[0] - 1)
eigvals, eigvecs = np.linalg.eigh(C)
d = eigvecs[:, np.argmax(eigvals)] / np.linalg.norm(eigvecs[:, np.argmax(eigvals)])
n = np.array([-d[1], d[0]]) / np.linalg.norm([-d[1], d[0]])

y = (X @ n).reshape(H, W)

# Normalize to [0,1] for display
y_min, y_max = np.percentile(y, [1, 99])
y_disp = np.clip((y - y_min) / (y_max - y_min + 1e-12), 0.0, 1.0)

# ---- 4) Show and save as Figure 3 ----
fig = plt.figure(figsize=(9, 4.2))

ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(toy_rgb)
ax1.set_title("Figure 3A: Original RGB (with shadows)")
ax1.axis("off")

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(y_disp, cmap="gray", vmin=0.0, vmax=1.0)
ax2.set_title("Figure 3B: Invariant greyscale (projection onto n)")
ax2.axis("off")

plt.tight_layout()
plt.savefig("Figure3_invariant_demo.png", dpi=180)
plt.show()
