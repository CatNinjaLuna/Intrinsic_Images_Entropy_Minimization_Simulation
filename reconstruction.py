import os
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite

EPS = 1e-8

# ---------- sRGB <-> linear ----------
def srgb_to_linear(img):
    img = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
    a = 0.055
    return np.where(img <= 0.04045, img / 12.92, ((img + a) / (1 + a)) ** 2.4)

def linear_to_srgb(img):
    a = 0.055
    out = np.where(img <= 0.0031308, 12.92 * img, (1 + a) * np.power(img, 1/2.4) - a)
    return np.clip(out, 0.0, 1.0)

# ---------- Paper-friendly chromaticities ----------
def geometric_mean_chroma(rgb_lin):
    """Geometric-mean chromaticity: c_k = R_k / (R G B)^(1/3)."""
    R, G, B = rgb_lin[..., 0], rgb_lin[..., 1], rgb_lin[..., 2]
    RM = np.cbrt(np.maximum(R * G * B, EPS))
    c = np.stack([R / (RM + EPS), G / (RM + EPS), B / (RM + EPS)], axis=-1)
    return np.clip(c, EPS, None)

def log_chroma3(c):
    """ρ = log(c) in R^3; note ρ lies on plane orthogonal to u=(1,1,1)/√3."""
    return np.log(np.clip(c, EPS, None))

def plane_basis_U():
    """Return a 2x3 orthonormal basis U spanning the plane orthogonal to u=(1,1,1)/√3."""
    u = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    u = u / np.linalg.norm(u)

    # Pick any vector not parallel to u, Gram-Schmidt to get e1 ⟂ u
    a = np.array([1.0, -1.0, 0.0])
    a = a - u * (a @ u)
    e1 = a / (np.linalg.norm(a) + EPS)

    # e2 = u × e1
    e2 = np.cross(u, e1)
    e2 = e2 / (np.linalg.norm(e2) + EPS)

    # U maps R^3 -> R^2 (rows are e1^T and e2^T)
    U = np.stack([e1, e2], axis=0)
    return U

def project_entropy_min_angle(chi, step_deg=0.25, clip_quantiles=(0.01, 0.99), bins='scott'):
    """Sweep θ to find min-entropy projection I = χ1 cosθ + χ2 sinθ."""
    thetas = np.arange(0.0, 180.0, step_deg, dtype=np.float64)
    x, y = chi[..., 0], chi[..., 1]
    ent = []

    # Entropy helper with Scott's rule (as in the paper) unless fixed bins passed
    def entropy_1d(vals):
        v = vals.reshape(-1)
        if clip_quantiles is not None:
            lo, hi = np.quantile(v, clip_quantiles)
            v = v[(v >= lo) & (v <= hi)]
        if isinstance(bins, int):
            nbins = bins
        else:
            # Scott's rule bandwidth → bins from data range
            std = np.std(v)
            N = max(1, v.size)
            bw = 3.5 * std * (N ** (-1/3))
            if bw < 1e-6:  # degenerate
                nbins = 64
            else:
                nbins = int(np.clip(np.ceil((v.max() - v.min()) / bw), 16, 512))
        hist, _ = np.histogram(v, bins=nbins)
        p = hist.astype(np.float64)
        p = p[p > 0]
        p /= (p.sum() + EPS)
        return -np.sum(p * np.log2(p + 1e-24))

    for th in thetas:
        ct, st = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
        I = ct * x + st * y
        ent.append(entropy_1d(I))

    ent = np.array(ent)
    theta_star = thetas[np.argmin(ent)]
    return theta_star, thetas, ent

def normalize01(img, clip_quantiles=(0.01, 0.99)):
    v = img.reshape(-1)
    lo, hi = np.quantile(v, clip_quantiles)
    out = (img - lo) / (hi - lo + 1e-12)
    return np.clip(out, 0.0, 1.0)

# ---------- Re-integration to full color ----------
def reintegrate_color_from_invariant_chroma(rgb_lin, U, rho, theta_star, intensity_ref='p99'):
    """
    Build a shadow-free *color* image by:
      1) Projecting onto the min-entropy direction in the χ-plane,
      2) Mapping back to ρ̃ -> c̃ = exp(ρ̃),
      3) Converting to L1 chromaticity r̃ (sum=1),
      4) Multiplying by a clean global intensity to "relight" (flat lighting).
    """
    # 1) χ = U ρ
    chi = np.tensordot(rho, U.T, axes=([2], [1]))  # (H,W,2)

    # 2) Projection and back-projection in χ-plane
    ct, st = np.cos(np.deg2rad(theta_star)), np.sin(np.deg2rad(theta_star))
    n = np.array([ct, st], dtype=np.float64)  # unit direction in χ-plane
    I = chi[..., 0] * ct + chi[..., 1] * st
    chi_proj = np.stack([I * n[0], I * n[1]], axis=-1)  # (H,W,2)

    # 3) Back to 3D ρ̃ and c̃
    rho_tilde = np.tensordot(chi_proj, U, axes=([2], [0]))  # (H,W,3)
    c_tilde = np.exp(rho_tilde)  # geometric-mean chroma ~ reflectance ratios

    # 4) Convert to L1 chromaticity r̃,g̃,b̃ (bounded & well-behaved)
    sumc = np.sum(c_tilde, axis=-1, keepdims=True) + EPS
    r_tilde = c_tilde / sumc  # (H,W,3), sum=1

    # 5) Choose a shadow-free "flat" intensity to relight
    I_orig = np.sum(rgb_lin, axis=-1)  # original intensity (contains shadows)
    if intensity_ref == 'p99':
        I_ref = np.percentile(I_orig, 99.0)
    elif isinstance(intensity_ref, (int, float)):
        I_ref = float(intensity_ref)
    else:
        I_ref = np.median(I_orig)
    I_ref = max(I_ref, 1e-3)

    # 6) Compose color image with invariant chroma and clean intensity
    rgb_relit_lin = r_tilde * I_ref  # flat lighting → no shadows
    rgb_relit_lin = np.clip(rgb_relit_lin, 0.0, 1e6)

    return rgb_relit_lin, r_tilde, I

def main():
    input_path = "input_image.jpg"  # <-- your image path
    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)

    # 1) Load
    rgb = imread(input_path)
    if rgb.ndim == 2 or rgb.shape[-1] not in (3, 4):
        raise ValueError("Input must be RGB or RGBA.")
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    # 2) Linearize
    rgb_lin = srgb_to_linear(rgb)

    # 3) Geometric-mean chroma -> ρ, then χ = U ρ
    c = geometric_mean_chroma(rgb_lin)
    rho = log_chroma3(c)
    U = plane_basis_U()
    chi = np.tensordot(rho, U.T, axes=([2], [1]))  # (H,W,2)

    # 4) Find θ* via entropy minimization (Scott’s rule + 1–99% clipping)
    theta_star, thetas, entropies = project_entropy_min_angle(chi, step_deg=0.25)

    print(f"[INFO] θ* (min entropy) = {theta_star:.4f} degrees")

    # 5) Intrinsic grayscale (for display): projection value I normalized
    intrinsic_gray = normalize01(chi[..., 0] * np.cos(np.deg2rad(theta_star)) +
                                 chi[..., 1] * np.sin(np.deg2rad(theta_star)))

    # 6) Reintegrate to shadow-free color
    rgb_relit_lin, r_tilde, I_proj = reintegrate_color_from_invariant_chroma(
        rgb_lin, U, rho, theta_star, intensity_ref='p99'
    )
    rgb_relit = linear_to_srgb(np.clip(rgb_relit_lin, 0.0, 1.0))

    # 7) Save outputs
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_gray = os.path.join(outdir, f"{base}_intrinsic_gray.png")
    out_curve = os.path.join(outdir, f"{base}_entropy_curve.png")
    out_chroma = os.path.join(outdir, f"{base}_invariant_chroma.png")
    out_color = os.path.join(outdir, f"{base}_shadow_free_color.png")

    imwrite(out_gray, (intrinsic_gray * 255).astype(np.uint8))
    print(f"[INFO] Saved intrinsic grayscale → {out_gray}")

    # Entropy curve
    plt.figure(figsize=(7, 4))
    plt.plot(thetas, entropies)
    plt.xlabel("Projection angle θ (degrees)")
    plt.ylabel("Entropy H(θ) [bits]")
    plt.title(f"Entropy vs θ — min at {theta_star:.2f}°")
    plt.tight_layout()
    plt.savefig(out_curve, dpi=180)
    print(f"[INFO] Saved entropy curve → {out_curve}")

    # Invariant L1 chromaticity visualization
    inv_chroma_vis = np.clip(r_tilde / (r_tilde.max(axis=(0,1)) + EPS), 0, 1)
    imwrite(out_chroma, (linear_to_srgb(inv_chroma_vis) * 255).astype(np.uint8))
    print(f"[INFO] Saved invariant chromaticity (viz) → {out_chroma}")

    # Shadow-free color
    imwrite(out_color, (rgb_relit * 255).astype(np.uint8))
    print(f"[INFO] Saved shadow-free *color* image → {out_color}")

    # Preview panel
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(rgb)
    ax1.set_title("Original RGB"); ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(intrinsic_gray, cmap="gray", vmin=0.0, vmax=1.0)
    ax2.set_title(f"Intrinsic grayscale (θ*={theta_star:.2f}°)"); ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(rgb_relit)
    ax3.set_title("Shadow-free color (flat relight)"); ax3.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
