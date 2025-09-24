# intrinsic_entropy_reintegrate.py
import os
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite

EPS = 1e-8

# ---------- sRGB <-> linear ----------
def srgb_to_linear(img):
    """Convert uint8 sRGB or float sRGB in [0,1] to linear RGB float32."""
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    a = 0.055
    return np.where(img <= 0.04045, img / 12.92, ((img + a) / (1 + a)) ** 2.4)

def linear_to_srgb(img):
    """Convert linear RGB float to sRGB float in [0,1]."""
    a = 0.055
    out = np.where(img <= 0.0031308, 12.92 * img, (1 + a) * np.power(img, 1/2.4) - a)
    return np.clip(out, 0.0, 1.0)

# ---------- Geometric-mean chromaticities ----------
def geometric_mean_chroma(rgb_lin):
    """c_k = R_k / (R G B)^(1/3)."""
    R, G, B = rgb_lin[..., 0], rgb_lin[..., 1], rgb_lin[..., 2]
    RM = np.cbrt(np.maximum(R * G * B, EPS))
    c = np.stack([R / (RM + EPS), G / (RM + EPS), B / (RM + EPS)], axis=-1)
    return np.clip(c, EPS, None)

def log_chroma3(c):
    """ρ = log(c) in R^3."""
    return np.log(np.clip(c, EPS, None))

def plane_basis_U():
    """
    Return a 2x3 orthonormal basis U (rows) spanning the plane ⟂ u=(1,1,1)/√3.
    With rho in R^3, chi = rho @ U^T -> R^2.
    """
    u = np.array([1.0, 1.0, 1.0], dtype=np.float64); u /= np.linalg.norm(u)
    a = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    a = a - u * (a @ u)
    e1 = a / (np.linalg.norm(a) + EPS)
    e2 = np.cross(u, e1); e2 /= (np.linalg.norm(e2) + EPS)
    return np.stack([e1, e2], axis=0)  # (2,3)

# ---------- Entropy sweep ----------
def project_entropy_min_angle(chi, step_deg=0.25, clip_quantiles=(0.01, 0.99), bins='scott'):
    """Sweep θ to find min-entropy projection I = χ1 cosθ + χ2 sinθ."""
    thetas = np.arange(0.0, 180.0, step_deg, dtype=np.float64)
    x, y = chi[..., 0], chi[..., 1]
    ent = []

    def entropy_1d(vals):
        v = vals.reshape(-1)
        if clip_quantiles is not None:
            lo, hi = np.quantile(v, clip_quantiles)
            v = v[(v >= lo) & (v <= hi)]
        if isinstance(bins, int):
            nbins = bins
        else:
            std = np.std(v); N = max(1, v.size)
            bw = 3.5 * std * (N ** (-1/3))
            if bw < 1e-8: nbins = 64
            else:
                nbins = int(np.clip(np.ceil((v.max() - v.min()) / (bw + 1e-12)), 16, 512))
        hist, _ = np.histogram(v, bins=nbins)
        p = hist.astype(np.float64); p = p[p > 0]; p /= (p.sum() + EPS)
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

# ---------- Re-integration with canonical illumination intercept ----------
def reintegrate_color_from_invariant_chroma(rgb_lin, U, rho, theta_star,
                                            intensity_ref='p99', bright_pct=95):
    """
    Re-integrate to color using:
      s : invariant coord along v* (min-entropy direction)
      t0: canonical global intercept along w* (illumination axis, ⟂ v*)
    """
    chi = np.tensordot(rho, U.T, axes=([2], [0]))  # (H,W,2)

    ct, st = np.cos(np.deg2rad(theta_star)), np.sin(np.deg2rad(theta_star))
    v = np.array([ct, st], dtype=np.float64)
    w = np.array([-st, ct], dtype=np.float64)

    s = chi[..., 0] * v[0] + chi[..., 1] * v[1]
    t = chi[..., 0] * w[0] + chi[..., 1] * w[1]

    I_orig = np.sum(rgb_lin, axis=-1)
    thr = np.percentile(I_orig, bright_pct)
    mask = I_orig >= thr
    t0 = np.median(t[mask]) if np.any(mask) else np.median(t)

    chi_tilde = np.stack([s * v[0] + t0 * w[0], s * v[1] + t0 * w[1]], axis=-1)

    rho_tilde = np.tensordot(chi_tilde, U, axes=([2], [0]))     # (H,W,3)
    c_tilde = np.exp(rho_tilde)
    sumc = np.sum(c_tilde, axis=-1, keepdims=True) + EPS
    r_tilde = c_tilde / sumc

    # Small gray-world normalization
    g = np.mean(r_tilde.reshape(-1, 3), axis=0) + EPS
    r_tilde = r_tilde / g
    r_tilde = np.clip(r_tilde / (np.sum(r_tilde, axis=-1, keepdims=True) + EPS), 0, 1)

    if intensity_ref == 'p99':
        I_ref = np.percentile(I_orig, 99.0)
    elif isinstance(intensity_ref, (int, float)):
        I_ref = float(intensity_ref)
    else:
        I_ref = np.median(I_orig)
    I_ref = max(I_ref, 1e-3)

    rgb_relit_lin = np.clip(r_tilde * I_ref, 0.0, 1e6)
    return rgb_relit_lin, r_tilde, s

# ---------- Main ----------
def main():
    # ---- EDIT THIS PATH ----
    input_path = "input_image.jpg"
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

    # 3) Geometric-mean chroma -> ρ, then χ = rho @ U^T
    c = geometric_mean_chroma(rgb_lin)
    rho = log_chroma3(c)
    U = plane_basis_U()
    chi = np.tensordot(rho, U.T, axes=([2], [0]))  # (H,W,2)

    # 4) Find θ* via entropy minimization
    theta_star, thetas, entropies = project_entropy_min_angle(chi, step_deg=0.25)
    print(f"[INFO] θ* (min entropy) = {theta_star:.4f} degrees")

    # 5) Intrinsic grayscale (projection at θ*)
    intrinsic_scalar = chi[..., 0] * np.cos(np.deg2rad(theta_star)) + \
                       chi[..., 1] * np.sin(np.deg2rad(theta_star))
    intrinsic_gray = normalize01(intrinsic_scalar)

    # 6) Reintegrate to shadow-free color
    rgb_relit_lin, r_tilde, s_vals = reintegrate_color_from_invariant_chroma(
        rgb_lin, U, rho, theta_star, intensity_ref='p99', bright_pct=95
    )
    rgb_relit = linear_to_srgb(np.clip(rgb_relit_lin, 0.0, 1.0))

    # 7) Save outputs (images)
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_A = os.path.join(outdir, f"{base}_figA_original.png")
    out_B = os.path.join(outdir, f"{base}_figB_intrinsic_gray.png")
    out_C = os.path.join(outdir, f"{base}_figC_entropy_curve.png")
    out_D = os.path.join(outdir, f"{base}_figD_shadow_free_color.png")
    out_panel = os.path.join(outdir, f"{base}_figABCD_panel.png")

    imwrite(out_A, rgb if rgb.dtype == np.uint8 else (np.clip(rgb,0,1)*255).astype(np.uint8))
    imwrite(out_B, (intrinsic_gray * 255).astype(np.uint8))
    imwrite(out_D, (rgb_relit * 255).astype(np.uint8))

    # 8) Entropy curve with θ* annotation (this is your “angle ↔ min entropy” figure)
    plt.figure(figsize=(8, 4))
    plt.plot(thetas, entropies, linewidth=2)
    plt.axvline(theta_star, linestyle="--")
    ymin, ymax = np.min(entropies), np.max(entropies)
    plt.text(theta_star + 1, ymin + 0.05*(ymax - ymin), f"θ* = {theta_star:.2f}°")
    plt.xlabel("Projection angle θ (degrees)")
    plt.ylabel("Entropy H(θ) [bits]")
    plt.title("Entropy vs θ (min annotated)")
    plt.tight_layout()
    plt.savefig(out_C, dpi=200); plt.close()

    # 9) Optional 4-up panel (A–D)
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(2, 2, 1); ax1.imshow(rgb); ax1.set_title("A. Original RGB"); ax1.axis("off")
    ax2 = fig.add_subplot(2, 2, 2); ax2.imshow(intrinsic_gray, cmap="gray", vmin=0, vmax=1); ax2.set_title(f"B. Intrinsic grayscale (θ*={theta_star:.2f}°)"); ax2.axis("off")
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(thetas, entropies); ax3.axvline(theta_star, linestyle="--")
    ax3.set_title("C. Entropy vs θ"); ax3.set_xlabel("θ (degrees)"); ax3.set_ylabel("H(θ) [bits]")
    ax4 = fig.add_subplot(2, 2, 4); ax4.imshow(rgb_relit); ax4.set_title("D. Shadow-free color"); ax4.axis("off")
    plt.tight_layout(); plt.savefig(out_panel, dpi=220); plt.show()

    print("[SAVED]")
    print("  A:", out_A)
    print("  B:", out_B)
    print("  C:", out_C, " (annotated with θ*)")
    print("  D:", out_D)
    print("Panel:", out_panel)

if __name__ == "__main__":
    main()
