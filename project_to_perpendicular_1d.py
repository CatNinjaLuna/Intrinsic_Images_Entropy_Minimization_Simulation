import numpy as np
import matplotlib.pyplot as plt

# --- Data generator: several parallel lines in 2D ---
def generate_parallel_lines(num_lines=5, pts_per_line=100, angle_deg=25,
                            along_noise=0.05, isotropic_noise=0.03, seed=0):
    """
    Returns:
        points: (N,2) array
        labels: (N,) line index for each point
        d: unit vector for the common line direction
        n: unit vector for the perpendicular (invariant) direction
    """
    rng = np.random.default_rng(seed)

    # Common line direction d (illumination-change direction)
    theta = np.deg2rad(angle_deg)
    d = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    d /= np.linalg.norm(d)

    # Perpendicular direction n (invariant direction)
    n = np.array([-d[1], d[0]], dtype=float)
    n /= np.linalg.norm(n)

    # Place lines at different offsets along n
    offsets = np.linspace(-2, 2, num_lines)

    pts = []
    labels = []
    for i, off in enumerate(offsets):
        base = off * n
        # Parameter t runs along the line direction d
        t = np.linspace(-3, 3, pts_per_line) + rng.normal(0, along_noise, pts_per_line)
        line_pts = base + np.outer(t, d)
        # Small isotropic measurement noise
        line_pts += rng.normal(0, isotropic_noise, line_pts.shape)
        pts.append(line_pts)
        labels.extend([i] * pts_per_line)

    points = np.vstack(pts)
    labels = np.array(labels)
    return points, labels, d, n

# --- Projection to 1-D along n ---
def project_to_invariant_1d(points, n):
    """Project 2-D points onto the invariant direction n to get 1-D values."""
    n = np.asarray(n, dtype=float)
    n /= np.linalg.norm(n)
    return points @ n

if __name__ == "__main__":
    # 1) Generate synthetic data
    points, labels, d, n = generate_parallel_lines()

    # 2) Project onto the invariant direction (perpendicular to the lines)
    y = project_to_invariant_1d(points, n)

    # 3) Figure 2A: Parallel lines + invariant directions
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], c=labels, s=8, cmap="tab10")
    scale = 2.5
    plt.arrow(0, 0, d[0]*scale, d[1]*scale, head_width=0.12,
              length_includes_head=True, color="red", label="Line direction d")
    plt.arrow(0, 0, n[0]*scale, n[1]*scale, head_width=0.12,
              length_includes_head=True, color="blue", label="Invariant direction n")
    plt.title("Figure 2A: Parallel lines in 2D with invariant directions")
    plt.xlabel("log(B/R)")
    plt.ylabel("log(G/R)")

    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig("Figure2A_parallel_lines_invariant_directions.png", dpi=180)
    plt.show()

    # 4) Figure 2B: Histogram of 1-D projected values
    plt.figure(figsize=(7, 4))
    plt.hist(y, bins=60, color="gray")
    plt.title("Figure 2B: Projected 1D values (lines collapse to peaks)")
    plt.xlabel("Projection onto n")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("Figure2B_projected_1D_values.png", dpi=180)
    plt.show()
