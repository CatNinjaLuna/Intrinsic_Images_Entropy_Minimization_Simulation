import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_lines = 5  # number of parallel lines (different surface reflectances)
points_per_line = 50  # pixels per line (illumination variations)
# Different illumination changes shift points along the same line in this space
line_slope = 0.5  # slope of parallel lines in log-chromaticity space

# Generate points along parallel lines
x_vals = np.linspace(0, 5, points_per_line) # simulates a range of illumination conditions
lines = []
offsets = np.linspace(0, 4, num_lines)  
'''
- Different lines shifted vertically [differnet materials]
- For each offsets, y_vals = slope * x_vals + offset
- There are 5 parallel lines, each representing 1 material under varying light
'''

for offset in offsets:
    y_vals = line_slope * x_vals + offset
    lines.append(np.vstack([x_vals, y_vals]))

# Combine all points
points = np.hstack(lines)  # shape (2, num_lines * points_per_line)

# Plot original points in log-chromaticity space
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for i in range(num_lines):
    plt.plot(lines[i][0], lines[i][1], '.', label=f'Surface {i+1}')
plt.title('Pixels in Log-Chromaticity Space')
plt.xlabel('log(G/R)')
plt.ylabel('log(B/R)')
plt.grid(True)
plt.legend()

# Calculate projection direction perpendicular to lines --- find invariant projection
# line slope = m => perpendicular slope = -1/m
perp_slope = -1 / line_slope
# Unit vector in projection direction
theta = np.arctan(perp_slope)
proj_dir = np.array([np.cos(theta), np.sin(theta)])

# Project points onto this direction (greyscale invariant image)
projected = proj_dir.dot(points)
'''
All 2D points are collapse onto this 1D axis
This simulates a grayscale intrinsic image where illumination has been factored out.
'''

# Plot projection direction on original plot
origin = np.array([0,0])
plt.arrow(origin[0], origin[1], 2*proj_dir[0], 2*proj_dir[1], 
          color='black', width=0.05, label='Projection Direction')
plt.legend()

# Plot histogram of projected values (greyscale image)
plt.subplot(1,2,2)
plt.hist(projected, bins=30, color='gray')
plt.title('Projection onto Invariant Direction (Greyscale)')
plt.xlabel('Projected Intensity')
plt.ylabel('Number of Pixels')
plt.grid(True)

plt.tight_layout()
plt.savefig('Figure 1')
plt.show()
