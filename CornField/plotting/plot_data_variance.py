import matplotlib.pyplot as plt
import numpy as np
import math

# =========================
# 可修改数据区域
# =========================
fields = [
    "Field A\nCorn",
    "Field B\nCorn",
    "Field C\nCorn",
    "Field D\nSoybean",
    "Field E\nSoybean",
]

density = [7.792, 7.258, 10.3, 18.615, 19.592]
height = [2.483, 2.06, 2.41, 0.674, 0.753]
lai = [3.80522, 1.58354, None, 3.8501, 0.6618]
leaf_width = [7.833, 5.18, None, 5.5, 4.3]
leaf_length = [62.667, 51.9, None, 8.5, 6.0]
leaf_count = [12.667, 10.333, None, 56.33, 16.67]

# consistent color palette
colors = ["#FFFFFF", "#DDAA33", "#BB5566", "#004488", "#000000"]
  # Set1 颜色方案

font_size = 54

# =========================
# 图1：Density vs Height
# =========================
fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)

# 手动设置文字偏移，避免重叠
text_offsets = [
    (0.3, 0.05),   
    (0.3, -0.1),   
    (0.3, -0.05),
    (-0.5, -0.1),
    (-0.5, 0.1),
]

for i in range(len(fields)):
    ax.scatter(density[i], height[i], color=colors[i], s=800, linewidths=2, edgecolors='black')
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=2)
    # Put Field D & E (indices 3,4) labels to the left of their points
    ha = 'right' if i in (3, 4) else 'left'
    ax.text(density[i] + text_offsets[i][0],
            height[i] + text_offsets[i][1],
            fields[i], fontsize=font_size - 4, va='center', ha=ha)

# 计算边距
x_margin = (max(density) - min(density)) * 0.20
y_margin = (max(height) - min(height)) * 0.20

# ❗ 修改：不从 0 开始，而是围绕数据范围
ax.set_xlim(min(density) - x_margin, max(density) + x_margin)
ax.set_ylim(min(height) - y_margin, max(height) + y_margin)

ax.set_xlabel("Avg Canopy Density \n (plants/m²)", fontsize=font_size)
ax.set_ylabel("Avg Canopy Height (m)", fontsize=font_size)
ax.tick_params(axis='both', labelsize=font_size)

plt.savefig("scatter_density_height.png", dpi=300)
plt.close()

# =========================
# 图2：Leaf Area vs Leaf Density（不改）
# =========================
leaf_area = []
leaf_density = []
field_valid = []

for i in range(len(fields)):
    if leaf_width[i] is not None and leaf_length[i] is not None:
        area = math.pi * leaf_width[i] * leaf_length[i] / 4
        leaf_area.append(area)
        leaf_density.append(density[i] * leaf_count[i])
        field_valid.append(fields[i])

fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)

text_offsets_leaf = [
    (-15, 35),    # Field A Corn (left side)
    (-15, -20),   # Field B Corn (left side)
    (15, 10),     # Field D Soybean (default right side)
    (15, 10),     # Field E Soybean (default right side)
]

for i in range(len(field_valid)):
    ax.scatter(leaf_area[i], leaf_density[i], color=colors[i], s=800, linewidths=2, edgecolors='black')
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=2)
    # Left-align labels where x offset is negative (A,B)
    ha = 'right' if text_offsets_leaf[i][0] < 0 else 'left'
    ax.text(leaf_area[i] + text_offsets_leaf[i][0],
            leaf_density[i] + text_offsets_leaf[i][1],
            field_valid[i], fontsize=font_size - 4, va='center', ha=ha)

# 计算边距
x_margin = (max(leaf_area) - min(leaf_area)) * 0.25
y_margin = (max(leaf_density) - min(leaf_density)) * 0.25
ax.set_xlim(0, max(leaf_area) + x_margin)
ax.set_ylim(0, max(leaf_density) + y_margin)

ax.set_xlabel("Single Leaf Area (cm²)", fontsize=font_size)
ax.set_ylabel("Leaf Density (leaves/m²)", fontsize=font_size)
ax.tick_params(axis='both', labelsize=font_size)

plt.savefig("scatter_leafarea_density.png", dpi=300)
plt.close()

# =========================
# 图3：LAI Bar Plot（不改）
# =========================
lai_valid = []
field_lai = []
for i in range(len(fields)):
    if lai[i] is not None:
        field_lai.append(fields[i])
        lai_valid.append(lai[i])

fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)

# First bar black; keep the original 2nd-4th bar colors.
bar_colors = ["#000000", "#DDAA33", "#BB5566", "#004488"][:len(field_lai)]
bars = ax.bar(field_lai, lai_valid, color=bar_colors, edgecolor="black", linewidth=2)

ax.grid(True, linestyle='--', alpha=0.7, linewidth=2)
ax.set_axisbelow(True)

ax.set_xticklabels(field_lai, rotation=35, ha='right', fontsize=font_size*0.8)

ax.set_ylabel("Leaf Area Index (m²/m²)", fontsize=font_size)
ax.set_ylim(bottom=0)
ax.tick_params(axis='y', labelsize=font_size)

plt.savefig("bar_lai.png", dpi=300)
plt.close()

print("✅ All three plots saved successfully!")
