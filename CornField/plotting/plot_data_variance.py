import matplotlib.pyplot as plt
import numpy as np
import math

# =========================
# 可修改数据区域
# =========================
fields = [
    "Field A Corn",
    "Field B Corn",
    "Field C Corn",
    "Field D Soybean",
    "Field E Soybean",
]

density = [7.792, 7.258, 10.3, 18.615, 19.592]
height = [2.483, 2.06, 2.41, 0.674, 0.753]
lai = [3.80522, 1.58354, None, 3.8501, 0.6618]
leaf_width = [7.833, 5.18, None, 5.5, 4.3]
leaf_length = [62.667, 51.9, None, 8.5, 6.0]
leaf_count = [12.667, 10.333, None, 56.33, 16.67]

# consistent color palette
colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]

font_size = 40

# =========================
# 图1：Density vs Height
# =========================
fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)

# 手动设置文字偏移，避免重叠
text_offsets = [
    (0.3, 0.05),   
    (0.3, -0.1),   
    (0.3, -0.05),
    (0.3, -0.05),
    (0.3, 0.05),
]

for i in range(len(fields)):
    ax.scatter(density[i], height[i], color=colors[i], s=600)
    ax.text(density[i] + text_offsets[i][0],
            height[i] + text_offsets[i][1],
            fields[i], fontsize=font_size - 4, va='center')

# 计算边距
x_margin = (max(density) - min(density)) * 0.20
y_margin = (max(height) - min(height)) * 0.20

# ❗ 修改：不从 0 开始，而是围绕数据范围
ax.set_xlim(min(density) - x_margin, max(density) + x_margin)
ax.set_ylim(min(height) - y_margin, max(height) + y_margin)

ax.set_xlabel("Density (plants/m²)", fontsize=font_size)
ax.set_ylabel("Average Height (m)", fontsize=font_size)
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

fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)

text_offsets_leaf = [
    (15, 10),    
    (15, -15),   
    (15, 10),    
    (15, 10),    
]

for i in range(len(field_valid)):
    ax.scatter(leaf_area[i], leaf_density[i], color=colors[i], s=600)
    ax.text(leaf_area[i] + text_offsets_leaf[i][0],
            leaf_density[i] + text_offsets_leaf[i][1],
            field_valid[i], fontsize=font_size - 4, va='center')

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

fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)

bars = ax.bar(field_lai, lai_valid, color=colors[:len(field_lai)])
ax.set_xticklabels(field_lai, rotation=30, ha='right', fontsize=font_size*0.75)

ax.set_ylabel("LAI (m²/m²)", fontsize=font_size)
ax.set_ylim(bottom=0)
ax.tick_params(axis='y', labelsize=font_size)

plt.savefig("bar_lai.png", dpi=300)
plt.close()

print("✅ All three plots saved successfully!")
