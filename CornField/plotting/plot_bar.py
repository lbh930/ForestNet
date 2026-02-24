import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 50,
    "axes.labelsize": 50,
    "xtick.labelsize": 50,
    "ytick.labelsize": 50,
    "legend.fontsize": 50
})

# =====================================================================
# ★★★★★ 你只需要改这里的数据 ★★★★★
# =====================================================================

# ---- LEAF AREA INDEX ----
lai_data = {
    "Field D\nSoybean": {
        "GT": 3.8501,
        "Est": 3.281,
    },
    "Field E\nSoybean": {
        "GT": 0.6618,
        "Est": 0.740,
    },
    "Field A\nCorn": {
        "GT": 3.80522,
        "Est": 3.281,
    },
    "Field B\nCorn": {
        "GT": 1.58354,
        "Est": 1.876,
    },
}

# ---- DENSITY ----
density_data = {
    "Field A\nCorn":       {"GT": 7.792,  "Est": 8.287},
    "Field B\nCorn":      {"GT": 7.258,  "Est": 6.62},
    "Field C\nCorn":             {"GT": 10.3,   "Est": 8.355},
    "Field D\nSoybean":    {"GT": 18.615, "Est": 18.496},
    "Field E\nSoybean":   {"GT": 19.592, "Est": 18.645},
}

# ---- HEIGHT ----
height_data = {
    "Field A\nCorn":       {"GT": 2.483, "Est": 2.91},
    "Field B\nCorn":      {"GT": 2.06,  "Est": 1.979},
    "Field C\nCorn":             {"GT": 2.41,  "Est": 2.563},
    "Field D\nSoybean":    {"GT": 0.674, "Est": 0.724},
    "Field E\nSoybean":   {"GT": 0.753, "Est": 0.845},
}

# =====================================================================
# ★★★★★ Bar Plot 绘图函数 ★★★★★
# =====================================================================

def plot_bar(data_dict, ylabel, output_filename):
    labels = list(data_dict.keys())
    gt_raw  = [data_dict[k]["GT"] for k in labels]
    est_raw = [data_dict[k]["Est"] for k in labels]

    def to_float_list(vals):
        out = []
        for v in vals:
            if v is None:
                out.append(np.nan)
                continue
            # 将 'None' / '' / 非法字符串 转为 NaN
            if isinstance(v, str):
                v_strip = v.strip().lower()
                if v_strip in ("none", "nan", ""):
                    out.append(np.nan)
                    continue
            try:
                out.append(float(v))
            except Exception:
                out.append(np.nan)
        return out

    gt_vals  = to_float_list(gt_raw)
    est_vals = to_float_list(est_raw)

    x = np.arange(len(labels))
    width = 0.33

    fig, ax = plt.subplots(figsize=(24, 10))

    # Matplotlib 对 None 会报错，这里已转换为 NaN；含 NaN 的柱会被跳过绘制
    ax.bar(x - width/2 - width/2, gt_vals,  width=width, label="GT", color="#DDAA33")
    ax.bar(x, est_vals, width=width, label="Est",    color="#004488")
    
    ax.set_ylabel(ylabel, fontsize=45)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="right")

    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    print(f"Saved: {output_filename}")


# =====================================================================
# ★★★★★ 生成三张图 ★★★★★
# =====================================================================

plot_bar(density_data, "Density (plants/m²)", "placeholder_density_barplot.png")
plot_bar(height_data,  "Average Height (m)",  "placeholder_height_barplot.png")
plot_bar(lai_data,     "Leaf Area Index (m²/m²)",         "placeholder_lai_barplot.png")
