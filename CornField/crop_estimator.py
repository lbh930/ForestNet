#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix for laspy >= 2.0 (ScaledArrayView issue):
- las.x, las.y, las.z are ScaledArrayView; use np.array() instead of .astype().
"""

import argparse, sys, os, time, math
import numpy as np
import matplotlib.pyplot as plt

try:
    import tqdm
    TQDM = True
except Exception:
    TQDM = False

try:
    from scipy.ndimage import gaussian_filter
    SCIPY = True
except Exception:
    SCIPY = False

try:
    from skimage.filters import frangi, threshold_otsu
    from skimage.morphology import skeletonize, remove_small_objects
    from skimage.exposure import rescale_intensity
    SKIMAGE = True
except Exception:
    SKIMAGE = False

try:
    import laspy
    LASPY = True
except Exception:
    LASPY = False


def log(msg):
    print(msg, flush=True)

def progress(iterable, desc=""):
    if TQDM:
        return tqdm.tqdm(iterable, desc=desc)
    return iterable

class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time()
        log(f"[start] {self.name}")
        return self
    def __exit__(self, exc_type, exc, tb):
        log(f"[done ] {self.name} ({time.time()-self.t0:.2f}s)")


def read_las_points(path):
    if not LASPY:
        log("laspy missing: pip install laspy")
        sys.exit(1)
    with Timer("Load LAS/LAZ"):
        las = laspy.read(path)
        # fix for laspy >= 2.0: ScaledArrayView to numpy
        x = np.array(las.x, dtype=np.float64)
        y = np.array(las.y, dtype=np.float64)
        z = np.array(las.z, dtype=np.float64)
    log(f"[info] points: {len(x):,}")
    return x, y, z


def remove_outliers(x, y, z, grid_res=0.05, min_points_per_cell=1):
    with Timer("Outlier removal"):
        q1, q3 = np.percentile(z, [25, 75])
        iqr = q3 - q1
        mask = (z > q1-3*iqr) & (z < q3+3*iqr)
        x, y, z = x[mask], y[mask], z[mask]
        return x, y, z


def remove_ground_threshold(x, y, z, thresh=0.3):
    with Timer("Ground removal"):
        base = z.min()
        zcut = base + thresh
        keep = z > zcut
        log(f"[info] ground z-thresh ~ {zcut:.3f} m; kept {keep.sum():,}/{len(z):,}")
        return x[keep], y[keep], z[keep]


def build_chm(x, y, z, res=0.05, sigma=1.0):
    with Timer("Build CHM"):
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        nx = int((xmax - xmin)/res)+1
        ny = int((ymax - ymin)/res)+1
        ix = np.floor((x - xmin)/res).astype(int)
        iy = np.floor((y - ymin)/res).astype(int)
        chm = np.full((ny, nx), np.nan, np.float32)
        for i in progress(range(len(x)), desc="CHM"):
            gx, gy = ix[i], iy[i]
            if 0 <= gx < nx and 0 <= gy < ny:
                if np.isnan(chm[gy,gx]) or z[i] > chm[gy,gx]:
                    chm[gy,gx] = z[i]
        chm -= np.nanmin(chm)
    if SCIPY:
        chm = gaussian_filter(np.nan_to_num(chm, nan=0), sigma=sigma)
    return chm, dict(xmin=xmin,ymin=ymin,res=res)


def enhance_and_centerlines(chm):
    if not SKIMAGE:
        log("scikit-image missing")
        return None
    from skimage.filters import frangi, threshold_otsu
    from skimage.morphology import skeletonize, remove_small_objects
    from skimage.exposure import rescale_intensity

    with Timer("Enhance ridges"):
        a = np.nan_to_num(chm, nan=0.0)
        a = rescale_intensity(a, in_range='image', out_range=(0,1))
        v = frangi(a)
        v = rescale_intensity(v, in_range='image', out_range=(0,1))
    with Timer("Skeletonize"):
        thr = threshold_otsu(v[v>0]) if np.any(v>0) else 0.0
        bw = v > thr
        bw = remove_small_objects(bw, 64)
        skel = skeletonize(bw)
        skel = remove_small_objects(skel, 64)
    return skel


def overlay(chm, skel, out):
    with Timer("Render"):
        plt.figure(figsize=(12,8),dpi=160)
        plt.imshow(np.nan_to_num(chm,nan=0), cmap='viridis', origin='lower')
        if skel is not None:
            y,x = np.nonzero(skel)
            plt.scatter(x,y,s=0.2,c='white',marker='.')
        plt.axis('off')
        plt.savefig(out,bbox_inches='tight',pad_inches=0)
        plt.close()
    log(f"[save] {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('las')
    p.add_argument('--res',type=float,default=0.05)
    p.add_argument('--ground_thresh',type=float,default=0.3)
    p.add_argument('--out',default='crop_rows_overlay.png')
    a = p.parse_args()
    if not os.path.exists(a.las):
        log(f"missing file {a.las}"); sys.exit(1)

    x,y,z = read_las_points(a.las)
    x,y,z = remove_outliers(x,y,z)
    x,y,z = remove_ground_threshold(x,y,z,a.ground_thresh)
    chm,meta = build_chm(x,y,z,res=a.res)
    skel = enhance_and_centerlines(chm)
    overlay(chm, skel, a.out)

if __name__ == '__main__':
    main()