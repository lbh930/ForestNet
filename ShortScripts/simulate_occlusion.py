import argparse
import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import laspy
import numpy as np

# ------------------------
# parameters
# ------------------------
CELL_SIZE = 2.0       # heightmap grid size (m)


_KNOWN_LABEL_DIM_NAMES = (
    "label",
    "labels",
    "tree_id",
    "treeid",
    "treeID",
    "instance",
    "instance_id",
    "instanceid",
)


def _find_sidecar_labels_file(input_las_path: str) -> Optional[str]:
    p = Path(input_las_path)
    candidates = [
        p.with_name(f"{p.stem}_labels.npy"),
        p.with_name(f"{p.stem}_labels.txt"),
        p.with_name(f"{p.stem}_label.npy"),
        p.with_name(f"{p.stem}_label.txt"),
        p.with_suffix(".npy"),
        p.with_suffix(".txt"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _load_labels_file(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load labels file.

    Supported formats:
    - .npy: array with shape (4, N) or (N, 4)
    - .txt: space-delimited with shape (4, N) or (N, 4)

    Convention:
      First three rows/cols are x,y,z (meters), last row/col is integer labels.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"labels file not found: {path}")

    if p.suffix.lower() == ".npy":
        arr = np.load(str(p))
    elif p.suffix.lower() in {".txt", ".xyz", ".labels"}:
        arr = np.loadtxt(str(p))
    else:
        raise ValueError(f"unsupported labels file extension: {p.suffix} (expected .npy or .txt)")

    if arr.ndim != 2:
        raise ValueError(f"labels array must be 2D, got shape={arr.shape}")

    if arr.shape[0] == 4:
        x_l, y_l, z_l, labels = arr[0], arr[1], arr[2], arr[3]
    elif arr.shape[1] == 4:
        x_l, y_l, z_l, labels = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    else:
        raise ValueError(
            f"labels array must have one dimension = 4, got shape={arr.shape}"
        )

    labels_i = labels.astype(np.int64, copy=False)
    return (
        np.asarray(x_l, dtype=np.float64),
        np.asarray(y_l, dtype=np.float64),
        np.asarray(z_l, dtype=np.float64),
        labels_i,
    )


def _save_labels_file(
    path: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = np.vstack(
        [
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
            np.asarray(labels, dtype=np.int64),
        ]
    )
    if p.suffix.lower() == ".npy":
        np.save(str(p), out)
    elif p.suffix.lower() == ".txt":
        np.savetxt(str(p), out, fmt="%.8f")
    else:
        raise ValueError(f"unsupported labels output extension: {p.suffix} (use .npy or .txt)")


@dataclass
class ProgressPrinter:
    enabled: bool = True
    every: int = 10_000
    label: str = ""
    total: Optional[int] = None
    _t0: float = 0.0
    _last_t: float = 0.0

    def start(self, total: Optional[int] = None, label: Optional[str] = None) -> None:
        if total is not None:
            self.total = total
        if label is not None:
            self.label = label
        self._t0 = time.time()
        self._last_t = self._t0
        if self.enabled:
            total_str = f"/{self.total}" if self.total is not None else ""
            print(f"[{self.label}] start (0{total_str})")

    def update(self, i: int) -> None:
        if not self.enabled:
            return
        if self.total is None:
            if i % self.every == 0 and i > 0:
                now = time.time()
                dt = now - self._last_t
                elapsed = now - self._t0
                self._last_t = now
                print(f"[{self.label}] {i} items | +{dt:.1f}s | elapsed {elapsed:.1f}s")
            return

        # total known
        if (i % self.every) != 0 and i != self.total:
            return

        now = time.time()
        elapsed = now - self._t0
        frac = min(max(i / max(self.total, 1), 0.0), 1.0)
        eta = (elapsed / frac - elapsed) if frac > 0 else float("inf")
        pct = 100.0 * frac
        eta_str = f"{eta:.1f}s" if np.isfinite(eta) else "?"
        print(f"[{self.label}] {i}/{self.total} ({pct:.1f}%) | elapsed {elapsed:.1f}s | ETA {eta_str}")

    def done(self) -> None:
        if not self.enabled:
            return
        elapsed = time.time() - self._t0
        print(f"[{self.label}] done | elapsed {elapsed:.1f}s")

def _compute_cell_indices(x: np.ndarray, y: np.ndarray, *, cell_size: float):
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0")
    x0 = float(np.min(x))
    y0 = float(np.min(y))
    ix = np.floor((x - x0) / cell_size).astype(np.int32)
    iy = np.floor((y - y0) / cell_size).astype(np.int32)
    return ix, iy, x0, y0


def canopy_height_per_point(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    cell_size: float,
    mode: str = "bilinear",
    max_grid_cells: int = 50_000_000,
    progress: Optional[ProgressPrinter] = None,
):
    """Build a low-resolution top-view heightmap and return canopy Z for each point.

    Heightmap definition: per (x,y) grid cell, canopy height is max(Z) of points in that cell.

    Modes:
    - "nearest": use the cell's max(Z) (piecewise-constant per cell)
    - "bilinear": bilinear interpolation between adjacent cell max(Z) values for smooth transitions
    """
    if progress is None:
        progress = ProgressPrinter(enabled=False)
    progress.label = progress.label or "heightmap"
    progress.start(total=None)

    if mode not in {"nearest", "bilinear"}:
        raise ValueError(f"unsupported heightmap mode: {mode} (expected 'nearest' or 'bilinear')")

    ix, iy, x0, y0 = _compute_cell_indices(x, y, cell_size=cell_size)

    # Packed 64-bit cell key (stable, sortable) without allocating a huge dense grid.
    key = (ix.astype(np.int64) << 32) ^ (iy.astype(np.int64) & np.int64(0xFFFFFFFF))
    order = np.argsort(key)
    key_s = key[order]
    z_s = z[order]

    unique_keys, idx_start = np.unique(key_s, return_index=True)
    cell_max_z = np.maximum.reduceat(z_s, idx_start)

    # Map each point back to its cell max Z (nearest/cell-constant).
    cell_index = np.searchsorted(unique_keys, key)
    canopy_z_nearest = cell_max_z[cell_index]

    if mode == "nearest":
        canopy_z = canopy_z_nearest
    else:
        # Bilinear interpolation between 4 neighboring cell max-Z values.
        nx = int(ix.max()) + 1
        ny = int(iy.max()) + 1
        if nx <= 0 or ny <= 0:
            canopy_z = canopy_z_nearest
        elif (nx * ny) > int(max_grid_cells):
            # Avoid massive dense allocations; fall back to nearest.
            canopy_z = canopy_z_nearest
        else:
            grid = np.full((nx, ny), np.nan, dtype=np.float32)
            ux = (unique_keys >> 32).astype(np.int64)
            uy = (unique_keys & np.int64(0xFFFFFFFF)).astype(np.int64)
            grid[ux, uy] = cell_max_z.astype(np.float32, copy=False)

            gx = (x - x0) / float(cell_size)
            gy = (y - y0) / float(cell_size)
            fx = gx - ix
            fy = gy - iy

            ix0 = ix
            iy0 = iy
            ix1 = np.minimum(ix0 + 1, nx - 1)
            iy1 = np.minimum(iy0 + 1, ny - 1)

            z00 = grid[ix0, iy0]
            z10 = grid[ix1, iy0]
            z01 = grid[ix0, iy1]
            z11 = grid[ix1, iy1]

            w00 = (1.0 - fx) * (1.0 - fy)
            w10 = fx * (1.0 - fy)
            w01 = (1.0 - fx) * fy
            w11 = fx * fy

            canopy_z = (z00 * w00 + z10 * w10 + z01 * w01 + z11 * w11).astype(np.float64, copy=False)

            # If any neighbor is missing (nan), fall back to nearest cell value.
            invalid = ~(np.isfinite(z00) & np.isfinite(z10) & np.isfinite(z01) & np.isfinite(z11))
            if np.any(invalid):
                canopy_z[invalid] = canopy_z_nearest[invalid]

    progress.done()
    meta = {
        "x0": x0,
        "y0": y0,
        "cell_size": float(cell_size),
        "mode": mode,
        "unique_cells": int(len(unique_keys)),
        "min_ix": int(ix.min()),
        "max_ix": int(ix.max()),
        "min_iy": int(iy.min()),
        "max_iy": int(iy.max()),
    }
    return canopy_z, meta

# ------------------------
# main simulation
# ------------------------
def simulate_occlusion(
    input_las,
    output_las,
    kappa,
    seed=0,
    *,
    cell_size=CELL_SIZE,
    labels_in: Optional[str] = None,
    labels_out: Optional[str] = None,
    embed_labels: bool = True,
    embed_labels_dim: str = "label",
    coord_tol: float = 1e-3,
    skip_label_coord_check: bool = False,
    verbose=True,
    progress_every=10_000,
    heightmap_mode: str = "bilinear",
):
    np.random.seed(seed)

    if verbose:
        print("[simulate] parameters")
        print(f"[simulate] input={input_las}")
        print(f"[simulate] output={output_las}")
        print(f"[simulate] kappa={kappa}")
        print(f"[simulate] seed={seed}")
        print(f"[simulate] cell_size={cell_size}")

    las = laspy.read(input_las)
    x = las.x
    y = las.y
    z = las.z

    labels = None
    if labels_in is None:
        # If labels already exist in the LAS/LAZ, they will be preserved automatically
        # by slicing `las.points`. If not, try to find a sidecar labels file.
        sidecar = _find_sidecar_labels_file(input_las)
        if sidecar is not None:
            labels_in = sidecar
            if verbose:
                print(f"[simulate] auto-detected labels file: {labels_in}")

    if labels_in is not None:
        if verbose:
            print(f"[simulate] labels_in={labels_in}")
        lx, ly, lz, labels = _load_labels_file(labels_in)
        if len(labels) != len(z):
            raise ValueError(
                f"labels count mismatch: labels={len(labels)} vs las points={len(z)}"
            )
        if not skip_label_coord_check:
            max_dx = float(np.max(np.abs(lx - x)))
            max_dy = float(np.max(np.abs(ly - y)))
            max_dz = float(np.max(np.abs(lz - z)))
            if verbose:
                print(
                    "[simulate] label XYZ alignment max abs diff (m): "
                    f"dx={max_dx:.6g} dy={max_dy:.6g} dz={max_dz:.6g}"
                )
            if max(max_dx, max_dy, max_dz) > float(coord_tol):
                raise ValueError(
                    "labels XYZ do not match LAS XYZ within tolerance. "
                    f"max(dx,dy,dz)={max(max_dx, max_dy, max_dz):.6g} > coord_tol={coord_tol}. "
                    "If you are sure the ordering matches, pass --skip-label-coord-check or increase --coord-tol."
                )

    if verbose:
        print(f"[simulate] loaded points: {len(z):,}")

    t0 = time.time()

    canopy_z, hm_meta = canopy_height_per_point(
        x,
        y,
        z,
        cell_size=cell_size,
        mode=heightmap_mode,
        progress=ProgressPrinter(enabled=verbose, every=progress_every, label="heightmap"),
    )
    if verbose:
        nx = hm_meta["max_ix"] - hm_meta["min_ix"] + 1
        ny = hm_meta["max_iy"] - hm_meta["min_iy"] + 1
        print(f"[simulate] heightmap cells used: {hm_meta['unique_cells']:,} (grid approx {nx}x{ny})")

    # Beer–Lambert attenuation based on vertical path length through canopy.
    # L is the vertical distance to canopy start in the same (x,y) cell, clamped at 0.
    L = np.maximum(0.0, canopy_z - z)
    P = np.exp(-kappa * L)

    if verbose:
        # Some quick distribution stats
        print(
            "[simulate] L stats (m): "
            f"min={float(L.min()):.3f} p50={float(np.median(L)):.3f} max={float(L.max()):.3f}"
        )
        print(
            "[simulate] P stats: "
            f"min={float(P.min()):.3f} p50={float(np.median(P)):.3f} max={float(P.max()):.3f}"
        )

    if verbose:
        print("[simulate] sampling points...")
    keep_mask = np.random.rand(len(P)) < P

    # Preserve all point dimensions (classification, intensity, rgb, etc.)
    out_header = copy.deepcopy(las.header)
    out = laspy.LasData(out_header)
    out.points = las.points[keep_mask]

    if labels is not None:
        # Default behavior: embed labels into output LAS/LAZ as an extra bytes dimension.
        if embed_labels:
            dim_name = embed_labels_dim.strip() or "label"
            if dim_name not in out.point_format.dimension_names:
                out.add_extra_dim(
                    laspy.ExtraBytesParams(
                        name=dim_name,
                        type=np.int32,
                        description="Point-wise label (e.g., -1,0,1,2,...)",
                    )
                )
            setattr(out, dim_name, labels[keep_mask].astype(np.int32, copy=False))
            if verbose:
                print(f"[simulate] embedded labels into LAS dimension: {dim_name}")

        # Optional: also write a sidecar labels file.
        if labels_out is not None or (not embed_labels):
            if labels_out is None:
                in_labels_path = Path(labels_in)
                out_path = Path(output_las)
                labels_out = str(out_path.with_suffix(in_labels_path.suffix))
            _save_labels_file(labels_out, x[keep_mask], y[keep_mask], z[keep_mask], labels[keep_mask])
            if verbose:
                print(f"[simulate] saved labels: {labels_out}")

    # If labels existed in the input LAS/LAZ as an extra dimension (e.g. "treeID"),
    # they have already been preserved by `out.points = las.points[keep_mask]`.
    # In verbose mode, print quick stats so it's obvious labels are present.
    if verbose:
        dims = set(out.point_format.dimension_names)
        found_name = next((n for n in _KNOWN_LABEL_DIM_NAMES if n in dims), None)
        if found_name is not None:
            try:
                arr = getattr(out, found_name)
                print(
                    f"[simulate] label dimension found: {found_name} "
                    f"(dtype={arr.dtype}, min={int(arr.min())}, max={int(arr.max())})"
                )
            except Exception as e:
                print(f"[simulate] label dimension found: {found_name} (stats unavailable: {e})")
        elif labels_in is None:
            print(
                "[simulate] note: no labels were embedded or detected. "
                "If your labels are stored in a sidecar file, place it next to the input as "
                "<stem>_labels.npy/.txt or pass --labels explicitly."
            )

    out.write(output_las)

    if verbose:
        kept = int(np.count_nonzero(keep_mask))
        removed = int(len(P) - kept)
        print(f"[simulate] kept points: {kept:,} (removed {removed:,})")
        print(f"[simulate] total elapsed: {time.time() - t0:.1f}s")
    print(f"Saved: {output_las}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Random thinning / occlusion simulation using Beer–Lambert law. "
            "All points are attenuated based on a low-resolution top-view heightmap."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        "--input_las",
        default="input_tls.las",
        help="Input LAS file (default: input_tls.las)",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output_las",
        default=None,
        help=(
            "Output LAS file. If omitted, defaults to: "
            "<input_stem>_occlusion_k=<kappa>.las"
        ),
    )
    parser.add_argument(
        "-k",
        "--kappa",
        type=float,
        required=True,
        help="Beer–Lambert attenuation coefficient (required)",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=CELL_SIZE,
        help=f"Heightmap grid size in meters (default: {CELL_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Optional labels file (.npy or space-delimited .txt) with 4 rows/cols: x y z label. "
            "If provided, occlusion is applied to labels and a filtered labels file is written."
        ),
    )
    parser.add_argument(
        "--labels-out",
        default=None,
        help=(
            "Output labels file path (.npy or .txt). "
            "Default: same name as output LAS but with labels input extension."
        ),
    )
    parser.add_argument(
        "--no-embed-labels",
        action="store_true",
        help=(
            "Do not embed labels into output LAS/LAZ. "
            "If set, a labels file will be written (default path if --labels-out omitted)."
        ),
    )
    parser.add_argument(
        "--labels-dim",
        default="label",
        help="Extra dimension name to store labels in LAS/LAZ (default: label)",
    )
    parser.add_argument(
        "--coord-tol",
        type=float,
        default=1e-3,
        help="Max allowed absolute XYZ difference (m) between LAS and labels file (default: 1e-3)",
    )
    parser.add_argument(
        "--skip-label-coord-check",
        action="store_true",
        help="Skip XYZ alignment check between LAS and labels file (default: disabled)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Print progress every N iterations (default: 10000)",
    )
    parser.add_argument(
        "--heightmap-mode",
        choices=["nearest", "bilinear"],
        default="bilinear",
        help="Canopy height lookup mode (default: bilinear)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress/meta prints (default: enabled)",
    )

    args = parser.parse_args(argv)

    output_las = args.output
    if output_las is None:
        in_path = Path(args.input)
        kappa_str = f"{args.kappa:g}"
        out_name = f"{in_path.stem}_occlusion_k={kappa_str}{in_path.suffix}"
        output_las = str(in_path.with_name(out_name))

    simulate_occlusion(
        input_las=args.input,
        output_las=output_las,
        kappa=args.kappa,
        seed=args.seed,
        cell_size=args.cell_size,
        labels_in=args.labels,
        labels_out=args.labels_out,
        embed_labels=not args.no_embed_labels,
        embed_labels_dim=args.labels_dim,
        coord_tol=args.coord_tol,
        skip_label_coord_check=args.skip_label_coord_check,
        verbose=not args.quiet,
        progress_every=args.progress_every,
        heightmap_mode=args.heightmap_mode,
    )

# ------------------------
# example
# ------------------------
if __name__ == "__main__":
    main()
