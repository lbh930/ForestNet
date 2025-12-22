import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import laspy
import numpy as np

# ------------------------
# parameters
# ------------------------
CELL_SIZE = 1.0       # heightmap grid size (m)


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
    progress: Optional[ProgressPrinter] = None,
):
    """Build a low-resolution top-view heightmap and return canopy Z for each point.

    Heightmap definition: per (x,y) grid cell, canopy height is max(Z) of points in that cell.
    """
    if progress is None:
        progress = ProgressPrinter(enabled=False)
    progress.label = progress.label or "heightmap"
    progress.start(total=None)

    ix, iy, x0, y0 = _compute_cell_indices(x, y, cell_size=cell_size)

    # Packed 64-bit cell key (stable, sortable) without allocating a huge dense grid.
    key = (ix.astype(np.int64) << 32) ^ (iy.astype(np.int64) & np.int64(0xFFFFFFFF))
    order = np.argsort(key)
    key_s = key[order]
    z_s = z[order]

    unique_keys, idx_start = np.unique(key_s, return_index=True)
    cell_max_z = np.maximum.reduceat(z_s, idx_start)

    # Map each point back to its cell max Z.
    cell_index = np.searchsorted(unique_keys, key)
    canopy_z = cell_max_z[cell_index]

    progress.done()
    meta = {
        "x0": x0,
        "y0": y0,
        "cell_size": float(cell_size),
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
    verbose=True,
    progress_every=10_000,
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

    if verbose:
        print(f"[simulate] loaded points: {len(z):,}")

    t0 = time.time()

    canopy_z, hm_meta = canopy_height_per_point(
        x,
        y,
        z,
        cell_size=cell_size,
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

    out = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    out.header = las.header
    # Preserve all point dimensions (classification, intensity, rgb, etc.)
    out.points = las.points[keep_mask]

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
        "--progress-every",
        type=int,
        default=10_000,
        help="Print progress every N iterations (default: 10000)",
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
        out_name = f"{in_path.stem}_occlusion_k={kappa_str}.las"
        output_las = str(in_path.with_name(out_name))

    simulate_occlusion(
        input_las=args.input,
        output_las=output_las,
        kappa=args.kappa,
        seed=args.seed,
        cell_size=args.cell_size,
        verbose=not args.quiet,
        progress_every=args.progress_every,
    )

# ------------------------
# example
# ------------------------
if __name__ == "__main__":
    main()
