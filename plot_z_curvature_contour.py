#!/usr/bin/env python3
"""
Continuous s-vs-z colormap with contour lines from exported PyElastica frame files.

Run from the main output folder:

    cd outputs
    python plot_z_curvature_contour.py

Expected structure:

    outputs/
    ├── simulation_parameters.txt
    └── frames_txt/
        ├── generation_frame_00000.txt
        ├── target_elongation_frame_00000.txt
        ├── unwinding_frame_00000.txt
        └── ...

Produces continuous blue-red colormap PDFs:

    z_curvature_contour_all.pdf
    z_curvature_contour_generation.pdf
    z_curvature_contour_target_elongation.pdf
    z_curvature_contour_unwinding.pdf

Color quantity:

    Q(s) = (kappa_1 - kappa0_1)^2
           + Lambda * kappa_2^2
           + Gamma  * kappa_3^2

Vertical axis:

    z = clamp separation / rod arclength
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def read_parameters(path: Path) -> dict[str, str]:
    params: dict[str, str] = {}
    if not path.exists():
        return params

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            params[key.strip()] = value.strip()

    return params


def get_float_param(params: dict[str, str], key: str, default: float) -> float:
    try:
        return float(params.get(key, default))
    except Exception:
        return float(default)


def read_frame_table(frame_path: Path):
    meta: dict[str, str] = {}
    header = None
    numeric_lines = []

    with frame_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("#"):
                clean = stripped[1:].strip()
                if ":" in clean:
                    key, value = clean.split(":", 1)
                    meta[key.strip()] = value.strip()
                continue

            if header is None:
                header = stripped.split()
                continue

            numeric_lines.append(stripped)

    if header is None:
        raise ValueError(f"No table header found in {frame_path}")

    if not numeric_lines:
        raise ValueError(f"No numeric rows found in {frame_path}")

    ncol = len(header)
    rows = []

    for i, line in enumerate(numeric_lines, start=1):
        parts = line.split()
        if len(parts) != ncol:
            raise ValueError(
                f"{frame_path.name}: row {i} has {len(parts)} columns, expected {ncol}"
            )
        rows.append([float(x) for x in parts])

    arr = np.asarray(rows, dtype=float)
    cols = {name: arr[:, j] for j, name in enumerate(header)}

    stage = meta.get("stage", "unknown")

    try:
        time = float(meta.get("time", np.nan))
    except Exception:
        time = np.nan

    return stage, time, cols


def compute_frame_elongation_z(cols):
    """
    Compute clamp elongation:

        z = ||x_right - x_left|| / L

    where L is the final arclength value s[-1].
    """
    required = ["s", "x", "y", "z"]
    missing = [name for name in required if name not in cols]
    if missing:
        raise KeyError(f"Missing columns needed for elongation: {missing}")

    x = cols["x"]
    y = cols["y"]
    zz = cols["z"]
    s = cols["s"]

    p_left = np.array([x[0], y[0], zz[0]], dtype=float)
    p_right = np.array([x[-1], y[-1], zz[-1]], dtype=float)

    clamp_distance = float(np.linalg.norm(p_right - p_left))
    rod_length = float(s[-1] - s[0])

    if rod_length <= 0.0:
        return np.nan

    return clamp_distance / rod_length


def compute_quantity(cols, Lambda: float, Gamma: float, rest_subtract_all: bool = False):
    required = ["s", "kappa_1", "kappa_2", "kappa_3", "kappa0_1"]
    missing = [name for name in required if name not in cols]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    s = cols["s"]

    k1 = cols["kappa_1"]
    k2 = cols["kappa_2"]
    k3 = cols["kappa_3"]

    k10 = cols["kappa0_1"]

    if rest_subtract_all:
        if "kappa0_2" not in cols or "kappa0_3" not in cols:
            raise KeyError("Need kappa0_2 and kappa0_3 for --rest-subtract-all")
        k20 = cols["kappa0_2"]
        k30 = cols["kappa0_3"]
        q = (k1 - k10) ** 2 + Lambda * (k2 - k20) ** 2 + Gamma * (k3 - k30) ** 2
    else:
        q = (k1 - k10) ** 2 + Lambda * k2**2 + Gamma * k3**2

    return s, q


def collect_frames(frames_dir, stage_filter, Lambda, Gamma, rest_subtract_all):
    frame_paths = sorted(frames_dir.glob("*_frame_*.txt"))
    rows = []

    for frame_path in frame_paths:
        try:
            stage, time, cols = read_frame_table(frame_path)

            if stage_filter is not None and stage != stage_filter:
                continue

            elongation_z = compute_frame_elongation_z(cols)

            if not np.isfinite(elongation_z):
                print(f"Skipping {frame_path}: non-finite elongation z")
                continue

            s, q = compute_quantity(
                cols,
                Lambda=Lambda,
                Gamma=Gamma,
                rest_subtract_all=rest_subtract_all,
            )

            rows.append(
                {
                    "path": frame_path,
                    "stage": stage,
                    "time": time,
                    "elongation_z": elongation_z,
                    "s": s,
                    "q": q,
                }
            )

        except Exception as exc:
            print(f"Skipping {frame_path}: {exc}")

    rows.sort(key=lambda r: (r["elongation_z"], r["time"], str(r["path"])))
    return rows


def rows_to_grid(rows, n_s=None):
    """
    Convert frames to a rectangular grid.

    Each frame has values q(s).
    Elongation z is one y-value per frame.
    """
    if not rows:
        raise ValueError("No rows to grid")

    if n_s is None:
        n_s = max(len(r["s"]) for r in rows)

    s_min = min(float(np.nanmin(r["s"])) for r in rows)
    s_max = max(float(np.nanmax(r["s"])) for r in rows)
    s_grid = np.linspace(s_min, s_max, n_s)

    z_values = np.asarray([r["elongation_z"] for r in rows], dtype=float)
    q_grid = np.empty((len(rows), n_s), dtype=float)

    for i, r in enumerate(rows):
        s = np.asarray(r["s"], dtype=float)
        q = np.asarray(r["q"], dtype=float)

        finite = np.isfinite(s) & np.isfinite(q)
        s = s[finite]
        q = q[finite]

        order = np.argsort(s)
        s = s[order]
        q = q[order]

        s_unique, unique_idx = np.unique(s, return_index=True)
        q_unique = q[unique_idx]

        q_grid[i, :] = np.interp(s_grid, s_unique, q_unique)

    return s_grid, z_values, q_grid


def plot_continuous_colormap(
    rows,
    output_path: Path,
    title: str,
    log_color: bool = False,
    contour_levels: int = 12,
    n_s: int | None = None,
):
    if not rows:
        print(f"No data for: {title}")
        return

    s_grid, z_values, q_grid = rows_to_grid(rows, n_s=n_s)

    finite = np.isfinite(q_grid)
    if not np.any(finite):
        print(f"No finite Q values for: {title}")
        return

    if log_color:
        z_color = np.log10(np.maximum(q_grid, 1.0e-30))
        c_label = r"$\log_{10}(Q)$"
    else:
        z_color = q_grid
        c_label = r"$Q$"

    S, Z = np.meshgrid(s_grid, z_values)

    fig, ax = plt.subplots(figsize=(11, 6))

    pcm = ax.pcolormesh(
        S,
        Z,
        z_color,
        shading="auto",
        cmap="coolwarm",
    )

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(c_label)

    finite_z = z_color[np.isfinite(z_color)]
    if finite_z.size > 0 and np.nanmax(finite_z) > np.nanmin(finite_z):
        levels = np.linspace(np.nanmin(finite_z), np.nanmax(finite_z), contour_levels)
        cs = ax.contour(
            S,
            Z,
            z_color,
            levels=levels,
            colors="k",
            linewidths=0.55,
            alpha=0.65,
        )
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")

    ax.set_xlabel("Curvilinear abscissa $s$")
    ax.set_ylabel(r"Clamp elongation $z = d_{\mathrm{clamps}} / L$")
    ax.set_title(title)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-folder",
        type=str,
        default=".",
        help="Main output folder containing simulation_parameters.txt and frames_txt/",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["generation", "target_elongation", "unwinding", "translation", "z_init", "unwind_to_link"],
        help="Only plot one stage.",
    )
    parser.add_argument(
        "--lambda-bend",
        type=float,
        default=None,
        help="Override Lambda. Otherwise read from simulation_parameters.txt.",
    )
    parser.add_argument(
        "--gamma-twist",
        type=float,
        default=None,
        help="Override Gamma. Otherwise read from simulation_parameters.txt.",
    )
    parser.add_argument(
        "--log-color",
        action="store_true",
        help="Use log10(Q) color scale.",
    )
    parser.add_argument(
        "--rest-subtract-all",
        action="store_true",
        help="Use (kappa2-kappa0_2)^2 and (kappa3-kappa0_3)^2 too.",
    )
    parser.add_argument(
        "--contour-levels",
        type=int,
        default=12,
        help="Number of contour levels.",
    )
    parser.add_argument(
        "--n-s",
        type=int,
        default=None,
        help="Number of interpolated points along s. Default: number of rod nodes.",
    )

    args = parser.parse_args()

    output_folder = Path(args.output_folder).resolve()
    frames_dir = output_folder / "frames_txt"
    params_path = output_folder / "simulation_parameters.txt"

    if not frames_dir.exists():
        raise FileNotFoundError(f"Could not find frames directory: {frames_dir}")

    params = read_parameters(params_path)

    Lambda = args.lambda_bend
    if Lambda is None:
        Lambda = get_float_param(params, "lambda_bend", 1.0)

    Gamma = args.gamma_twist
    if Gamma is None:
        Gamma = get_float_param(params, "gamma_twist", 1.0)

    print(f"Output folder: {output_folder}")
    print(f"Frames dir   : {frames_dir}")
    print(f"Lambda       : {Lambda}")
    print(f"Gamma        : {Gamma}")

    if args.stage is not None:
        stages = [args.stage]
    else:
        stages = [None, "generation", "target_elongation", "unwinding", "z_init", "unwind_to_link", "translation"]

    if args.rest_subtract_all:
        formula = (
            r"$Q=(\kappa_1-\kappa_{1,0})^2"
            r"+\Lambda(\kappa_2-\kappa_{2,0})^2"
            r"+\Gamma(\kappa_3-\kappa_{3,0})^2$"
        )
    else:
        formula = (
            r"$Q=(\kappa_1-\kappa_{1,0})^2"
            r"+\Lambda\kappa_2^2+\Gamma\kappa_3^2$"
        )

    for stage in stages:
        rows = collect_frames(
            frames_dir=frames_dir,
            stage_filter=stage,
            Lambda=Lambda,
            Gamma=Gamma,
            rest_subtract_all=args.rest_subtract_all,
        )

        name = "all" if stage is None else stage

        suffix = ""
        if args.log_color:
            suffix += "_log"
        if args.rest_subtract_all:
            suffix += "_restsub"

        output_path = output_folder / f"z_curvature_contour_{name}{suffix}.pdf"

        title = f"{name}: {formula}\nLambda={Lambda:g}, Gamma={Gamma:g}"

        plot_continuous_colormap(
            rows=rows,
            output_path=output_path,
            title=title,
            log_color=args.log_color,
            contour_levels=args.contour_levels,
            n_s=args.n_s,
        )


if __name__ == "__main__":
    main()