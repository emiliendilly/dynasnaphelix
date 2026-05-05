#!/usr/bin/env python3
"""
PyElastica link-stop sweep for the multi-wrap rod experiment.

Protocol for each parallel run:
1) Generate the rod.
2) Move to a common initial axial extension z_init = 0.5.
3) Unwind until the iterative link reaches that run's assigned target link value.
4) Stop twisting there.
5) Translate the clamps axially from z = 0.5 to z = 0.1.
6) Record I2 and I3/I2 during that translation.

Sweep:
- target link values are swept over [0, -7] with 20 points by default
- each parallel job corresponds to one target link value

Main outputs per run:
- simulation_parameters.txt
- link_history_all_steps.txt
- per-frame txt tables
- unwinding_to_target_summary.txt
- translation_summary.txt
- final_summary.txt
- translation.gif

Notes:
- gamma is constant across the sweep
- Tw/Wr/Lk_closure remain optional and disabled by default
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# Optional progress bar
# -----------------------------------------------------------------------------
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


def progress_iter(iterable, total=None, desc=""):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)

    def _gen():
        if total is None:
            for x in iterable:
                yield x
            return

        bar_len = 30
        last_pct = -1
        for i, x in enumerate(iterable, 1):
            pct = int(100 * i / max(1, total))
            stride = max(1, total // 60)
            if i == 1 or i == total or (i % stride == 0 and pct != last_pct):
                filled = int(bar_len * i / max(1, total))
                bar = "#" * filled + "-" * (bar_len - filled)
                print(f"\r{desc:>24} [{bar}] {pct:3d}% ({i}/{total})", end="", flush=True)
                last_pct = pct
                if i == total:
                    print()
            yield x

    return _gen()


# -----------------------------------------------------------------------------
# Robust PyElastica imports
# -----------------------------------------------------------------------------
try:
    from elastica import BaseSystemCollection, Constraints, Forcing, CallBacks, CosseratRod
    from elastica.external_forces import NoForces
    from elastica.boundary_conditions import ConstraintBase
    from elastica.timestepper.symplectic_steppers import PositionVerlet
    from elastica.timestepper import extend_stepper_interface
except Exception:
    from elastica import BaseSystemCollection, Constraints, Forcing, CallBacks
    from elastica.rod.cosserat_rod import CosseratRod
    from elastica.external_forces import NoForces
    from elastica.boundary_conditions import ConstraintBase
    from elastica.timestepper.symplectic_steppers import PositionVerlet
    from elastica.timestepper import extend_stepper_interface


class Simulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="PyElastica multi-wrap rod link-stop sweep with constant gamma and post-stop axial translation."
    )
    parser.add_argument(
        "--lambda-bend",
        dest="lambda_bend",
        type=float,
        required=True,
        help="Lambda = B2/B1, where B1 is the naturally curved bending direction.",
    )
    parser.add_argument(
        "--gamma-twist",
        dest="gamma_twist",
        type=float,
        required=True,
        help="Constant Gamma = Bt/B1 used for every simulation in the sweep.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default="outputs_link_stop_sweep",
        help="Directory containing all sweep results. Each target link gets its own subfolder.",
    )
    parser.add_argument(
        "--extra-unwind-turns",
        dest="extra_unwind_turns",
        type=float,
        default=15.0,
        help="Maximum extra turns available during the unwinding stage.",
    )
    parser.add_argument(
        "--sim-dt-per-frame",
        dest="sim_dt_per_frame",
        type=float,
        default=0.1,
        help="Sampling interval for exported frame txt files and gif frames.",
    )
    parser.add_argument(
        "--torque-mode",
        dest="torque_mode",
        type=str,
        default="internal_couple",
        choices=["internal_couple", "internal_torques"],
        help="Torque-like quantity displayed/exported.",
    )
    parser.add_argument(
        "--torque-offset-samples",
        dest="torque_offset_samples",
        type=int,
        default=5,
        help="Offset from the boundaries when sampling projected torque-like quantity.",
    )

    parser.add_argument(
        "--z-init",
        dest="z_init",
        type=float,
        default=0.5,
        help="Initial axial extension z used before unwinding.",
    )
    parser.add_argument(
        "--z-final",
        dest="z_final",
        type=float,
        default=0.1,
        help="Final axial extension z reached during the translation stage.",
    )
    parser.add_argument(
        "--link-min",
        dest="link_min",
        type=float,
        default=0.0,
        help="First target link value in the sweep.",
    )
    parser.add_argument(
        "--link-max",
        dest="link_max",
        type=float,
        default=-7.0,
        help="Last target link value in the sweep.",
    )
    parser.add_argument(
        "--link-points",
        dest="link_points",
        type=int,
        default=20,
        help="Number of target link values in the sweep.",
    )
    parser.add_argument(
        "--translation-time",
        dest="translation_time",
        type=float,
        default=4.0,
        help="Duration of the post-stop axial translation stage.",
    )

    args = parser.parse_args()

    if args.lambda_bend <= 0:
        raise ValueError("--lambda-bend must be > 0")
    if args.gamma_twist <= 0:
        raise ValueError("--gamma-twist must be > 0")
    if args.sim_dt_per_frame <= 0:
        raise ValueError("--sim-dt-per-frame must be > 0")
    if args.torque_offset_samples < 0:
        raise ValueError("--torque-offset-samples must be >= 0")
    if args.z_init <= 0 or args.z_final <= 0:
        raise ValueError("--z-init and --z-final must be > 0")
    if args.link_points < 1:
        raise ValueError("--link-points must be >= 1")
    if args.translation_time < 0:
        raise ValueError("--translation-time must be >= 0")

    return args


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def normalize(v, eps=1.0e-14):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smoothstep_derivative(x):
    x = np.clip(x, 0.0, 1.0)
    return 6.0 * x * (1.0 - x)


def phase_alpha(time, t_start, duration):
    if duration <= 0.0:
        return 1.0 if time >= t_start else 0.0
    return smoothstep((time - t_start) / duration)


def phase_alpha_dot(time, t_start, duration):
    if duration <= 0.0:
        return 0.0
    return smoothstep_derivative((time - t_start) / duration) / duration


def conservative_dt(dl, radius, rho, E, safety=0.08):
    A = np.pi * radius**2
    I = (np.pi / 4.0) * radius**4
    c = np.sqrt(E / rho)
    dt_stretch = dl / (c + 1e-14)
    dt_bend = (dl**2) * np.sqrt((rho * A) / (E * I + 1e-30))
    return safety * min(dt_stretch, dt_bend)


def rot_about_axis(axis, angle):
    axis = normalize(axis)
    if np.linalg.norm(axis) < 1.0e-14 or abs(angle) < 1.0e-14:
        return np.eye(3)

    x, y, z = axis
    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )
    I = np.eye(3)
    return I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def any_perpendicular(v):
    v = normalize(v)
    basis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(v, basis)) > 0.9:
        basis = np.array([0.0, 1.0, 0.0])
    p = basis - np.dot(basis, v) * v
    return normalize(p)


def set_axes_equal_3d(ax, xyz, pad=0.08):
    mins = np.min(xyz, axis=1)
    maxs = np.max(xyz, axis=1)
    ctr = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)
    if span < 1.0e-12:
        span = 1.0
    half = 0.5 * (1.0 + 2.0 * pad) * span

    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def get_rod_length(system):
    if hasattr(system, "length"):
        return float(system.length)
    if hasattr(system, "rest_lengths"):
        return float(np.sum(np.asarray(system.rest_lengths)))
    pos = np.asarray(system.position_collection)
    dif = pos[:, 1:] - pos[:, :-1]
    seg = np.sqrt(np.sum(dif * dif, axis=0))
    return float(np.sum(seg))


def signed_angle_about_axis(u, v, axis):
    axis = normalize(axis)

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    u = u - np.dot(u, axis) * axis
    v = v - np.dot(v, axis) * axis

    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1.0e-14 or nv < 1.0e-14:
        return 0.0

    u /= nu
    v /= nv

    c = np.clip(np.dot(u, v), -1.0, 1.0)
    s = np.dot(axis, np.cross(u, v))
    return float(np.arctan2(s, c))


def minimal_rotation_map(a, b):
    a = normalize(a)
    b = normalize(b)

    c = np.dot(a, b)
    c = np.clip(c, -1.0, 1.0)

    if c > 1.0 - 1.0e-14:
        return np.eye(3)

    if c < -1.0 + 1.0e-12:
        axis = any_perpendicular(a)
        return rot_about_axis(axis, np.pi)

    axis = normalize(np.cross(a, b))
    angle = np.arccos(c)
    return rot_about_axis(axis, angle)


def transported_spin_increment(prev_t, prev_d1, curr_t, curr_d1):
    R = minimal_rotation_map(prev_t, curr_t)
    prev_d1_transported = R @ prev_d1
    return signed_angle_about_axis(prev_d1_transported, curr_d1, curr_t)


def cumulative_arclength_from_nodes(pos):
    edges = pos[:, 1:] - pos[:, :-1]
    dl = np.linalg.norm(edges, axis=0)
    s = np.zeros(pos.shape[1], dtype=float)
    s[1:] = np.cumsum(dl)
    return s


def element_field_to_nodes(elem_field):
    elem_field = np.asarray(elem_field)
    n_elems = elem_field.shape[1]
    n_nodes = n_elems + 1

    out = np.zeros((3, n_nodes), dtype=float)
    out[:, 0] = elem_field[:, 0]
    out[:, -1] = elem_field[:, -1]
    if n_elems > 1:
        out[:, 1:-1] = 0.5 * (elem_field[:, :-1] + elem_field[:, 1:])
    return out


def voronoi_field_to_nodes(voro_field):
    voro_field = np.asarray(voro_field)
    n_voro = voro_field.shape[1]
    n_elems = n_voro + 1
    n_nodes = n_elems + 1

    out = np.zeros((3, n_nodes), dtype=float)

    if n_voro == 1:
        out[:] = voro_field[:, [0]]
        return out

    out[:, 1] = voro_field[:, 0]
    out[:, -2] = voro_field[:, -1]

    for j in range(2, n_nodes - 2):
        left_v = j - 2
        right_v = j - 1
        out[:, j] = 0.5 * (voro_field[:, left_v] + voro_field[:, right_v])

    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out


# -----------------------------------------------------------------------------
# Director convention handling
# -----------------------------------------------------------------------------
def detect_director_mapping(rod, verbose=False):
    Dall = np.asarray(rod.director_collection)
    pos = np.asarray(rod.position_collection)

    tangents = pos[:, 1:] - pos[:, :-1]
    tan_norm = np.linalg.norm(tangents, axis=0)
    good = tan_norm > 1e-14
    if not np.any(good):
        raise ValueError("Could not determine rod tangents for director convention detection")

    tangents[:, good] /= tan_norm[good][None, :]

    col_scores = []
    row_scores = []

    for k in range(Dall.shape[2]):
        if not good[k]:
            continue
        D = Dall[:, :, k]
        t = tangents[:, k]
        d3_col = D[:, 2]
        d3_row = D[2, :]
        col_scores.append(abs(np.dot(d3_col, t)))
        row_scores.append(abs(np.dot(d3_row, t)))

    col_score = float(np.mean(col_scores)) if col_scores else -np.inf
    row_score = float(np.mean(row_scores)) if row_scores else -np.inf
    mapping_mode = "columns" if col_score >= row_score else "rows"

    if verbose:
        print("Director mapping detection:")
        print(f"  mean |D[:,2] · tangent| = {col_score:.8f}")
        print(f"  mean |D[2,:] · tangent| = {row_score:.8f}")
        print(f"  selected mapping        = {mapping_mode}")

    return mapping_mode


def validate_director_mapping_on_rod(rod, mapping_mode):
    Dall = np.asarray(rod.director_collection)
    pos = np.asarray(rod.position_collection)

    tangents = pos[:, 1:] - pos[:, :-1]
    tan_norm = np.linalg.norm(tangents, axis=0)
    good = tan_norm > 1e-14
    tangents[:, good] /= tan_norm[good][None, :]

    vals = []
    for k in range(Dall.shape[2]):
        if not good[k]:
            continue
        D = Dall[:, :, k]
        d3 = D[:, 2] if mapping_mode == "columns" else D[2, :]
        vals.append(np.dot(d3, tangents[:, k]))

    vals = np.asarray(vals, dtype=float)
    if vals.size:
        print("Director/tangent alignment diagnostic:")
        print(f"  mean(d3·tangent)   = {np.mean(vals): .8f}")
        print(f"  mean(|d3·tangent|) = {np.mean(np.abs(vals)): .8f}")
        print(f"  min(|d3·tangent|)  = {np.min(np.abs(vals)): .8f}")
        print(f"  max(|d3·tangent|)  = {np.max(np.abs(vals)): .8f}")


def material_vec_to_lab(D, m_material, mapping_mode):
    D = np.asarray(D, dtype=float)
    m_material = np.asarray(m_material, dtype=float)
    if mapping_mode == "columns":
        return D @ m_material
    if mapping_mode == "rows":
        return D.T @ m_material
    raise ValueError(f"Unknown mapping_mode={mapping_mode!r}")


def node_frames_from_directors(director_collection, mapping_mode):
    D = np.asarray(director_collection)
    n_elems = D.shape[2]
    n_nodes = n_elems + 1

    d1n = np.zeros((3, n_nodes))
    d2n = np.zeros((3, n_nodes))
    d3n = np.zeros((3, n_nodes))

    if mapping_mode == "columns":
        d1n[:, :-1] = D[:, 0, :]
        d2n[:, :-1] = D[:, 1, :]
        d3n[:, :-1] = D[:, 2, :]
        d1n[:, -1] = D[:, 0, -1]
        d2n[:, -1] = D[:, 1, -1]
        d3n[:, -1] = D[:, 2, -1]
    elif mapping_mode == "rows":
        d1n[:, :-1] = D[0, :, :]
        d2n[:, :-1] = D[1, :, :]
        d3n[:, :-1] = D[2, :, :]
        d1n[:, -1] = D[0, :, -1]
        d2n[:, -1] = D[1, :, -1]
        d3n[:, -1] = D[2, :, -1]
    else:
        raise ValueError(f"Unknown mapping_mode={mapping_mode!r}")

    return d1n, d2n, d3n


def node_material_to_lab(field_node_material, director_collection, mapping_mode):
    field_node_material = np.asarray(field_node_material)
    n_nodes = field_node_material.shape[1]
    out = np.zeros_like(field_node_material)

    for i in range(n_nodes - 1):
        D = director_collection[:, :, i]
        out[:, i] = material_vec_to_lab(D, field_node_material[:, i], mapping_mode)

    out[:, -1] = out[:, -2]
    return out


# -----------------------------------------------------------------------------
# Row-wise director interpolation for generation BC
# -----------------------------------------------------------------------------
def rotation_matrix_to_axis_angle(R):
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    cos_theta = 0.5 * (tr - 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1.0e-12:
        return np.array([0.0, 0.0, 1.0]), 0.0

    if np.pi - theta < 1.0e-8:
        A = 0.5 * (R + np.eye(3))
        axis = np.array(
            [
                np.sqrt(max(A[0, 0], 0.0)),
                np.sqrt(max(A[1, 1], 0.0)),
                np.sqrt(max(A[2, 2], 0.0)),
            ]
        )
        if R[2, 1] - R[1, 2] < 0:
            axis[0] *= -1
        if R[0, 2] - R[2, 0] < 0:
            axis[1] *= -1
        if R[1, 0] - R[0, 1] < 0:
            axis[2] *= -1
        axis = normalize(axis)
        return axis, theta

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    ) / (2.0 * np.sin(theta))
    axis = normalize(axis)
    return axis, theta


def interpolate_director_rows(D_start, D_target, alpha):
    C0 = D_start.T
    C1 = D_target.T
    Rmap = C1 @ C0.T
    axis, theta = rotation_matrix_to_axis_angle(Rmap)
    Ra = rot_about_axis(axis, alpha * theta)
    C = Ra @ C0
    return C.T, axis, theta


# -----------------------------------------------------------------------------
# Initial explicit multi-wrap circle
# -----------------------------------------------------------------------------
def build_multiwrap_circle_geometry(
    n_elems: int,
    n_turns: int,
    circle_radius: float,
    z_perturbation: float = 0.0,
):
    n_nodes = n_elems + 1
    theta = np.linspace(0.0, 2.0 * np.pi * n_turns, n_nodes)

    position = np.zeros((3, n_nodes))
    position[0, :] = circle_radius * np.cos(theta)
    position[1, :] = circle_radius * np.sin(theta)
    position[2, :] = z_perturbation * np.sin(theta / max(1, n_turns))

    edges = position[:, 1:] - position[:, :-1]
    lengths = np.linalg.norm(edges, axis=0)
    tangents = edges / lengths[None, :]

    directors = np.zeros((3, 3, n_elems))
    d1_ref = np.array([0.0, 0.0, 1.0])

    for i in range(n_elems):
        d3 = tangents[:, i]

        d1 = d1_ref - np.dot(d1_ref, d3) * d3
        if np.linalg.norm(d1) < 1.0e-12:
            alt = np.array([1.0, 0.0, 0.0])
            d1 = alt - np.dot(alt, d3) * d3

        d1 = normalize(d1)
        d2 = normalize(np.cross(d3, d1))
        d1 = normalize(np.cross(d2, d3))

        directors[0, :, i] = d1
        directors[1, :, i] = d2
        directors[2, :, i] = d3

    base_length = float(np.sum(lengths))
    return position, directors, base_length


def make_current_configuration_rest_state(rod):
    pos = rod.position_collection
    edges = pos[:, 1:] - pos[:, :-1]
    elem_lengths = np.linalg.norm(edges, axis=0)

    rod.rest_lengths[:] = elem_lengths
    if hasattr(rod, "rest_voronoi_lengths") and elem_lengths.size >= 2:
        rod.rest_voronoi_lengths[:] = 0.5 * (elem_lengths[:-1] + elem_lengths[1:])

    rod.rest_sigma[:] = rod.sigma.copy()
    rod.rest_kappa[:] = rod.kappa.copy()

    rod.velocity_collection[:] = 0.0
    rod.omega_collection[:] = 0.0

    for name in (
        "internal_forces",
        "internal_torques",
        "external_forces",
        "external_torques",
        "internal_couple",
        "external_couple",
        "internal_couples",
        "external_couples",
    ):
        if hasattr(rod, name):
            getattr(rod, name)[:] = 0.0


def apply_curvature_aligned_anisotropic_rigidity(
    rod,
    lambda_bend=1.0,
    gamma_twist=1.0,
):
    BM = np.asarray(rod.bend_matrix)
    n_voronoi = BM.shape[2]
    new_BM = np.zeros_like(BM)

    base_B = 0.5 * (BM[0, 0, :] + BM[1, 1, :])

    for k in range(n_voronoi):
        B1 = float(base_B[k])
        B2 = float(lambda_bend * B1)
        Bt = float(gamma_twist * B1)

        kappa0 = np.asarray(rod.rest_kappa[:2, k], dtype=float)
        nrm = np.linalg.norm(kappa0)

        if nrm < 1.0e-14:
            u = np.array([1.0, 0.0])
        else:
            u = kappa0 / nrm
        v = np.array([-u[1], u[0]])

        B2x2 = B1 * np.outer(u, u) + B2 * np.outer(v, v)

        Bk = np.zeros((3, 3))
        Bk[0:2, 0:2] = B2x2
        Bk[2, 2] = Bt
        new_BM[:, :, k] = Bk

    rod.bend_matrix[:] = new_BM
    if hasattr(rod, "inv_bend_matrix"):
        for k in range(n_voronoi):
            rod.inv_bend_matrix[:, :, k] = np.linalg.inv(rod.bend_matrix[:, :, k])


# -----------------------------------------------------------------------------
# No-force helper
# -----------------------------------------------------------------------------
class RampGravity(NoForces):
    def __init__(self, g_vec, ramp_time):
        super().__init__()
        self.g = np.asarray(g_vec, dtype=float)
        self.ramp_time = float(ramp_time)

    def apply_forces(self, system, time: float):
        a = 1.0 if self.ramp_time <= 0 else smoothstep(time / self.ramp_time)
        system.external_forces += (a * self.g)[:, None] * system.mass[None, :]


# -----------------------------------------------------------------------------
# Boundary conditions
# -----------------------------------------------------------------------------
class PullCenterAlignClampBC(ConstraintBase):
    """
    Stage 1: pinned x,y + pull in ±z, free rotation
    Stage 2: translate endpoints to z-axis (x=0,y=0), free rotation
    Stage 3: rotate endpoint frames so left tangent = +z and right tangent = -z
    """

    def __init__(
        self,
        total_pull_z,
        t_pull,
        t_center,
        t_align,
        _system=None,
        **kwargs,
    ):
        super().__init__(_system=_system)

        self.total_pull_z = float(total_pull_z)
        self.t_pull = float(t_pull)
        self.t_center = float(t_center)
        self.t_align = float(t_align)

        self.t1_end = self.t_pull
        self.t2_end = self.t1_end + self.t_center
        self.t3_end = self.t2_end + self.t_align

        self.left_ref = None
        self.right_ref = None

        self.stage2_initialized = False
        self.stage3_initialized = False

        self.pL_stage2_start = None
        self.pR_stage2_start = None
        self.pL_stage2_target = None
        self.pR_stage2_target = None

        self.DL_stage3_start = None
        self.DR_stage3_start = None
        self.DL_stage3_target = None
        self.DR_stage3_target = None
        self.axis_left = np.array([0.0, 0.0, 1.0])
        self.axis_right = np.array([0.0, 0.0, 1.0])
        self.theta_left = 0.0
        self.theta_right = 0.0

    def _initialize_stage2_if_needed(self, system, time):
        if self.stage2_initialized or time < self.t1_end:
            return

        self.pL_stage2_start = system.position_collection[:, 0].copy()
        self.pR_stage2_start = system.position_collection[:, -1].copy()

        self.pL_stage2_target = np.array([0.0, 0.0, self.pL_stage2_start[2]])
        self.pR_stage2_target = np.array([0.0, 0.0, self.pR_stage2_start[2]])

        self.stage2_initialized = True

    def _initialize_stage3_if_needed(self, system, time):
        if self.stage3_initialized or time < self.t2_end:
            return

        self.DL_stage3_start = system.director_collection[:, :, 0].copy()
        self.DR_stage3_start = system.director_collection[:, :, -1].copy()

        self.DL_stage3_target = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        self.DR_stage3_target = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        _, self.axis_left, self.theta_left = interpolate_director_rows(
            self.DL_stage3_start, self.DL_stage3_target, 0.0
        )
        _, self.axis_right, self.theta_right = interpolate_director_rows(
            self.DR_stage3_start, self.DR_stage3_target, 0.0
        )

        self.stage3_initialized = True

    def constrain_values(self, system, time):
        if self.left_ref is None:
            self.left_ref = system.position_collection[:, 0].copy()
            self.right_ref = system.position_collection[:, -1].copy()

        self._initialize_stage2_if_needed(system, time)
        self._initialize_stage3_if_needed(system, time)

        if time <= self.t1_end:
            a = phase_alpha(time, 0.0, self.t_pull)
            dz = 0.5 * self.total_pull_z * a

            system.position_collection[0, 0] = self.left_ref[0]
            system.position_collection[1, 0] = self.left_ref[1]
            system.position_collection[2, 0] = self.left_ref[2] - dz

            system.position_collection[0, -1] = self.right_ref[0]
            system.position_collection[1, -1] = self.right_ref[1]
            system.position_collection[2, -1] = self.right_ref[2] + dz
            return

        if time <= self.t2_end:
            a = phase_alpha(time, self.t1_end, self.t_center)
            pL = (1.0 - a) * self.pL_stage2_start + a * self.pL_stage2_target
            pR = (1.0 - a) * self.pR_stage2_start + a * self.pR_stage2_target
            system.position_collection[:, 0] = pL
            system.position_collection[:, -1] = pR
            return

        system.position_collection[:, 0] = self.pL_stage2_target
        system.position_collection[:, -1] = self.pR_stage2_target

        a = phase_alpha(time, self.t2_end, self.t_align)
        DL_now, _, _ = interpolate_director_rows(
            self.DL_stage3_start, self.DL_stage3_target, a
        )
        DR_now, _, _ = interpolate_director_rows(
            self.DR_stage3_start, self.DR_stage3_target, a
        )

        system.director_collection[:, :, 0] = DL_now
        system.director_collection[:, :, -1] = DR_now

    def constrain_rates(self, system, time):
        if self.left_ref is None:
            return

        if time <= self.t1_end:
            adot = phase_alpha_dot(time, 0.0, self.t_pull)
            vz = 0.5 * self.total_pull_z * adot

            system.velocity_collection[0, 0] = 0.0
            system.velocity_collection[1, 0] = 0.0
            system.velocity_collection[2, 0] = -vz

            system.velocity_collection[0, -1] = 0.0
            system.velocity_collection[1, -1] = 0.0
            system.velocity_collection[2, -1] = +vz

            system.omega_collection[:, 0] = 0.0
            system.omega_collection[:, -1] = 0.0
            return

        if time <= self.t2_end:
            adot = phase_alpha_dot(time, self.t1_end, self.t_center)

            vL = adot * (self.pL_stage2_target - self.pL_stage2_start)
            vR = adot * (self.pR_stage2_target - self.pR_stage2_start)

            system.velocity_collection[:, 0] = vL
            system.velocity_collection[:, -1] = vR

            system.omega_collection[:, 0] = 0.0
            system.omega_collection[:, -1] = 0.0
            return

        system.velocity_collection[:, 0] = 0.0
        system.velocity_collection[:, -1] = 0.0

        adot = phase_alpha_dot(time, self.t2_end, self.t_align)
        if time <= self.t3_end:
            system.omega_collection[:, 0] = adot * self.theta_left * self.axis_left
            system.omega_collection[:, -1] = adot * self.theta_right * self.axis_right
        else:
            system.omega_collection[:, 0] = 0.0
            system.omega_collection[:, -1] = 0.0


class AxialSeparationBC(ConstraintBase):
    def __init__(
        self,
        p0_init,
        pL_init,
        D0,
        DL,
        axis_hat,
        target_span,
        t_move,
        _system=None,
        **kwargs,
    ):
        super().__init__(_system=_system)

        self.p0_init = np.asarray(p0_init, dtype=float).copy()
        self.pL_init = np.asarray(pL_init, dtype=float).copy()
        self.D0 = np.asarray(D0, dtype=float).copy()
        self.DL = np.asarray(DL, dtype=float).copy()
        self.axis_hat = normalize(axis_hat)

        self.initial_span = float(np.dot(self.pL_init - self.p0_init, self.axis_hat))
        self.target_span = float(target_span)
        self.delta_span = self.initial_span - self.target_span
        self.t_move = float(t_move)

    def _alpha(self, time):
        if self.t_move <= 0.0:
            return 1.0
        return smoothstep(time / self.t_move)

    def _alphadot(self, time):
        if self.t_move <= 0.0:
            return 0.0
        return smoothstep_derivative(time / self.t_move) / self.t_move

    def constrain_values(self, system, time):
        a = self._alpha(time)
        shift = 0.5 * self.delta_span * a * self.axis_hat
        system.position_collection[:, 0] = self.p0_init + shift
        system.position_collection[:, -1] = self.pL_init - shift
        system.director_collection[:, :, 0] = self.D0
        system.director_collection[:, :, -1] = self.DL

    def constrain_rates(self, system, time):
        adot = self._alphadot(time)
        vel = 0.5 * self.delta_span * adot * self.axis_hat
        system.velocity_collection[:, 0] = vel
        system.velocity_collection[:, -1] = -vel
        system.omega_collection[:, 0] = 0.0
        system.omega_collection[:, -1] = 0.0


class TwistClampsAboutZAxisBC(ConstraintBase):
    def __init__(
        self,
        end_positions,
        end_directors,
        phi_extra_final,
        t_extra,
        z_axis=np.array([0.0, 0.0, 1.0]),
        _system=None,
        **kwargs,
    ):
        super().__init__(_system=_system)

        self.p0_init = np.asarray(end_positions[:, 0], dtype=float).copy()
        self.pL_init = np.asarray(end_positions[:, 1], dtype=float).copy()

        self.D0_init = np.asarray(end_directors[:, :, 0], dtype=float).copy()
        self.DL_init = np.asarray(end_directors[:, :, 1], dtype=float).copy()

        self.phi_extra_final = float(phi_extra_final)
        self.t_extra = float(t_extra)
        self.z_axis = normalize(z_axis)

    def phi(self, time):
        if self.t_extra <= 0.0:
            return self.phi_extra_final
        x = time / self.t_extra
        return smoothstep(x) * self.phi_extra_final

    def phidot(self, time):
        if self.t_extra <= 0.0:
            return 0.0
        x = time / self.t_extra
        return (smoothstep_derivative(x) / self.t_extra) * self.phi_extra_final

    def _angles(self, time):
        phi_now = self.phi(time)
        return +0.5 * phi_now, -0.5 * phi_now

    def constrain_values(self, system, time):
        a0, aL = self._angles(time)

        R0 = rot_about_axis(self.z_axis, a0)
        RL = rot_about_axis(self.z_axis, aL)

        system.position_collection[:, 0] = self.p0_init
        system.position_collection[:, -1] = self.pL_init

        system.director_collection[:, :, 0] = self.D0_init @ R0.T
        system.director_collection[:, :, -1] = self.DL_init @ RL.T

    def constrain_rates(self, system, time):
        w = self.phidot(time)

        system.velocity_collection[:, 0] = 0.0
        system.velocity_collection[:, -1] = 0.0

        system.omega_collection[:, 0] = (+0.5 * w) * self.z_axis
        system.omega_collection[:, -1] = (-0.5 * w) * self.z_axis


# -----------------------------------------------------------------------------
# Torque-like quantities
# -----------------------------------------------------------------------------
def get_torque_like_array(rod, torque_mode="internal_couple"):
    mode = str(torque_mode).strip().lower()

    if mode == "internal_couple":
        for name in ("internal_couple", "internal_couples"):
            if hasattr(rod, name):
                return getattr(rod, name)
        return None

    if mode == "internal_torques":
        for name in ("internal_torques", "internal_torque"):
            if hasattr(rod, name):
                return getattr(rod, name)
        return None

    raise ValueError(
        f"Unknown torque_mode={torque_mode!r}. "
        "Use 'internal_couple' or 'internal_torques'."
    )


def compute_projected_torque_like_samples(
    rod,
    axis_hat,
    director_mapping_mode,
    torque_mode="internal_couple",
):
    T = get_torque_like_array(rod, torque_mode=torque_mode)
    if T is None:
        return np.empty((0,), dtype=float)

    T = np.asarray(T)
    if T.ndim != 2 or T.shape[0] != 3:
        return np.empty((0,), dtype=float)

    axis_hat = normalize(axis_hat)
    n_torque = T.shape[1]
    n_dir = rod.director_collection.shape[2]
    n_use = min(n_torque, n_dir)

    vals = np.empty(n_use, dtype=float)
    for k in range(n_use):
        Dk = rod.director_collection[:, :, k]
        mk_lab = material_vec_to_lab(Dk, T[:, k], director_mapping_mode)
        vals[k] = float(np.dot(mk_lab, axis_hat))

    return vals


def compute_rod_torque_like_at_offset(
    rod,
    axis_hat,
    director_mapping_mode,
    offset_samples=4,
    torque_mode="internal_couple",
):
    vals = compute_projected_torque_like_samples(
        rod,
        axis_hat,
        director_mapping_mode,
        torque_mode=torque_mode,
    )
    if vals.size == 0:
        return np.nan, np.nan

    k = int(offset_samples)
    k = max(0, min(k, vals.size - 1))

    left_val = float(vals[k])
    right_val = float(vals[-1 - k])
    return left_val, right_val


# -----------------------------------------------------------------------------
# Kirchhoff quantities
# -----------------------------------------------------------------------------
def _batch_matvec_3x3(A, x):
    return np.einsum("ijk,jk->ik", A, x)


def constitutive_internal_force_and_couple(rod):
    sigma_diff = np.asarray(rod.sigma) - np.asarray(rod.rest_sigma)
    kappa_diff = np.asarray(rod.kappa) - np.asarray(rod.rest_kappa)

    n_const = _batch_matvec_3x3(np.asarray(rod.shear_matrix), sigma_diff)
    m_const = _batch_matvec_3x3(np.asarray(rod.bend_matrix), kappa_diff)
    return n_const, m_const


def compute_kirchhoff_metrics(rod, eps_force=1e-14):
    n_const_elem, m_const_voro = constitutive_internal_force_and_couple(rod)

    n_node = element_field_to_nodes(n_const_elem)
    m_node = voronoi_field_to_nodes(m_const_voro)

    I2_node = np.linalg.norm(n_node, axis=0)
    I3_node = np.einsum("in,in->n", n_node, m_node)

    ratio_node = np.full(I2_node.shape, np.nan, dtype=float)
    good = I2_node > eps_force
    ratio_node[good] = I3_node[good] / I2_node[good]

    mean_I2 = float(np.nanmean(I2_node))
    mean_I3_over_I2 = float(np.nanmean(ratio_node))
    return mean_I2, mean_I3_over_I2, I2_node, I3_node, ratio_node


def compute_mean_I2_over_I3(I2_node, I3_node, eps=1e-14):
    out = np.full(I2_node.shape, np.nan, dtype=float)
    good = np.abs(I3_node) > eps
    out[good] = I2_node[good] / I3_node[good]
    return float(np.nanmean(out)), out


# -----------------------------------------------------------------------------
# Axis fitting
# -----------------------------------------------------------------------------
def fit_helix_axis_and_radius(pos3xn, d3n=None):
    pts = pos3xn.T.copy()
    centroid = np.mean(pts, axis=0)

    X = pts - centroid
    C = X.T @ X / max(1, len(pts) - 1)
    evals, evecs = np.linalg.eigh(C)
    axis_hat = evecs[:, np.argmax(evals)]
    axis_hat = normalize(axis_hat)

    if d3n is not None:
        avg_tan = normalize(np.mean(d3n, axis=1))
        if np.dot(axis_hat, avg_tan) < 0:
            axis_hat = -axis_hat
    else:
        chord = normalize(pts[-1] - pts[0])
        if np.dot(axis_hat, chord) < 0:
            axis_hat = -axis_hat

    radial = X - np.outer(X @ axis_hat, axis_hat)
    radii = np.linalg.norm(radial, axis=1)
    radius = float(np.mean(radii))

    return centroid, axis_hat, radius, radii


# -----------------------------------------------------------------------------
# Link / optional Tw Wr
# -----------------------------------------------------------------------------
class LinkTracker:
    def __init__(self, rod, director_mapping_mode, initial_link):
        self.director_mapping_mode = director_mapping_mode
        self.link = float(initial_link)

        d1n, _, d3n = node_frames_from_directors(rod.director_collection, director_mapping_mode)

        self.prev_t_left = normalize(d3n[:, 0])
        self.prev_t_right = normalize(d3n[:, -1])

        self.prev_d1_left = normalize(d1n[:, 0] - np.dot(d1n[:, 0], self.prev_t_left) * self.prev_t_left)
        self.prev_d1_right = normalize(d1n[:, -1] - np.dot(d1n[:, -1], self.prev_t_right) * self.prev_t_right)

        self.history = []

    def adopt_state_without_increment(self, rod):
        d1n, _, d3n = node_frames_from_directors(rod.director_collection, self.director_mapping_mode)

        self.prev_t_left = normalize(d3n[:, 0])
        self.prev_t_right = normalize(d3n[:, -1])

        self.prev_d1_left = normalize(d1n[:, 0] - np.dot(d1n[:, 0], self.prev_t_left) * self.prev_t_left)
        self.prev_d1_right = normalize(d1n[:, -1] - np.dot(d1n[:, -1], self.prev_t_right) * self.prev_t_right)

    def update_measured(self, rod, time, stage_name):
        d1n, _, d3n = node_frames_from_directors(rod.director_collection, self.director_mapping_mode)

        curr_t_left = normalize(d3n[:, 0])
        curr_t_right = normalize(d3n[:, -1])

        curr_d1_left = normalize(d1n[:, 0] - np.dot(d1n[:, 0], curr_t_left) * curr_t_left)
        curr_d1_right = normalize(d1n[:, -1] - np.dot(d1n[:, -1], curr_t_right) * curr_t_right)

        dpsi_left = transported_spin_increment(
            self.prev_t_left, self.prev_d1_left, curr_t_left, curr_d1_left
        )
        dpsi_right = transported_spin_increment(
            self.prev_t_right, self.prev_d1_right, curr_t_right, curr_d1_right
        )

        self.link -= (dpsi_left + dpsi_right) / (2.0 * np.pi)

        self.history.append(
            {
                "time": float(time),
                "stage": str(stage_name),
                "dpsi_left": float(dpsi_left),
                "dpsi_right": float(dpsi_right),
                "link": float(self.link),
            }
        )

        self.prev_t_left = curr_t_left
        self.prev_t_right = curr_t_right
        self.prev_d1_left = curr_d1_left
        self.prev_d1_right = curr_d1_right

        return self.link

    def update_direct(self, time, stage_name, link_value):
        self.link = float(link_value)
        self.history.append(
            {
                "time": float(time),
                "stage": str(stage_name),
                "dpsi_left": np.nan,
                "dpsi_right": np.nan,
                "link": float(self.link),
            }
        )
        return self.link


def compute_tw_wr_lk_closure_optional(rod, director_mapping_mode, compute_tw_wr=False):
    if not compute_tw_wr:
        return np.nan, np.nan, np.nan
    return np.nan, np.nan, np.nan


# -----------------------------------------------------------------------------
# Cloning
# -----------------------------------------------------------------------------
def copy_if_present(dst, src, name, slicer):
    if hasattr(dst, name) and hasattr(src, name):
        getattr(dst, name)[...] = slicer(getattr(src, name))


def extract_and_build_subrod(old_rod, e0, e1, radius, rho, E, nu):
    node0 = e0
    node1 = e1

    pos_seg = old_rod.position_collection[:, node0:node1 + 1].copy()

    start_new = pos_seg[:, 0].copy()
    end_new = pos_seg[:, -1].copy()
    chord = end_new - start_new
    L_new = np.linalg.norm(chord)
    direction_new = normalize(chord)
    normal_new = any_perpendicular(direction_new)

    n_new = e1 - e0
    G = E / (2.0 * (1.0 + nu))

    try:
        new_rod = CosseratRod.straight_rod(
            n_new,
            start_new,
            direction_new,
            normal_new,
            L_new,
            radius,
            rho,
            youngs_modulus=E,
            shear_modulus=G,
        )
    except TypeError:
        try:
            new_rod = CosseratRod.straight_rod(
                n_new,
                start_new,
                direction_new,
                normal_new,
                L_new,
                radius,
                rho,
                youngs_modulus=E,
                poisson_ratio=nu,
            )
        except TypeError:
            new_rod = CosseratRod.straight_rod(
                n_new,
                start_new,
                direction_new,
                normal_new,
                L_new,
                radius,
                rho,
                youngs_modulus=E,
            )

    new_rod.position_collection[...] = old_rod.position_collection[:, node0:node1 + 1]

    if hasattr(new_rod, "velocity_collection") and hasattr(old_rod, "velocity_collection"):
        new_rod.velocity_collection[...] = old_rod.velocity_collection[:, node0:node1 + 1]
    if hasattr(new_rod, "acceleration_collection"):
        new_rod.acceleration_collection[...] = 0.0
    if hasattr(new_rod, "mass") and hasattr(old_rod, "mass"):
        new_rod.mass[...] = old_rod.mass[node0:node1 + 1]

    new_rod.director_collection[...] = old_rod.director_collection[:, :, e0:e1]
    if hasattr(new_rod, "omega_collection") and hasattr(old_rod, "omega_collection"):
        new_rod.omega_collection[...] = old_rod.omega_collection[:, e0:e1]
    if hasattr(new_rod, "alpha_collection"):
        new_rod.alpha_collection[...] = 0.0

    copy_if_present(new_rod, old_rod, "rest_lengths", lambda a: a[e0:e1])
    copy_if_present(new_rod, old_rod, "lengths", lambda a: a[e0:e1])
    copy_if_present(new_rod, old_rod, "radius", lambda a: a[e0:e1])
    copy_if_present(new_rod, old_rod, "tangents", lambda a: a[:, e0:e1])
    copy_if_present(new_rod, old_rod, "dilatation", lambda a: a[e0:e1])
    copy_if_present(new_rod, old_rod, "sigma", lambda a: a[:, e0:e1])
    copy_if_present(new_rod, old_rod, "rest_sigma", lambda a: a[:, e0:e1])
    copy_if_present(new_rod, old_rod, "shear_matrix", lambda a: a[:, :, e0:e1])
    copy_if_present(new_rod, old_rod, "inv_shear_matrix", lambda a: a[:, :, e0:e1])

    copy_if_present(new_rod, old_rod, "rest_voronoi_lengths", lambda a: a[e0:e1 - 1])
    copy_if_present(new_rod, old_rod, "voronoi_lengths", lambda a: a[e0:e1 - 1])
    copy_if_present(new_rod, old_rod, "voronoi_dilatation", lambda a: a[e0:e1 - 1])
    copy_if_present(new_rod, old_rod, "kappa", lambda a: a[:, e0:e1 - 1])
    copy_if_present(new_rod, old_rod, "rest_kappa", lambda a: a[:, e0:e1 - 1])
    copy_if_present(new_rod, old_rod, "bend_matrix", lambda a: a[:, :, e0:e1 - 1])
    copy_if_present(new_rod, old_rod, "inv_bend_matrix", lambda a: a[:, :, e0:e1 - 1])

    for name in (
        "external_forces",
        "internal_forces",
        "external_torques",
        "internal_torques",
        "external_couple",
        "internal_couple",
        "external_couples",
        "internal_couples",
    ):
        if hasattr(new_rod, name):
            getattr(new_rod, name)[...] = 0.0

    return new_rod


def clone_full_rod(old_rod, radius, rho, E, nu):
    n_elems = old_rod.director_collection.shape[2]
    return extract_and_build_subrod(old_rod, 0, n_elems, radius, rho, E, nu)


# -----------------------------------------------------------------------------
# Tube mesh utilities
# -----------------------------------------------------------------------------
def build_tube_topology(n_nodes, n_theta):
    def idx(i, j):
        return i * n_theta + j

    faces = []
    for i in range(n_nodes - 1):
        for j in range(n_theta):
            jp = (j + 1) % n_theta
            a = idx(i, j)
            b = idx(i, jp)
            c = idx(i + 1, jp)
            d = idx(i + 1, j)
            faces.append([a, b, c])
            faces.append([a, c, d])
    return np.asarray(faces, dtype=int)


def tube_vertices_from_frames(pos3xn, director_collection, radius, thetas, director_mapping_mode):
    d1n, d2n, _ = node_frames_from_directors(director_collection, director_mapping_mode)
    n_nodes = pos3xn.shape[1]

    ct = np.cos(thetas)[None, :]
    st = np.sin(thetas)[None, :]

    C = pos3xn[:, :, None]
    d1 = d1n[:, :, None]
    d2 = d2n[:, :, None]

    P = C + radius * (d1 * ct + d2 * st)
    V = np.transpose(P, (1, 2, 0)).reshape(n_nodes * thetas.size, 3)
    return V


def shiny_facecolors(
    V,
    faces_idx,
    light_dir=(0.35, 0.85, 0.4),
    ambient=0.10,
    diffuse_w=0.60,
    specular_w=0.65,
    shininess=35,
):
    light = np.asarray(light_dir, dtype=float)
    light /= (np.linalg.norm(light) + 1e-14)

    F = V[faces_idx]
    e1 = F[:, 1, :] - F[:, 0, :]
    e2 = F[:, 2, :] - F[:, 0, :]
    n = np.cross(e1, e2)
    n /= (np.linalg.norm(n, axis=1)[:, None] + 1e-14)

    d = np.clip(n @ light, 0.0, 1.0)
    s = d ** shininess
    inten = np.clip(ambient + diffuse_w * d + specular_w * s, 0.0, 1.0)
    rgba = np.column_stack([inten, inten, inten, np.ones_like(inten)])
    return rgba


# -----------------------------------------------------------------------------
# Optional plotting helper
# -----------------------------------------------------------------------------
class LiveRodPlotter:
    def __init__(self, radius, director_mapping_mode, title="Live rod", every=1000):
        self.radius = float(radius)
        self.director_mapping_mode = director_mapping_mode
        self.title = title
        self.every = int(max(1, every))

        self.fig = None
        self.ax = None
        self.line = None
        self.ends = None
        self.axis_line = None

        self.initialized = False
        self.last_draw_step = -1

    def init(self, pos):
        plt.ion()
        self.fig = plt.figure(figsize=(7, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title(self.title)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        self.line, = self.ax.plot(pos[0], pos[1], pos[2], lw=2)
        self.ends = self.ax.scatter(
            [pos[0, 0], pos[0, -1]],
            [pos[1, 0], pos[1, -1]],
            [pos[2, 0], pos[2, -1]],
            s=45,
        )
        self.axis_line, = self.ax.plot([], [], [], "--", lw=2)

        set_axes_equal_3d(self.ax, pos)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.initialized = True

    def update(self, pos, step=None, axis_origin=None, axis_hat=None, force=False):
        if (step is not None) and (not force):
            if step == self.last_draw_step:
                return
            if (step % self.every) != 0:
                return

        if not self.initialized:
            self.init(pos)

        self.line.set_data(pos[0], pos[1])
        self.line.set_3d_properties(pos[2])

        self.ends.remove()
        self.ends = self.ax.scatter(
            [pos[0, 0], pos[0, -1]],
            [pos[1, 0], pos[1, -1]],
            [pos[2, 0], pos[2, -1]],
            s=45,
            color="crimson",
        )

        if axis_origin is not None and axis_hat is not None:
            pts = pos.T
            axis_hat = normalize(axis_hat)
            ss = (pts - axis_origin[None, :]) @ axis_hat
            smin = np.min(ss)
            smax = np.max(ss)
            a0 = axis_origin + smin * axis_hat
            a1 = axis_origin + smax * axis_hat
            self.axis_line.set_data([a0[0], a1[0]], [a0[1], a1[1]])
            self.axis_line.set_3d_properties([a0[2], a1[2]])
        else:
            self.axis_line.set_data([], [])
            self.axis_line.set_3d_properties([])

        set_axes_equal_3d(self.ax, pos)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
        self.last_draw_step = -1 if step is None else step

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.show()


# -----------------------------------------------------------------------------
# GIF export for translation stage
# -----------------------------------------------------------------------------
def export_translation_gif(
    frames_pos,
    frames_directors,
    frames_z,
    frames_I2_mean,
    frames_I3_over_I2_mean,
    frames_time,
    axis_origin,
    axis_hat,
    radius,
    gif_path,
    director_mapping_mode,
    fps=12,
    dpi=100,
):
    n_frames = len(frames_pos)
    n_elems = frames_directors[0].shape[2]
    n_nodes = n_elems + 1

    n_theta = 30
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    faces_idx = build_tube_topology(n_nodes, n_theta)

    V0 = tube_vertices_from_frames(frames_pos[0], frames_directors[0], radius, thetas, director_mapping_mode)
    FC0 = shiny_facecolors(V0, faces_idx)

    fig = plt.figure(figsize=(12, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axQ = fig.add_subplot(1, 2, 2)

    ax3d.set_axis_off()
    tube = Poly3DCollection(V0[faces_idx], facecolors=FC0, linewidths=0.0)
    tube.set_edgecolor((0, 0, 0, 0))
    ax3d.add_collection3d(tube)

    all_pts = np.concatenate([p.T for p in frames_pos], axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = max(maxs - mins) + 10.0 * radius
    half = 0.48 * span

    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)
    try:
        ax3d.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax3d.view_init(elev=18, azim=155)

    (axis_line,) = ax3d.plot([0, 0], [0, 0], [0, 0], "--", lw=2, color="tab:green")

    axQ.set_xlabel("z")
    axQ.set_ylabel("quantity")
    axQ.grid(True)
    axQ.plot(frames_z, frames_I2_mean, lw=2.5, label=r"$\langle I_2 \rangle$")
    axQ.plot(frames_z, frames_I3_over_I2_mean, lw=2.5, label=r"$\langle I_3/I_2 \rangle$")
    (p1,) = axQ.plot([frames_z[0]], [frames_I2_mean[0]], "o", markersize=7)
    (p2,) = axQ.plot([frames_z[0]], [frames_I3_over_I2_mean[0]], "o", markersize=7)
    axQ.legend(loc="best")

    zmin, zmax = float(np.min(frames_z)), float(np.max(frames_z))
    qall = np.concatenate([frames_I2_mean, frames_I3_over_I2_mean])
    finite = np.isfinite(qall)
    qmin = float(np.min(qall[finite])) if np.any(finite) else -1.0
    qmax = float(np.max(qall[finite])) if np.any(finite) else 1.0
    dz = max(1e-6, zmax - zmin)
    dq = max(1e-9, qmax - qmin)
    axQ.set_xlim(zmin - 0.05 * dz, zmax + 0.05 * dz)
    axQ.set_ylim(qmin - 0.12 * dq, qmax + 0.12 * dq)

    time_text = fig.text(0.5, 0.02, "", ha="center")

    def draw_frame(frame_idx):
        pos = frames_pos[frame_idx]
        D = frames_directors[frame_idx]

        V = tube_vertices_from_frames(pos, D, radius, thetas, director_mapping_mode)
        FC = shiny_facecolors(V, faces_idx)
        tube.set_verts(V[faces_idx])
        tube.set_facecolor(FC)

        pts = pos.T
        ss = (pts - axis_origin[None, :]) @ axis_hat
        a0 = axis_origin + np.min(ss) * axis_hat
        a1 = axis_origin + np.max(ss) * axis_hat
        axis_line.set_data([a0[0], a1[0]], [a0[1], a1[1]])
        axis_line.set_3d_properties([a0[2], a1[2]])

        p1.set_data([frames_z[frame_idx]], [frames_I2_mean[frame_idx]])
        p2.set_data([frames_z[frame_idx]], [frames_I3_over_I2_mean[frame_idx]])

        time_text.set_text(
            f"t={frames_time[frame_idx]:.3f}s   "
            f"z={frames_z[frame_idx]:.4f}   "
            f"<I2>={frames_I2_mean[frame_idx]:.4e}   "
            f"<I3/I2>={frames_I3_over_I2_mean[frame_idx]:.4e}"
        )

    plt.tight_layout()
    print(f"Saving GIF: {gif_path}  ({n_frames} frames @ {fps} fps)")
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, gif_path, dpi=dpi):
        for i in progress_iter(range(n_frames), total=n_frames, desc="Writing GIF"):
            draw_frame(i)
            writer.grab_frame()

    plt.close(fig)
    print("GIF saved.")


# -----------------------------------------------------------------------------
# Export helpers
# -----------------------------------------------------------------------------
def write_parameters_txt(path, params_dict):
    with open(path, "w", encoding="utf-8") as f:
        for k, v in params_dict.items():
            f.write(f"{k}: {v}\n")


def write_link_history_txt(path, link_tracker):
    with open(path, "w", encoding="utf-8") as f:
        f.write("time stage dpsi_left dpsi_right link_iterative\n")
        for row in link_tracker.history:
            f.write(
                f"{row['time']:.16e} {row['stage']} "
                f"{row['dpsi_left']:.16e} {row['dpsi_right']:.16e} {row['link']:.16e}\n"
            )


def export_frame_table_txt(
    path,
    rod,
    director_mapping_mode,
    link_value,
    tw_value,
    wr_value,
    lk_closure_value,
    stage_name,
    time_value,
    imposed_rotation,
):
    pos = np.asarray(rod.position_collection)
    s = cumulative_arclength_from_nodes(pos)

    kappa_node = voronoi_field_to_nodes(np.asarray(rod.kappa))
    rest_kappa_node = voronoi_field_to_nodes(np.asarray(rod.rest_kappa))

    sigma_node = element_field_to_nodes(np.asarray(rod.sigma))
    rest_sigma_node = element_field_to_nodes(np.asarray(rod.rest_sigma))

    n_const_elem, m_const_voro = constitutive_internal_force_and_couple(rod)
    n_const_node = element_field_to_nodes(n_const_elem)
    m_const_node = voronoi_field_to_nodes(m_const_voro)

    n_const_lab_node = node_material_to_lab(
        n_const_node, rod.director_collection, director_mapping_mode
    )
    m_const_lab_node = node_material_to_lab(
        m_const_node, rod.director_collection, director_mapping_mode
    )

    mean_I2, mean_I3_over_I2, I2_node, I3_node, ratio_node = compute_kirchhoff_metrics(rod)
    mean_I2_over_I3, tau_inv_node = compute_mean_I2_over_I3(I2_node, I3_node)

    N = pos.shape[1]

    table = np.column_stack(
        [
            s,
            pos[0], pos[1], pos[2],
            kappa_node[0], kappa_node[1], kappa_node[2],
            rest_kappa_node[0], rest_kappa_node[1], rest_kappa_node[2],
            sigma_node[0], sigma_node[1], sigma_node[2],
            rest_sigma_node[0], rest_sigma_node[1], rest_sigma_node[2],
            n_const_node[0], n_const_node[1], n_const_node[2],
            n_const_lab_node[0], n_const_lab_node[1], n_const_lab_node[2],
            m_const_node[0], m_const_node[1], m_const_node[2],
            m_const_lab_node[0], m_const_lab_node[1], m_const_lab_node[2],
            I2_node,
            I3_node,
            ratio_node,
            tau_inv_node,
            np.full(N, link_value),
            np.full(N, tw_value),
            np.full(N, wr_value),
            np.full(N, lk_closure_value),
            np.full(N, mean_I2),
            np.full(N, mean_I3_over_I2),
            np.full(N, mean_I2_over_I3),
            np.full(N, time_value),
            np.full(N, imposed_rotation),
        ]
    )

    header = (
        "s "
        "x y z "
        "kappa_1 kappa_2 kappa_3 "
        "kappa0_1 kappa0_2 kappa0_3 "
        "sigma_1 sigma_2 sigma_3 "
        "sigma0_1 sigma0_2 sigma0_3 "
        "n_mat_1 n_mat_2 n_mat_3 "
        "n_lab_x n_lab_y n_lab_z "
        "m_mat_1 m_mat_2 m_mat_3 "
        "m_lab_x m_lab_y m_lab_z "
        "I2 I3 I3_over_I2 I2_over_I3 "
        "link_iterative Tw Wr Lk_closure "
        "mean_I2 mean_I3_over_I2 mean_I2_over_I3 "
        "time imposed_rotation"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# stage: {stage_name}\n")
        f.write(f"# time: {time_value:.16e}\n")
        f.write(f"# link_iterative: {link_value:.16e}\n")
        f.write(f"# Tw: {tw_value:.16e}\n")
        f.write(f"# Wr: {wr_value:.16e}\n")
        f.write(f"# Lk_closure: {lk_closure_value:.16e}\n")
        f.write(f"# imposed_rotation: {imposed_rotation:.16e}\n")
        np.savetxt(f, table, header=header, comments="")


def export_unwinding_to_target_summary_txt(
    path,
    frames_t,
    frames_phi,
    frames_link,
    frames_I2_mean,
    frames_I3_over_I2_mean,
):
    data = np.column_stack(
        [
            frames_t,
            frames_phi,
            frames_link,
            frames_I2_mean,
            frames_I3_over_I2_mean,
        ]
    )
    header = "time phi_extra link_iterative mean_I2 mean_I3_over_I2"
    np.savetxt(path, data, header=header, comments="")


def export_translation_summary_txt(
    path,
    frames_t,
    frames_z,
    frames_link,
    frames_phi_const,
    frames_I2_mean,
    frames_I3_over_I2_mean,
):
    data = np.column_stack(
        [
            frames_t,
            frames_z,
            frames_link,
            frames_phi_const,
            frames_I2_mean,
            frames_I3_over_I2_mean,
        ]
    )
    header = "time z_value link_iterative imposed_rotation mean_I2 mean_I3_over_I2"
    np.savetxt(path, data, header=header, comments="")


def export_scheduled_stage_frames(
    rod,
    stage_name,
    time_value,
    schedule,
    next_idx,
    frames_dir_path,
    director_mapping_mode,
    link_value,
    imposed_rotation,
    compute_tw_wr=False,
):
    tw_value, wr_value, lk_closure_value = compute_tw_wr_lk_closure_optional(
        rod, director_mapping_mode, compute_tw_wr=compute_tw_wr
    )

    while next_idx < len(schedule) and time_value >= schedule[next_idx] - 1.0e-14:
        frame_path = frames_dir_path / f"{stage_name}_frame_{next_idx:05d}.txt"
        export_frame_table_txt(
            path=frame_path,
            rod=rod,
            director_mapping_mode=director_mapping_mode,
            link_value=link_value,
            tw_value=tw_value,
            wr_value=wr_value,
            lk_closure_value=lk_closure_value,
            stage_name=stage_name,
            time_value=float(time_value),
            imposed_rotation=float(imposed_rotation),
        )
        next_idx += 1
    return next_idx


# -----------------------------------------------------------------------------
# Sweep helpers
# -----------------------------------------------------------------------------
def link_to_folder_name(target_link):
    return f"link_{target_link:0.6f}".replace(".", "p")


def build_link_sweep(link_min=0.0, link_max=-7.0, n_points=20):
    if n_points == 1:
        return np.asarray([link_min], dtype=float)
    return np.asarray(np.linspace(link_min, link_max, n_points), dtype=float)


# -----------------------------------------------------------------------------
# Single experiment runner
# -----------------------------------------------------------------------------
def run_single_experiment(args, target_link_value, output_dir):
    COMPUTE_TW_WR = False

    SAVE_GIF = True
    FPS = 12
    DPI = 80
    sim_dt_per_frame = float(args.sim_dt_per_frame)

    TORQUE_MODE = str(args.torque_mode)
    TORQUE_OFFSET_SAMPLES = int(args.torque_offset_samples)

    LIVE_PLOT = False
    LIVE_PLOT_EVERY_STEPS = 4500
    PLOT_FINAL = False

    n_elems = 120
    n_turns = 8
    circle_radius = 0.0039
    rod_radius = 0.001

    density = 500.0
    youngs_modulus = 3.0e4
    poisson_ratio = 0.4

    z_perturbation = 1.0e-5

    lambda_bend = float(args.lambda_bend)
    gamma_twist = float(args.gamma_twist)

    z_init = float(args.z_init)
    z_final = float(args.z_final)
    target_link = float(target_link_value)

    base_length_est = 2.0 * np.pi * circle_radius * n_turns
    total_pull_z = 0.8 * base_length_est

    t_pull = 4.2
    t_center = 1.2
    t_align = 1.2
    t_hold_gen = 0.6

    t_move_to_z_init = 1.0
    t_hold_after_move = 2.5

    EXTRA_UNWIND_TURNS = float(args.extra_unwind_turns)
    phi_extra_final = 2.0 * np.pi * EXTRA_UNWIND_TURNS
    t_extra = 2 * EXTRA_UNWIND_TURNS

    t_translate = float(args.translation_time)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir_path = output_dir / "frames_txt"
    frames_dir_path.mkdir(parents=True, exist_ok=True)

    gif_path = str(output_dir / "translation_stage.gif")
    params_path = output_dir / "simulation_parameters.txt"
    link_history_path = output_dir / "link_history_all_steps.txt"
    unwind_summary_path = output_dir / "unwinding_to_target_summary.txt"
    translation_summary_path = output_dir / "translation_summary.txt"

    dl = base_length_est / n_elems
    dt = conservative_dt(dl, rod_radius, density, youngs_modulus, safety=0.15)

    damp_fac_gen = np.exp(np.log(0.999) * 150 / n_elems)
    damp_fac_post = np.exp(np.log(0.9995) * 150 / n_elems)

    print("\n" + "=" * 80)
    print(f"Running target link = {target_link:.8f}")
    print(f"z_init = {z_init:.8f}, z_final = {z_final:.8f}")
    print(f"Constant gamma = {gamma_twist:.8f}")
    print(f"Output folder  = {output_dir}")
    print("=" * 80)

    position, directors, base_length = build_multiwrap_circle_geometry(
        n_elems=n_elems,
        n_turns=n_turns,
        circle_radius=circle_radius,
        z_perturbation=z_perturbation,
    )

    G = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    try:
        rod0 = CosseratRod.straight_rod(
            n_elements=n_elems,
            start=position[:, 0],
            direction=np.array([1.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            base_length=base_length,
            base_radius=rod_radius,
            density=density,
            youngs_modulus=youngs_modulus,
            shear_modulus=G,
            position=position,
            directors=directors,
        )
    except TypeError:
        try:
            rod0 = CosseratRod.straight_rod(
                n_elements=n_elems,
                start=position[:, 0],
                direction=np.array([1.0, 0.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                base_length=base_length,
                base_radius=rod_radius,
                density=density,
                youngs_modulus=youngs_modulus,
                poisson_ratio=poisson_ratio,
                position=position,
                directors=directors,
            )
        except TypeError:
            rod0 = CosseratRod.straight_rod(
                n_elements=n_elems,
                start=position[:, 0],
                direction=np.array([1.0, 0.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                base_length=base_length,
                base_radius=rod_radius,
                density=density,
                youngs_modulus=youngs_modulus,
                position=position,
                directors=directors,
            )

    make_current_configuration_rest_state(rod0)
    apply_curvature_aligned_anisotropic_rigidity(
        rod0,
        lambda_bend=lambda_bend,
        gamma_twist=gamma_twist,
    )

    director_mapping_mode = detect_director_mapping(rod0, verbose=True)

    link_tracker = LinkTracker(
        rod=rod0,
        director_mapping_mode=director_mapping_mode,
        initial_link=float(n_turns),
    )

    write_parameters_txt(
        params_path,
        {
            "n_elems": n_elems,
            "n_turns_initial": n_turns,
            "circle_radius": circle_radius,
            "rod_radius": rod_radius,
            "density": density,
            "youngs_modulus": youngs_modulus,
            "poisson_ratio": poisson_ratio,
            "lambda_bend": lambda_bend,
            "gamma_twist": gamma_twist,
            "z_init": z_init,
            "z_final": z_final,
            "target_link": target_link,
            "link_min": args.link_min,
            "link_max": args.link_max,
            "link_points": args.link_points,
            "base_length_est": base_length_est,
            "base_length_actual": base_length,
            "total_pull_z": total_pull_z,
            "t_pull": t_pull,
            "t_center": t_center,
            "t_align": t_align,
            "t_hold_gen": t_hold_gen,
            "t_move_to_z_init": t_move_to_z_init,
            "t_hold_after_move": t_hold_after_move,
            "extra_unwind_turns": EXTRA_UNWIND_TURNS,
            "phi_extra_final": phi_extra_final,
            "t_extra": t_extra,
            "translation_time": t_translate,
            "dt": dt,
            "sim_dt_per_frame": sim_dt_per_frame,
            "torque_mode": TORQUE_MODE,
            "torque_offset_samples": TORQUE_OFFSET_SAMPLES,
            "output_dir": str(output_dir),
            "gif_path": gif_path,
            "compute_tw_wr": COMPUTE_TW_WR,
        },
    )

    # -------------------------------------------------------------------------
    # Stage 1-3: generation
    # -------------------------------------------------------------------------
    sim_gen = Simulator()
    sim_gen.append(rod0)
    sim_gen.constrain(rod0).using(
        PullCenterAlignClampBC,
        total_pull_z=total_pull_z,
        t_pull=t_pull,
        t_center=t_center,
        t_align=t_align,
    )
    sim_gen.add_forcing_to(rod0).using(RampGravity, g_vec=np.zeros(3), ramp_time=0.0)
    sim_gen.finalize()

    stepper_gen = PositionVerlet()
    do_step_gen, stages_and_updates_gen = extend_stepper_interface(stepper_gen, sim_gen)

    final_time_gen = t_pull + t_center + t_align + t_hold_gen
    n_steps_gen = int(np.ceil(final_time_gen / dt))

    gen_export_schedule = np.arange(0.0, final_time_gen + 1e-12, sim_dt_per_frame)
    gen_export_idx = 0

    print("Running generation stage...")
    tgen = 0.0
    for _ in progress_iter(range(n_steps_gen), total=n_steps_gen, desc="Generation"):
        tgen = do_step_gen(stepper_gen, stages_and_updates_gen, sim_gen, tgen, dt)
        rod0.velocity_collection[:, 1:-1] *= damp_fac_gen
        rod0.omega_collection[:, 1:-1] *= damp_fac_gen

        current_link = link_tracker.update_measured(rod0, time=tgen, stage_name="generation")

        gen_export_idx = export_scheduled_stage_frames(
            rod=rod0,
            stage_name="generation",
            time_value=tgen,
            schedule=gen_export_schedule,
            next_idx=gen_export_idx,
            frames_dir_path=frames_dir_path,
            director_mapping_mode=director_mapping_mode,
            link_value=current_link,
            imposed_rotation=0.0,
            compute_tw_wr=COMPUTE_TW_WR,
        )

    validate_director_mapping_on_rod(rod0, director_mapping_mode)

    # -------------------------------------------------------------------------
    # Stage 4: move to common z_init
    # -------------------------------------------------------------------------
    rod1 = clone_full_rod(rod0, rod_radius, density, youngs_modulus, poisson_ratio)
    link_tracker.adopt_state_without_increment(rod1)

    p0_move = rod1.position_collection[:, 0].copy()
    pL_move = rod1.position_collection[:, -1].copy()
    D0_move = rod1.director_collection[:, :, 0].copy()
    DL_move = rod1.director_collection[:, :, -1].copy()

    rod_arc_length = get_rod_length(rod1)
    target_span_init = z_init * rod_arc_length
    target_span_final = z_final * rod_arc_length

    z_axis = np.array([0.0, 0.0, 1.0])

    sim_move = Simulator()
    sim_move.append(rod1)
    sim_move.constrain(rod1).using(
        AxialSeparationBC,
        p0_init=p0_move,
        pL_init=pL_move,
        D0=D0_move,
        DL=DL_move,
        axis_hat=z_axis,
        target_span=target_span_init,
        t_move=t_move_to_z_init,
    )
    sim_move.add_forcing_to(rod1).using(RampGravity, g_vec=np.zeros(3), ramp_time=0.0)
    sim_move.finalize()

    stepper_move = PositionVerlet()
    do_step_move, stages_and_updates_move = extend_stepper_interface(stepper_move, sim_move)

    final_time_move = t_move_to_z_init + t_hold_after_move
    n_steps_move = int(np.ceil(final_time_move / dt))

    move_export_schedule = np.arange(0.0, final_time_move + 1e-12, sim_dt_per_frame)
    move_export_idx = 0

    print("Running move-to-z_init stage...")
    tmove = 0.0
    for _ in progress_iter(range(n_steps_move), total=n_steps_move, desc="Move to z_init"):
        tmove = do_step_move(stepper_move, stages_and_updates_move, sim_move, tmove, dt)
        rod1.velocity_collection[:, 1:-1] *= damp_fac_post
        rod1.omega_collection[:, 1:-1] *= damp_fac_post

        current_link = link_tracker.update_measured(rod1, time=tmove, stage_name="z_init")

        move_export_idx = export_scheduled_stage_frames(
            rod=rod1,
            stage_name="z_init",
            time_value=tmove,
            schedule=move_export_schedule,
            next_idx=move_export_idx,
            frames_dir_path=frames_dir_path,
            director_mapping_mode=director_mapping_mode,
            link_value=current_link,
            imposed_rotation=0.0,
            compute_tw_wr=COMPUTE_TW_WR,
        )

    _, _, d3n_move = node_frames_from_directors(rod1.director_collection, director_mapping_mode)
    axis_origin_move, axis_hat_move, helix_radius_move, _ = fit_helix_axis_and_radius(
        rod1.position_collection, d3n=d3n_move
    )

    # -------------------------------------------------------------------------
    # Stage 5: unwind until target link
    # -------------------------------------------------------------------------
    rod2 = clone_full_rod(rod1, rod_radius, density, youngs_modulus, poisson_ratio)
    link_tracker.adopt_state_without_increment(rod2)

    link_at_unwind_start = link_tracker.link
    if target_link > link_at_unwind_start + 1e-12:
        raise RuntimeError(
            f"Target link {target_link:.6f} is above starting link {link_at_unwind_start:.6f}."
        )

    end_positions_uw = np.zeros((3, 2))
    end_positions_uw[:, 0] = rod2.position_collection[:, 0]
    end_positions_uw[:, 1] = rod2.position_collection[:, -1]

    end_directors_uw = np.zeros((3, 3, 2))
    end_directors_uw[:, :, 0] = rod2.director_collection[:, :, 0]
    end_directors_uw[:, :, 1] = rod2.director_collection[:, :, -1]

    sim_uw = Simulator()
    sim_uw.append(rod2)
    sim_uw.constrain(rod2).using(
        TwistClampsAboutZAxisBC,
        end_positions=end_positions_uw,
        end_directors=end_directors_uw,
        phi_extra_final=phi_extra_final,
        t_extra=t_extra,
        z_axis=z_axis,
    )
    sim_uw.add_forcing_to(rod2).using(RampGravity, g_vec=np.zeros(3), ramp_time=0.0)
    sim_uw.finalize()

    stepper_uw = PositionVerlet()
    do_step_uw, stages_and_updates_uw = extend_stepper_interface(stepper_uw, sim_uw)

    def phi_extra_of_t(tt):
        x = tt / max(t_extra, 1e-12)
        return smoothstep(x) * phi_extra_final

    uw_times = []
    uw_phi = []
    uw_link = []
    uw_I2_mean = []
    uw_I3_over_I2_mean = []

    uw_export_idx = 0

    print("Running unwinding-to-target-link stage...")
    tuw = 0.0
    n_steps_uw = int(np.ceil(t_extra / dt))
    reached_target = False
    phi_stop = None
    t_stop = None
    link_stop = None

    for _ in progress_iter(range(n_steps_uw), total=n_steps_uw, desc="Unwind to link"):
        tuw = do_step_uw(stepper_uw, stages_and_updates_uw, sim_uw, tuw, dt)
        rod2.velocity_collection[:, 1:-1] *= damp_fac_post
        rod2.omega_collection[:, 1:-1] *= damp_fac_post

        current_phi = float(phi_extra_of_t(tuw))
        current_link = link_at_unwind_start - current_phi / (2.0 * np.pi)
        link_tracker.update_direct(tuw, "unwind_to_link", current_link)

        mean_I2, mean_I3_over_I2, _, _, _ = compute_kirchhoff_metrics(rod2)

        uw_times.append(tuw)
        uw_phi.append(current_phi)
        uw_link.append(current_link)
        uw_I2_mean.append(mean_I2)
        uw_I3_over_I2_mean.append(mean_I3_over_I2)

        while uw_export_idx * sim_dt_per_frame <= tuw + 1.0e-14:
            frame_path = frames_dir_path / f"unwind_to_link_frame_{uw_export_idx:05d}.txt"
            tw_value, wr_value, lk_closure_value = compute_tw_wr_lk_closure_optional(
                rod2, director_mapping_mode, compute_tw_wr=COMPUTE_TW_WR
            )
            export_frame_table_txt(
                path=frame_path,
                rod=rod2,
                director_mapping_mode=director_mapping_mode,
                link_value=current_link,
                tw_value=tw_value,
                wr_value=wr_value,
                lk_closure_value=lk_closure_value,
                stage_name="unwind_to_link",
                time_value=float(tuw),
                imposed_rotation=float(current_phi),
            )
            uw_export_idx += 1

        if current_link <= target_link + 1.0e-14:
            reached_target = True
            phi_stop = current_phi
            t_stop = tuw
            link_stop = current_link
            break

    if not reached_target:
        raise RuntimeError(
            f"Did not reach target link {target_link:.6f}. "
            f"Last link was {current_link:.6f}. Increase --extra-unwind-turns."
        )

    uw_times = np.asarray(uw_times, dtype=float)
    uw_phi = np.asarray(uw_phi, dtype=float)
    uw_link = np.asarray(uw_link, dtype=float)
    uw_I2_mean = np.asarray(uw_I2_mean, dtype=float)
    uw_I3_over_I2_mean = np.asarray(uw_I3_over_I2_mean, dtype=float)

    export_unwinding_to_target_summary_txt(
        unwind_summary_path,
        frames_t=uw_times,
        frames_phi=uw_phi,
        frames_link=uw_link,
        frames_I2_mean=uw_I2_mean,
        frames_I3_over_I2_mean=uw_I3_over_I2_mean,
    )

    # -------------------------------------------------------------------------
    # Stage 6: translate from z_init to z_final with fixed rotation
    # -------------------------------------------------------------------------
    rod3 = clone_full_rod(rod2, rod_radius, density, youngs_modulus, poisson_ratio)
    link_tracker.adopt_state_without_increment(rod3)

    p0_tr = rod3.position_collection[:, 0].copy()
    pL_tr = rod3.position_collection[:, -1].copy()
    D0_tr = rod3.director_collection[:, :, 0].copy()
    DL_tr = rod3.director_collection[:, :, -1].copy()

    sim_tr = Simulator()
    sim_tr.append(rod3)
    sim_tr.constrain(rod3).using(
        AxialSeparationBC,
        p0_init=p0_tr,
        pL_init=pL_tr,
        D0=D0_tr,
        DL=DL_tr,
        axis_hat=z_axis,
        target_span=target_span_final,
        t_move=t_translate,
    )
    sim_tr.add_forcing_to(rod3).using(RampGravity, g_vec=np.zeros(3), ramp_time=0.0)
    sim_tr.finalize()

    stepper_tr = PositionVerlet()
    do_step_tr, stages_and_updates_tr = extend_stepper_interface(stepper_tr, sim_tr)

    tr_record_times = np.arange(0.0, t_translate + 1e-12, sim_dt_per_frame)
    if tr_record_times.size == 0 or tr_record_times[-1] < t_translate - 1e-14:
        tr_record_times = np.append(tr_record_times, t_translate)

    frames_pos = []
    frames_directors = []
    frames_z = []
    frames_I2_mean = []
    frames_I3_over_I2_mean = []
    frames_time = []
    frames_link = []
    frames_phi = []

    tr_export_schedule = tr_record_times.copy()
    tr_export_idx = 0

    print("Running translation stage...")
    ttr = 0.0
    for t_target in progress_iter(tr_record_times, total=len(tr_record_times), desc="Translate z"):
        while ttr < t_target - 1.0e-14:
            ttr = do_step_tr(stepper_tr, stages_and_updates_tr, sim_tr, ttr, dt)
            rod3.velocity_collection[:, 1:-1] *= damp_fac_post
            rod3.omega_collection[:, 1:-1] *= damp_fac_post

            link_tracker.update_direct(t_stop + ttr, "translation", link_stop)

        mean_I2, mean_I3_over_I2, _, _, _ = compute_kirchhoff_metrics(rod3)

        current_span = float(np.dot(rod3.position_collection[:, -1] - rod3.position_collection[:, 0], z_axis))
        current_z = current_span / rod_arc_length

        frames_pos.append(rod3.position_collection.copy())
        frames_directors.append(rod3.director_collection.copy())
        frames_z.append(current_z)
        frames_I2_mean.append(mean_I2)
        frames_I3_over_I2_mean.append(mean_I3_over_I2)
        frames_time.append(t_stop + ttr)
        frames_link.append(link_stop)
        frames_phi.append(phi_stop)

        tr_export_idx = export_scheduled_stage_frames(
            rod=rod3,
            stage_name="translation",
            time_value=t_stop + ttr,
            schedule=t_stop + tr_export_schedule,
            next_idx=tr_export_idx,
            frames_dir_path=frames_dir_path,
            director_mapping_mode=director_mapping_mode,
            link_value=link_stop,
            imposed_rotation=phi_stop,
            compute_tw_wr=COMPUTE_TW_WR,
        )

    frames_z = np.asarray(frames_z, dtype=float)
    frames_I2_mean = np.asarray(frames_I2_mean, dtype=float)
    frames_I3_over_I2_mean = np.asarray(frames_I3_over_I2_mean, dtype=float)
    frames_time = np.asarray(frames_time, dtype=float)
    frames_link = np.asarray(frames_link, dtype=float)
    frames_phi = np.asarray(frames_phi, dtype=float)

    export_translation_summary_txt(
        translation_summary_path,
        frames_t=frames_time,
        frames_z=frames_z,
        frames_link=frames_link,
        frames_phi_const=frames_phi,
        frames_I2_mean=frames_I2_mean,
        frames_I3_over_I2_mean=frames_I3_over_I2_mean,
    )

    write_link_history_txt(link_history_path, link_tracker)

    if SAVE_GIF:
        export_translation_gif(
            frames_pos=frames_pos,
            frames_directors=frames_directors,
            frames_z=frames_z,
            frames_I2_mean=frames_I2_mean,
            frames_I3_over_I2_mean=frames_I3_over_I2_mean,
            frames_time=frames_time,
            axis_origin=axis_origin_move,
            axis_hat=axis_hat_move,
            radius=rod_radius,
            gif_path=gif_path,
            director_mapping_mode=director_mapping_mode,
            fps=FPS,
            dpi=DPI,
        )

    final_mean_I2, final_mean_I3_over_I2, _, _, _ = compute_kirchhoff_metrics(rod3)

    final_summary = {
        "target_link": target_link,
        "link_stop": float(link_stop),
        "phi_stop": float(phi_stop),
        "t_stop": float(t_stop),
        "z_init": z_init,
        "z_final": z_final,
        "gamma_twist": gamma_twist,
        "lambda_bend": lambda_bend,
        "final_mean_I2": final_mean_I2,
        "final_mean_I3_over_I2": final_mean_I3_over_I2,
        "axis_origin_x": float(axis_origin_move[0]),
        "axis_origin_y": float(axis_origin_move[1]),
        "axis_origin_z": float(axis_origin_move[2]),
        "axis_hat_x": float(axis_hat_move[0]),
        "axis_hat_y": float(axis_hat_move[1]),
        "axis_hat_z": float(axis_hat_move[2]),
        "helix_radius": float(helix_radius_move),
        "left_x": float(rod3.position_collection[0, 0]),
        "left_y": float(rod3.position_collection[1, 0]),
        "left_z": float(rod3.position_collection[2, 0]),
        "right_x": float(rod3.position_collection[0, -1]),
        "right_y": float(rod3.position_collection[1, -1]),
        "right_z": float(rod3.position_collection[2, -1]),
        "output_dir": str(output_dir),
        "translation_summary_path": str(translation_summary_path),
        "gif_path": str(gif_path),
    }

    with open(output_dir / "final_summary.txt", "w", encoding="utf-8") as f:
        for k, v in final_summary.items():
            f.write(f"{k}: {v}\n")

    if PLOT_FINAL:
        pass

    print(f"Finished target link = {target_link:.8f}")
    return final_summary


# -----------------------------------------------------------------------------
# Sweep summary export
# -----------------------------------------------------------------------------
def write_sweep_overview(output_root, link_values, summaries):
    output_root = Path(output_root)

    with open(output_root / "link_sweep_order.txt", "w", encoding="utf-8") as f:
        f.write("# target link values in the order they were run\n")
        for x in link_values:
            f.write(f"{x:.16e}\n")

    header = (
        "target_link link_stop phi_stop t_stop "
        "z_init z_final gamma_twist lambda_bend "
        "final_mean_I2 final_mean_I3_over_I2 "
        "axis_origin_x axis_origin_y axis_origin_z "
        "axis_hat_x axis_hat_y axis_hat_z "
        "helix_radius "
        "left_x left_y left_z right_x right_y right_z"
    )

    rows = []
    for s in summaries:
        rows.append(
            [
                s["target_link"],
                s["link_stop"],
                s["phi_stop"],
                s["t_stop"],
                s["z_init"],
                s["z_final"],
                s["gamma_twist"],
                s["lambda_bend"],
                s["final_mean_I2"],
                s["final_mean_I3_over_I2"],
                s["axis_origin_x"],
                s["axis_origin_y"],
                s["axis_origin_z"],
                s["axis_hat_x"],
                s["axis_hat_y"],
                s["axis_hat_z"],
                s["helix_radius"],
                s["left_x"],
                s["left_y"],
                s["left_z"],
                s["right_x"],
                s["right_y"],
                s["right_z"],
            ]
        )

    np.savetxt(
        output_root / "link_sweep_overview.txt",
        np.asarray(rows, dtype=float),
        header=header,
        comments="",
    )

    with open(output_root / "link_sweep_paths.txt", "w", encoding="utf-8") as f:
        f.write("target_link output_dir translation_summary_path gif_path\n")
        for s in summaries:
            f.write(
                f"{s['target_link']:.16e} "
                f"{s['output_dir']} "
                f"{s['translation_summary_path']} "
                f"{s['gif_path']}\n"
            )


def _parallel_worker(payload):
    args_dict = payload["args_dict"]
    target_link = payload["target_link"]
    subdir = payload["subdir"]

    class Args:
        pass

    args = Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    return run_single_experiment(args=args, target_link_value=target_link, output_dir=subdir)


# -----------------------------------------------------------------------------
# Main sweep entry point
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    link_values = build_link_sweep(
        link_min=args.link_min,
        link_max=args.link_max,
        n_points=args.link_points,
    )

    with open(output_root / "sweep_parameters.txt", "w", encoding="utf-8") as f:
        f.write(f"lambda_bend: {args.lambda_bend}\n")
        f.write(f"gamma_twist_constant: {args.gamma_twist}\n")
        f.write(f"z_init: {args.z_init}\n")
        f.write(f"z_final: {args.z_final}\n")
        f.write(f"link_min: {args.link_min}\n")
        f.write(f"link_max: {args.link_max}\n")
        f.write(f"link_points: {args.link_points}\n")
        f.write(f"extra_unwind_turns: {args.extra_unwind_turns}\n")
        f.write(f"translation_time: {args.translation_time}\n")
        f.write(f"sim_dt_per_frame: {args.sim_dt_per_frame}\n")
        f.write(f"torque_mode: {args.torque_mode}\n")
        f.write(f"torque_offset_samples: {args.torque_offset_samples}\n")
        f.write("protocol: generate -> z_init -> unwind to target link -> translate to z_final\n")
        f.write("target_link_values:\n")
        for x in link_values:
            f.write(f"  - {x:.16e}\n")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    args_dict = vars(args).copy()

    jobs = []
    for target_link in link_values:
        subdir = output_root / link_to_folder_name(target_link)
        jobs.append(
            {
                "args_dict": args_dict,
                "target_link": float(target_link),
                "subdir": str(subdir),
            }
        )

    max_workers = min(len(jobs), os.cpu_count() or 1)

    summaries_by_link = {}
    print(f"Launching parallel link-stop sweep with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_to_link = {
            ex.submit(_parallel_worker, job): job["target_link"]
            for job in jobs
        }

        for fut in progress_iter(as_completed(future_to_link), total=len(future_to_link), desc="Parallel sweep"):
            target_link = future_to_link[fut]
            try:
                summary = fut.result()
                summaries_by_link[target_link] = summary
                print(f"Completed target link = {target_link:.8f}")
            except Exception as e:
                print(f"FAILED target link = {target_link:.8f}: {e}")
                raise

    summaries = [summaries_by_link[float(x)] for x in link_values]

    write_sweep_overview(output_root, link_values, summaries)

    print("\nSweep complete.")
    print(f"Root output folder: {output_root}")
    print("Generated:")
    print(f"  {output_root / 'sweep_parameters.txt'}")
    print(f"  {output_root / 'link_sweep_order.txt'}")
    print(f"  {output_root / 'link_sweep_overview.txt'}")
    print(f"  {output_root / 'link_sweep_paths.txt'}")
    print("and one subfolder per target link.")


if __name__ == "__main__":
    main()