"""Verification scenarios and plot generation for the rotational plant."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .plant import MagnetorquerTorqueMapper, RigidBodyPlant, SimulationResult


MICRO_TESLA = 1e-6
DEFAULT_B_FIELD_T = np.array([28.0, -12.0, 41.0], dtype=float) * MICRO_TESLA


def _style_axis(ax: plt.Axes, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_omega_history(result: SimulationResult, output_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.time_s, result.omega_rad_s[:, 0], label=r"$\omega_x$")
    ax.plot(result.time_s, result.omega_rad_s[:, 1], label=r"$\omega_y$")
    ax.plot(result.time_s, result.omega_rad_s[:, 2], label=r"$\omega_z$")
    _style_axis(
        ax,
        xlabel="Time [s]",
        ylabel="Angular velocity [rad/s]",
        title=title,
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_kinetic_energy(result: SimulationResult, output_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.time_s, result.kinetic_energy_j, color="tab:purple")
    _style_axis(
        ax,
        xlabel="Time [s]",
        ylabel="Rotational kinetic energy [J]",
        title=title,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_torque_vs_dipole_mapping(
    mapper: MagnetorquerTorqueMapper,
    output_path: Path,
    *,
    magnetic_field_t: np.ndarray = DEFAULT_B_FIELD_T,
    max_dipole_a_m2: float = 0.25,
    samples: int = 301,
) -> dict[str, float]:
    b_field = np.asarray(magnetic_field_t, dtype=float)
    dipole_axis = np.linspace(-max_dipole_a_m2, max_dipole_a_m2, samples)
    axis_names = ["x", "y", "z"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    max_torque_norm = 0.0

    for axis_idx, axis_name in enumerate(axis_names):
        torques = np.zeros((samples, 3), dtype=float)
        for sample_idx, dipole_value in enumerate(dipole_axis):
            dipole = np.zeros(3, dtype=float)
            dipole[axis_idx] = dipole_value
            torques[sample_idx] = mapper.torque(dipole, b_field)

        max_torque_norm = max(max_torque_norm, float(np.max(np.linalg.norm(torques, axis=1))))
        axes[axis_idx].plot(dipole_axis, torques[:, 0] * 1e6, label=r"$\tau_x$")
        axes[axis_idx].plot(dipole_axis, torques[:, 1] * 1e6, label=r"$\tau_y$")
        axes[axis_idx].plot(dipole_axis, torques[:, 2] * 1e6, label=r"$\tau_z$")
        _style_axis(
            axes[axis_idx],
            xlabel="Commanded dipole [A*m^2]" if axis_idx == 2 else "",
            ylabel="Torque [uN*m]",
            title=f"Torque response to pure m_{axis_name} command, B = {b_field * 1e6} uT",
        )
        axes[axis_idx].legend(loc="best", ncol=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return {
        "b_field_norm_t": float(np.linalg.norm(b_field)),
        "max_torque_norm_n_m": max_torque_norm,
    }


def plot_magnetic_underactuation(
    mapper: MagnetorquerTorqueMapper,
    output_path: Path,
    *,
    magnetic_field_t: np.ndarray = DEFAULT_B_FIELD_T,
    dipole_magnitude_a_m2: float = 0.2,
    samples: int = 361,
) -> dict[str, float]:
    b_field = np.asarray(magnetic_field_t, dtype=float)
    b_norm = float(np.linalg.norm(b_field))
    if b_norm == 0.0:
        raise ValueError("magnetic_field_t must be non-zero to demonstrate underactuation.")

    b_hat = b_field / b_norm
    candidate = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(candidate, b_hat))) > 0.95:
        candidate = np.array([0.0, 1.0, 0.0], dtype=float)
    perpendicular_hat = candidate - np.dot(candidate, b_hat) * b_hat
    perpendicular_hat /= np.linalg.norm(perpendicular_hat)

    angle_deg = np.linspace(0.0, 180.0, samples)
    torque_norm = np.zeros(samples, dtype=float)
    torque_parallel = np.zeros(samples, dtype=float)

    for idx, angle in enumerate(np.deg2rad(angle_deg)):
        dipole = dipole_magnitude_a_m2 * (
            np.cos(angle) * b_hat + np.sin(angle) * perpendicular_hat
        )
        torque = mapper.torque(dipole, b_field)
        torque_norm[idx] = np.linalg.norm(torque)
        torque_parallel[idx] = np.dot(torque, b_hat)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    expected_norm = dipole_magnitude_a_m2 * b_norm * np.sin(np.deg2rad(angle_deg))
    axes[0].plot(angle_deg, torque_norm * 1e6, label="Simulated |m × B|")
    axes[0].plot(
        angle_deg,
        expected_norm * 1e6,
        "--",
        label=r"$|m||B|\sin(\theta)$",
    )
    _style_axis(
        axes[0],
        xlabel="",
        ylabel="Torque magnitude [uN*m]",
        title="Magnetic underactuation: torque vanishes when dipole aligns with B",
    )
    axes[0].legend(loc="best")

    axes[1].plot(angle_deg, torque_parallel * 1e12, color="tab:red")
    _style_axis(
        axes[1],
        xlabel="Angle between dipole and magnetic field [deg]",
        ylabel=r"$\tau \cdot \hat{B}$ [pN*m]",
        title="Torque component along B stays numerically zero",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return {
        "b_field_norm_t": b_norm,
        "dipole_magnitude_a_m2": float(dipole_magnitude_a_m2),
        "max_abs_tau_parallel_n_m": float(np.max(np.abs(torque_parallel))),
        "peak_torque_n_m": float(np.max(torque_norm)),
    }


def _trajectory_metrics(result: SimulationResult) -> dict[str, float]:
    omega_norm = np.linalg.norm(result.omega_rad_s, axis=1)
    h_norm = np.linalg.norm(result.angular_momentum_nms, axis=1)
    energy_0 = float(result.kinetic_energy_j[0])
    h0 = float(h_norm[0])

    return {
        "final_omega_norm_rad_s": float(omega_norm[-1]),
        "max_omega_norm_rad_s": float(np.max(omega_norm)),
        "max_total_torque_n_m": float(np.max(np.linalg.norm(result.total_torque_n_m, axis=1)))
        if result.total_torque_n_m.size
        else 0.0,
        "relative_energy_change": float((result.kinetic_energy_j[-1] - energy_0) / energy_0)
        if energy_0 > 0.0
        else 0.0,
        "relative_angular_momentum_change": float((h_norm[-1] - h0) / h0) if h0 > 0.0 else 0.0,
    }


def run_free_tumbling_case(
    output_dir: Path,
    *,
    inertia_matrix_kg_m2: np.ndarray | None = None,
) -> tuple[SimulationResult, dict[str, float]]:
    plant = RigidBodyPlant(inertia_matrix_kg_m2=inertia_matrix_kg_m2, dt_s=0.05)
    result = plant.simulate(
        omega0_rad_s=np.array([0.42, -0.31, 0.27], dtype=float),
        duration_s=220.0,
        dipole_command_a_m2=np.zeros(3),
        magnetic_field_t=np.zeros(3),
        disturbance_torque_n_m=np.zeros(3),
    )

    plot_omega_history(
        result,
        output_dir / "free_tumbling_omega.png",
        title="Zero-torque free tumbling response",
    )
    plot_kinetic_energy(
        result,
        output_dir / "free_tumbling_energy.png",
        title="Zero-torque rotational kinetic energy trend",
    )
    return result, _trajectory_metrics(result)


def run_constant_dipole_case(
    output_dir: Path,
    *,
    inertia_matrix_kg_m2: np.ndarray | None = None,
) -> tuple[SimulationResult, dict[str, float]]:
    plant = RigidBodyPlant(
        inertia_matrix_kg_m2=inertia_matrix_kg_m2,
        dt_s=0.1,
        magnetorquer=MagnetorquerTorqueMapper(max_dipole_a_m2=0.25),
    )
    constant_dipole = np.array([0.18, -0.08, 0.05], dtype=float)
    constant_b = DEFAULT_B_FIELD_T

    result = plant.simulate(
        omega0_rad_s=np.array([0.01, -0.015, 0.02], dtype=float),
        duration_s=300.0,
        dipole_command_a_m2=constant_dipole,
        magnetic_field_t=constant_b,
        disturbance_torque_n_m=np.zeros(3),
    )
    plot_omega_history(
        result,
        output_dir / "constant_dipole_omega.png",
        title="Response to constant magnetorquer dipole command",
    )
    plot_kinetic_energy(
        result,
        output_dir / "constant_dipole_energy.png",
        title="Kinetic energy under constant magnetic torque",
    )
    return result, _trajectory_metrics(result)


def run_random_disturbance_case(
    output_dir: Path,
    *,
    inertia_matrix_kg_m2: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[SimulationResult, dict[str, float]]:
    rng = np.random.default_rng(seed)
    plant = RigidBodyPlant(inertia_matrix_kg_m2=inertia_matrix_kg_m2, dt_s=0.05)
    steps = int(np.round(180.0 / plant.dt_s))
    disturbance_profile = rng.normal(loc=0.0, scale=1.8e-7, size=(steps, 3))
    disturbance_profile += 6.0e-8 * np.column_stack(
        [
            np.sin(np.linspace(0.0, 10.0 * np.pi, steps)),
            np.cos(np.linspace(0.0, 6.0 * np.pi, steps)),
            np.sin(np.linspace(0.0, 4.0 * np.pi, steps) + 0.3),
        ]
    )

    result = plant.simulate(
        omega0_rad_s=np.array([0.08, -0.05, 0.04], dtype=float),
        duration_s=180.0,
        dipole_command_a_m2=np.zeros(3),
        magnetic_field_t=np.zeros(3),
        disturbance_torque_n_m=disturbance_profile,
    )

    plot_omega_history(
        result,
        output_dir / "random_disturbance_omega.png",
        title="Angular velocity under random disturbance torque",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    torque_time = result.time_s[:-1]
    ax.plot(torque_time, result.total_torque_n_m[:, 0] * 1e6, label=r"$\tau_x$")
    ax.plot(torque_time, result.total_torque_n_m[:, 1] * 1e6, label=r"$\tau_y$")
    ax.plot(torque_time, result.total_torque_n_m[:, 2] * 1e6, label=r"$\tau_z$")
    _style_axis(
        ax,
        xlabel="Time [s]",
        ylabel="Disturbance torque [uN*m]",
        title="Injected random disturbance torque profile",
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "random_disturbance_torque.png", dpi=160)
    plt.close(fig)
    return result, _trajectory_metrics(result)


def run_timestep_sensitivity_case(
    output_dir: Path,
    *,
    inertia_matrix_kg_m2: np.ndarray | None = None,
) -> dict[str, object]:
    dt_values = np.array([0.5, 0.2, 0.1, 0.05, 0.02], dtype=float)
    reference_dt = 0.005
    duration_s = 120.0
    omega0 = np.array([0.38, -0.29, 0.21], dtype=float)

    reference_plant = RigidBodyPlant(
        inertia_matrix_kg_m2=inertia_matrix_kg_m2,
        dt_s=reference_dt,
    )
    reference_result = reference_plant.simulate(
        omega0_rad_s=omega0,
        duration_s=duration_s,
        dipole_command_a_m2=np.zeros(3),
        magnetic_field_t=np.zeros(3),
        disturbance_torque_n_m=np.zeros(3),
    )
    reference_final = reference_result.omega_rad_s[-1]
    reference_energy = reference_result.kinetic_energy_j[-1]

    final_omega_error = np.zeros_like(dt_values)
    energy_error = np.zeros_like(dt_values)

    for idx, dt_s in enumerate(dt_values):
        plant = RigidBodyPlant(inertia_matrix_kg_m2=inertia_matrix_kg_m2, dt_s=float(dt_s))
        result = plant.simulate(
            omega0_rad_s=omega0,
            duration_s=duration_s,
            dipole_command_a_m2=np.zeros(3),
            magnetic_field_t=np.zeros(3),
            disturbance_torque_n_m=np.zeros(3),
        )
        final_omega_error[idx] = np.linalg.norm(result.omega_rad_s[-1] - reference_final)
        energy_error[idx] = abs(result.kinetic_energy_j[-1] - reference_energy) / reference_energy

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].loglog(dt_values, final_omega_error, marker="o")
    _style_axis(
        axes[0],
        xlabel="",
        ylabel="Final-rate error [rad/s]",
        title=f"Time-step sensitivity relative to dt = {reference_dt} s reference",
    )
    axes[1].loglog(dt_values, energy_error, marker="o", color="tab:orange")
    _style_axis(
        axes[1],
        xlabel="Propagation timestep dt [s]",
        ylabel="Relative final energy error [-]",
        title="Energy drift increases as explicit Euler timestep grows",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "timestep_sensitivity.png", dpi=160)
    plt.close(fig)

    return {
        "reference_dt_s": reference_dt,
        "dt_values_s": dt_values.tolist(),
        "final_omega_error_rad_s": final_omega_error.tolist(),
        "relative_final_energy_error": energy_error.tolist(),
        "best_dt_s": float(dt_values[np.argmin(final_omega_error)]),
        "worst_dt_s": float(dt_values[np.argmax(final_omega_error)]),
    }


def build_markdown_report(summary: dict[str, object], output_dir: Path) -> str:
    return f"""# Rotational Dynamics + Magnetorquer Verification Report

## What was built
- A discrete-time spacecraft rigid-body rotational plant implementing
  `omega[k+1] = omega[k] + inv(I) @ (tau[k] - omega[k] x (I @ omega[k])) * dt`.
- A magnetorquer torque mapper implementing `tau_mag = m x B` with optional dipole saturation.
- A callable step interface (`RigidBodyPlant.step` / `RigidBodyPlant.__call__`) and a trajectory simulator (`RigidBodyPlant.simulate`).
- Verification scripts and plots covering zero-torque free motion, constant dipole excitation,
  random disturbance injection, time-step sensitivity, and magnetic underactuation.

## Generated figures
- ![Free tumble angular velocity]({(output_dir / "free_tumbling_omega.png").as_posix()})
- ![Free tumble kinetic energy]({(output_dir / "free_tumbling_energy.png").as_posix()})
- ![Constant dipole angular velocity]({(output_dir / "constant_dipole_omega.png").as_posix()})
- ![Constant dipole kinetic energy]({(output_dir / "constant_dipole_energy.png").as_posix()})
- ![Torque vs dipole mapping]({(output_dir / "torque_vs_dipole_mapping.png").as_posix()})
- ![Magnetic underactuation]({(output_dir / "magnetic_underactuation.png").as_posix()})
- ![Random disturbance angular velocity]({(output_dir / "random_disturbance_omega.png").as_posix()})
- ![Random disturbance torque]({(output_dir / "random_disturbance_torque.png").as_posix()})
- ![Time-step sensitivity]({(output_dir / "timestep_sensitivity.png").as_posix()})

## Numerical summary
```json
{json.dumps(summary, indent=2)}
```

## How to use the propagation interface
```python
import numpy as np
from spacecraft_dynamics import RigidBodyPlant

plant = RigidBodyPlant(dt_s=0.1)
omega_next, diagnostics = plant(
    omega_rad_s=np.array([0.05, -0.02, 0.01]),
    dipole_command_a_m2=np.array([0.12, 0.00, -0.08]),
    magnetic_field_t=np.array([28.0e-6, -12.0e-6, 41.0e-6]),
    disturbance_torque_n_m=np.zeros(3),
)
```
"""


def run_verification_suite(output_dir: Path | str) -> dict[str, object]:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    free_tumble_result, free_tumble_metrics = run_free_tumbling_case(output_path)
    constant_dipole_result, constant_dipole_metrics = run_constant_dipole_case(output_path)
    random_disturbance_result, random_disturbance_metrics = run_random_disturbance_case(output_path)

    mapper = MagnetorquerTorqueMapper(max_dipole_a_m2=0.25)
    torque_mapping_metrics = plot_torque_vs_dipole_mapping(
        mapper,
        output_path / "torque_vs_dipole_mapping.png",
    )
    underactuation_metrics = plot_magnetic_underactuation(
        mapper,
        output_path / "magnetic_underactuation.png",
    )
    timestep_metrics = run_timestep_sensitivity_case(output_path)

    summary: dict[str, object] = {
        "inertia_matrix_kg_m2": RigidBodyPlant().inertia_matrix_kg_m2.tolist(),
        "free_tumbling": free_tumble_metrics,
        "constant_dipole": {
            **constant_dipole_metrics,
            "steady_dipole_a_m2": constant_dipole_result.dipole_a_m2[0].tolist(),
            "steady_magnetic_field_t": constant_dipole_result.magnetic_field_t[0].tolist(),
            "steady_magnetic_torque_n_m": constant_dipole_result.magnetic_torque_n_m[0].tolist(),
        },
        "random_disturbance": {
            **random_disturbance_metrics,
            "rms_disturbance_torque_n_m": float(
                np.sqrt(np.mean(random_disturbance_result.total_torque_n_m**2))
            ),
        },
        "torque_vs_dipole_mapping": torque_mapping_metrics,
        "magnetic_underactuation": underactuation_metrics,
        "timestep_sensitivity": timestep_metrics,
        "generated_files": sorted(path.name for path in output_path.iterdir() if path.is_file()),
    }

    (output_path / "verification_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_path / "verification_report.md").write_text(
        build_markdown_report(summary, output_path),
        encoding="utf-8",
    )
    return summary
