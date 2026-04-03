"""Rigid-body rotational propagation and magnetorquer torque mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


VectorSource = (
    np.ndarray
    | list[float]
    | tuple[float, float, float]
    | Callable[[int, float, np.ndarray], np.ndarray | list[float] | tuple[float, float, float]]
    | None
)


def _as_vector3(name: str, value: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 elements, got shape {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _validate_inertia_matrix(inertia_matrix: np.ndarray) -> np.ndarray:
    inertia = np.asarray(inertia_matrix, dtype=float)
    if inertia.shape != (3, 3):
        raise ValueError(f"inertia_matrix must have shape (3, 3), got {inertia.shape}.")
    if not np.all(np.isfinite(inertia)):
        raise ValueError("inertia_matrix must contain only finite values.")
    if not np.allclose(inertia, inertia.T, atol=1e-12):
        raise ValueError("inertia_matrix must be symmetric.")

    eigenvalues = np.linalg.eigvalsh(inertia)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError(
            "inertia_matrix must be positive definite. "
            f"Eigenvalues were {eigenvalues.tolist()}."
        )
    return inertia


def _resolve_vector_source(
    source: VectorSource,
    *,
    step_index: int,
    time_s: float,
    omega_rad_s: np.ndarray,
    name: str,
) -> np.ndarray:
    if source is None:
        return np.zeros(3, dtype=float)

    if callable(source):
        value = source(step_index, time_s, omega_rad_s.copy())
        return _as_vector3(name, value)

    array = np.asarray(source, dtype=float)
    if array.ndim == 1:
        return _as_vector3(name, array)
    if array.ndim == 2 and array.shape[1] == 3:
        if step_index >= array.shape[0]:
            raise ValueError(
                f"{name} has {array.shape[0]} samples but step {step_index} was requested."
            )
        return _as_vector3(name, array[step_index])

    raise ValueError(
        f"{name} must be a 3-vector, an (N, 3) profile, a callback, or None; "
        f"got array shape {array.shape}."
    )


def cubesat_1u_inertia_matrix(
    *,
    mass_kg: float = 1.33,
    dimensions_m: tuple[float, float, float] = (0.10, 0.10, 0.1135),
    products_of_inertia_kg_m2: tuple[float, float, float] = (1.2e-5, -8.0e-6, 1.0e-5),
) -> np.ndarray:
    """
    Return a representative 1U CubeSat inertia matrix in body axes.

    The default body is modeled as a slightly elongated 1U bus with a mild
    internal mass offset, so the tensor is positive definite but not perfectly
    spherical. That asymmetry is useful for validating the gyroscopic coupling
    term during free-tumble propagation.
    """

    if mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive.")

    lengths = np.asarray(dimensions_m, dtype=float)
    if lengths.shape != (3,) or np.any(lengths <= 0.0):
        raise ValueError("dimensions_m must contain three positive side lengths.")

    lx, ly, lz = lengths
    i_xx = mass_kg * (ly * ly + lz * lz) / 12.0
    i_yy = mass_kg * (lx * lx + lz * lz) / 12.0
    i_zz = mass_kg * (lx * lx + ly * ly) / 12.0

    i_xy, i_xz, i_yz = np.asarray(products_of_inertia_kg_m2, dtype=float)
    inertia = np.array(
        [
            [i_xx, -i_xy, -i_xz],
            [-i_xy, i_yy, -i_yz],
            [-i_xz, -i_yz, i_zz],
        ],
        dtype=float,
    )
    return _validate_inertia_matrix(inertia)


@dataclass(frozen=True)
class SimulationResult:
    """Container for one propagated attitude-rate trajectory."""

    time_s: np.ndarray
    omega_rad_s: np.ndarray
    kinetic_energy_j: np.ndarray
    angular_momentum_nms: np.ndarray
    dipole_a_m2: np.ndarray
    magnetic_field_t: np.ndarray
    magnetic_torque_n_m: np.ndarray
    disturbance_torque_n_m: np.ndarray
    total_torque_n_m: np.ndarray


@dataclass(frozen=True)
class MagnetorquerTorqueMapper:
    """Map commanded magnetic dipole and local magnetic field into body torque."""

    max_dipole_a_m2: float | np.ndarray | None = None

    def clip_dipole(self, dipole_command_a_m2: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
        dipole = _as_vector3("dipole_command_a_m2", dipole_command_a_m2)
        if self.max_dipole_a_m2 is None:
            return dipole

        max_dipole = np.asarray(self.max_dipole_a_m2, dtype=float)
        if max_dipole.ndim == 0:
            if max_dipole <= 0.0:
                raise ValueError("max_dipole_a_m2 must be positive.")
            return np.clip(dipole, -float(max_dipole), float(max_dipole))

        max_vector = _as_vector3("max_dipole_a_m2", max_dipole)
        if np.any(max_vector <= 0.0):
            raise ValueError("max_dipole_a_m2 must contain positive limits.")
        return np.clip(dipole, -max_vector, max_vector)

    def torque(
        self,
        dipole_command_a_m2: np.ndarray | list[float] | tuple[float, float, float],
        magnetic_field_t: np.ndarray | list[float] | tuple[float, float, float],
    ) -> np.ndarray:
        dipole = self.clip_dipole(dipole_command_a_m2)
        magnetic_field = _as_vector3("magnetic_field_t", magnetic_field_t)
        return np.cross(dipole, magnetic_field)


@dataclass
class RigidBodyPlant:
    """
    Discrete-time rigid-body rotational plant.

    Propagation follows:
        omega[k+1] = omega[k] + inv(I) @ (tau[k] - omega[k] x (I @ omega[k])) * dt
    with magnetic torque tau_mag = m x B.
    """

    inertia_matrix_kg_m2: np.ndarray | None = None
    dt_s: float = 0.1
    magnetorquer: MagnetorquerTorqueMapper | None = None

    def __post_init__(self) -> None:
        if self.inertia_matrix_kg_m2 is None:
            self.inertia_matrix_kg_m2 = cubesat_1u_inertia_matrix()

        self.inertia_matrix_kg_m2 = _validate_inertia_matrix(self.inertia_matrix_kg_m2)
        if self.dt_s <= 0.0 or not np.isfinite(self.dt_s):
            raise ValueError("dt_s must be a finite positive timestep.")

        if self.magnetorquer is None:
            self.magnetorquer = MagnetorquerTorqueMapper()

        self._inertia_inv_1_kg_m2 = np.linalg.inv(self.inertia_matrix_kg_m2)

    def gyroscopic_coupling_n_m(self, omega_rad_s: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
        omega = _as_vector3("omega_rad_s", omega_rad_s)
        angular_momentum = self.inertia_matrix_kg_m2 @ omega
        return np.cross(omega, angular_momentum)

    def angular_momentum_nms(
        self,
        omega_rad_s: np.ndarray | list[float] | tuple[float, float, float],
    ) -> np.ndarray:
        omega = _as_vector3("omega_rad_s", omega_rad_s)
        return self.inertia_matrix_kg_m2 @ omega

    def kinetic_energy_j(
        self,
        omega_rad_s: np.ndarray | list[float] | tuple[float, float, float],
    ) -> float:
        omega = _as_vector3("omega_rad_s", omega_rad_s)
        return float(0.5 * omega @ self.inertia_matrix_kg_m2 @ omega)

    def propagate_omega(
        self,
        omega_rad_s: np.ndarray | list[float] | tuple[float, float, float],
        torque_n_m: np.ndarray | list[float] | tuple[float, float, float],
    ) -> np.ndarray:
        omega = _as_vector3("omega_rad_s", omega_rad_s)
        torque = _as_vector3("torque_n_m", torque_n_m)
        gyroscopic_term = self.gyroscopic_coupling_n_m(omega)
        omega_dot = self._inertia_inv_1_kg_m2 @ (torque - gyroscopic_term)
        return omega + self.dt_s * omega_dot

    def step(
        self,
        omega_rad_s: np.ndarray | list[float] | tuple[float, float, float],
        dipole_command_a_m2: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        magnetic_field_t: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        disturbance_torque_n_m: np.ndarray | list[float] | tuple[float, float, float] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        omega = _as_vector3("omega_rad_s", omega_rad_s)
        dipole = _as_vector3(
            "dipole_command_a_m2",
            np.zeros(3) if dipole_command_a_m2 is None else dipole_command_a_m2,
        )
        magnetic_field = _as_vector3(
            "magnetic_field_t",
            np.zeros(3) if magnetic_field_t is None else magnetic_field_t,
        )
        disturbance = _as_vector3(
            "disturbance_torque_n_m",
            np.zeros(3) if disturbance_torque_n_m is None else disturbance_torque_n_m,
        )

        magnetic_torque = self.magnetorquer.torque(dipole, magnetic_field)
        total_torque = magnetic_torque + disturbance
        omega_next = self.propagate_omega(omega, total_torque)
        diagnostics = {
            "dipole_a_m2": self.magnetorquer.clip_dipole(dipole),
            "magnetic_field_t": magnetic_field,
            "magnetic_torque_n_m": magnetic_torque,
            "disturbance_torque_n_m": disturbance,
            "total_torque_n_m": total_torque,
            "gyroscopic_torque_n_m": self.gyroscopic_coupling_n_m(omega),
        }
        return omega_next, diagnostics

    def __call__(
        self,
        omega_rad_s: np.ndarray | list[float] | tuple[float, float, float],
        dipole_command_a_m2: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        magnetic_field_t: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        disturbance_torque_n_m: np.ndarray | list[float] | tuple[float, float, float] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return self.step(
            omega_rad_s,
            dipole_command_a_m2=dipole_command_a_m2,
            magnetic_field_t=magnetic_field_t,
            disturbance_torque_n_m=disturbance_torque_n_m,
        )

    def simulate(
        self,
        *,
        omega0_rad_s: np.ndarray | list[float] | tuple[float, float, float],
        duration_s: float,
        dipole_command_a_m2: VectorSource = None,
        magnetic_field_t: VectorSource = None,
        disturbance_torque_n_m: VectorSource = None,
    ) -> SimulationResult:
        if duration_s <= 0.0 or not np.isfinite(duration_s):
            raise ValueError("duration_s must be finite and positive.")

        steps = int(np.round(duration_s / self.dt_s))
        if steps < 1:
            raise ValueError("duration_s must span at least one propagation step.")

        time_s = np.arange(steps + 1, dtype=float) * self.dt_s
        omega_rad_s = np.zeros((steps + 1, 3), dtype=float)
        angular_momentum_nms = np.zeros((steps + 1, 3), dtype=float)
        kinetic_energy_j = np.zeros(steps + 1, dtype=float)
        dipole_a_m2 = np.zeros((steps, 3), dtype=float)
        magnetic_field_samples_t = np.zeros((steps, 3), dtype=float)
        magnetic_torque_n_m = np.zeros((steps, 3), dtype=float)
        disturbance_samples_n_m = np.zeros((steps, 3), dtype=float)
        total_torque_n_m = np.zeros((steps, 3), dtype=float)

        omega_rad_s[0] = _as_vector3("omega0_rad_s", omega0_rad_s)
        angular_momentum_nms[0] = self.angular_momentum_nms(omega_rad_s[0])
        kinetic_energy_j[0] = self.kinetic_energy_j(omega_rad_s[0])

        for step_index in range(steps):
            current_time_s = time_s[step_index]
            dipole_k = _resolve_vector_source(
                dipole_command_a_m2,
                step_index=step_index,
                time_s=current_time_s,
                omega_rad_s=omega_rad_s[step_index],
                name="dipole_command_a_m2",
            )
            magnetic_field_k = _resolve_vector_source(
                magnetic_field_t,
                step_index=step_index,
                time_s=current_time_s,
                omega_rad_s=omega_rad_s[step_index],
                name="magnetic_field_t",
            )
            disturbance_k = _resolve_vector_source(
                disturbance_torque_n_m,
                step_index=step_index,
                time_s=current_time_s,
                omega_rad_s=omega_rad_s[step_index],
                name="disturbance_torque_n_m",
            )

            next_omega, diagnostics = self.step(
                omega_rad_s[step_index],
                dipole_command_a_m2=dipole_k,
                magnetic_field_t=magnetic_field_k,
                disturbance_torque_n_m=disturbance_k,
            )
            omega_rad_s[step_index + 1] = next_omega
            angular_momentum_nms[step_index + 1] = self.angular_momentum_nms(next_omega)
            kinetic_energy_j[step_index + 1] = self.kinetic_energy_j(next_omega)
            dipole_a_m2[step_index] = diagnostics["dipole_a_m2"]
            magnetic_field_samples_t[step_index] = diagnostics["magnetic_field_t"]
            magnetic_torque_n_m[step_index] = diagnostics["magnetic_torque_n_m"]
            disturbance_samples_n_m[step_index] = diagnostics["disturbance_torque_n_m"]
            total_torque_n_m[step_index] = diagnostics["total_torque_n_m"]

        return SimulationResult(
            time_s=time_s,
            omega_rad_s=omega_rad_s,
            kinetic_energy_j=kinetic_energy_j,
            angular_momentum_nms=angular_momentum_nms,
            dipole_a_m2=dipole_a_m2,
            magnetic_field_t=magnetic_field_samples_t,
            magnetic_torque_n_m=magnetic_torque_n_m,
            disturbance_torque_n_m=disturbance_samples_n_m,
            total_torque_n_m=total_torque_n_m,
        )
