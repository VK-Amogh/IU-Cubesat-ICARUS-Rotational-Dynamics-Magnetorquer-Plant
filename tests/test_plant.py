import numpy as np
import pytest

from spacecraft_dynamics import (
    MagnetorquerTorqueMapper,
    RigidBodyPlant,
    cubesat_1u_inertia_matrix,
)


def test_cubesat_inertia_is_symmetric_positive_definite() -> None:
    inertia = cubesat_1u_inertia_matrix()

    assert inertia.shape == (3, 3)
    assert np.allclose(inertia, inertia.T)
    assert np.all(np.linalg.eigvalsh(inertia) > 0.0)
    assert inertia[0, 0] > inertia[2, 2]


def test_magnetorquer_torque_matches_cross_product_and_is_perpendicular_to_b() -> None:
    mapper = MagnetorquerTorqueMapper()
    dipole = np.array([0.12, -0.03, 0.08])
    magnetic_field = np.array([27e-6, -14e-6, 41e-6])

    torque = mapper.torque(dipole, magnetic_field)

    assert np.allclose(torque, np.cross(dipole, magnetic_field))
    assert abs(float(np.dot(torque, magnetic_field))) < 1e-18


def test_magnetorquer_clips_dipole_when_limits_are_configured() -> None:
    mapper = MagnetorquerTorqueMapper(max_dipole_a_m2=np.array([0.2, 0.1, 0.15]))

    clipped = mapper.clip_dipole(np.array([0.4, -0.5, 0.05]))

    assert np.allclose(clipped, np.array([0.2, -0.1, 0.05]))


def test_propagate_omega_matches_discrete_rigid_body_equation() -> None:
    inertia = cubesat_1u_inertia_matrix()
    plant = RigidBodyPlant(inertia_matrix_kg_m2=inertia, dt_s=0.1)
    omega = np.array([0.4, -0.25, 0.18])
    torque = np.array([2.5e-6, -1.2e-6, 0.9e-6])

    propagated = plant.propagate_omega(omega, torque)

    expected = omega + np.linalg.inv(inertia) @ (
        torque - np.cross(omega, inertia @ omega)
    ) * plant.dt_s
    assert np.allclose(propagated, expected)


def test_step_accepts_default_zero_inputs() -> None:
    plant = RigidBodyPlant(dt_s=0.1)
    omega_next, diagnostics = plant.step(np.array([0.1, -0.2, 0.05]))

    assert omega_next.shape == (3,)
    assert np.allclose(diagnostics["dipole_a_m2"], np.zeros(3))
    assert np.allclose(diagnostics["magnetic_field_t"], np.zeros(3))
    assert np.allclose(diagnostics["magnetic_torque_n_m"], np.zeros(3))
    assert np.allclose(diagnostics["disturbance_torque_n_m"], np.zeros(3))
    assert np.allclose(diagnostics["total_torque_n_m"], np.zeros(3))


def test_constant_dipole_step_changes_rate_by_expected_magnetic_torque() -> None:
    plant = RigidBodyPlant(dt_s=0.2)
    omega0 = np.zeros(3)
    dipole = np.array([0.18, -0.08, 0.05])
    b_field = np.array([28e-6, -12e-6, 41e-6])

    omega_next, diagnostics = plant(
        omega0,
        dipole_command_a_m2=dipole,
        magnetic_field_t=b_field,
    )

    expected_torque = np.cross(dipole, b_field)
    expected_omega = omega0 + np.linalg.inv(plant.inertia_matrix_kg_m2) @ expected_torque * 0.2

    assert np.allclose(diagnostics["magnetic_torque_n_m"], expected_torque)
    assert np.allclose(omega_next, expected_omega)


def test_free_tumbling_is_nearly_energy_conserving_for_small_dt() -> None:
    plant = RigidBodyPlant(dt_s=0.005)

    result = plant.simulate(
        omega0_rad_s=np.array([0.35, -0.22, 0.17]),
        duration_s=40.0,
        dipole_command_a_m2=np.zeros(3),
        magnetic_field_t=np.zeros(3),
        disturbance_torque_n_m=np.zeros(3),
    )

    relative_energy_drift = abs(
        (result.kinetic_energy_j[-1] - result.kinetic_energy_j[0]) / result.kinetic_energy_j[0]
    )
    h_norm = np.linalg.norm(result.angular_momentum_nms, axis=1)
    relative_h_drift = abs((h_norm[-1] - h_norm[0]) / h_norm[0])

    assert relative_energy_drift < 2e-3
    assert relative_h_drift < 2e-3


def test_simulate_accepts_state_dependent_callbacks() -> None:
    plant = RigidBodyPlant(dt_s=0.1, magnetorquer=MagnetorquerTorqueMapper(max_dipole_a_m2=0.2))

    def dipole_callback(step_index: int, time_s: float, omega_rad_s: np.ndarray) -> np.ndarray:
        del step_index, time_s
        return np.array([-0.8 * omega_rad_s[0], 0.4, 0.0])

    def b_callback(step_index: int, time_s: float, omega_rad_s: np.ndarray) -> np.ndarray:
        del step_index, omega_rad_s
        return np.array([25e-6, 5e-6 * np.sin(time_s), 38e-6])

    result = plant.simulate(
        omega0_rad_s=np.array([0.2, -0.1, 0.05]),
        duration_s=5.0,
        dipole_command_a_m2=dipole_callback,
        magnetic_field_t=b_callback,
    )

    assert result.omega_rad_s.shape[0] == 51
    assert result.dipole_a_m2.shape == (50, 3)
    assert np.max(np.abs(result.dipole_a_m2)) <= 0.2 + 1e-12
    assert np.all(np.isfinite(result.omega_rad_s))


def test_smaller_timestep_tracks_reference_better_than_coarse_timestep() -> None:
    omega0 = np.array([0.38, -0.29, 0.21])
    duration_s = 20.0

    reference = RigidBodyPlant(dt_s=0.002).simulate(
        omega0_rad_s=omega0,
        duration_s=duration_s,
        dipole_command_a_m2=np.zeros(3),
        magnetic_field_t=np.zeros(3),
        disturbance_torque_n_m=np.zeros(3),
    )
    fine = RigidBodyPlant(dt_s=0.02).simulate(
        omega0_rad_s=omega0,
        duration_s=duration_s,
        dipole_command_a_m2=np.zeros(3),
        magnetic_field_t=np.zeros(3),
        disturbance_torque_n_m=np.zeros(3),
    )
    coarse = RigidBodyPlant(dt_s=0.2).simulate(
        omega0_rad_s=omega0,
        duration_s=duration_s,
        dipole_command_a_m2=np.zeros(3),
        magnetic_field_t=np.zeros(3),
        disturbance_torque_n_m=np.zeros(3),
    )

    fine_error = np.linalg.norm(fine.omega_rad_s[-1] - reference.omega_rad_s[-1])
    coarse_error = np.linalg.norm(coarse.omega_rad_s[-1] - reference.omega_rad_s[-1])

    assert fine_error < coarse_error


def test_invalid_inputs_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="shape"):
        RigidBodyPlant(inertia_matrix_kg_m2=np.eye(2))

    with pytest.raises(ValueError, match="positive timestep"):
        RigidBodyPlant(dt_s=0.0)

    plant = RigidBodyPlant(dt_s=0.1)
    with pytest.raises(ValueError, match="exactly 3 elements"):
        plant.step(np.array([1.0, 2.0]))
