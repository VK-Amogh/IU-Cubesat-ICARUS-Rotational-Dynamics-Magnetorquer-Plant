# Spacecraft Rotational Dynamics + Magnetorquer Plant

This project implements the rigid-body rotational plant and magnetorquer torque mapping requested in the Week 1 + Week 2 task brief, then verifies the model with simulation cases, plots, and automated tests.

## Implemented model

The body-rate propagation is:

```python
omega_next = omega + inv(I) @ (tau - cross(omega, I @ omega)) * dt
```

where:

- `omega` is the body angular velocity vector `[wx, wy, wz]` in rad/s
- `I` is the 3x3 spacecraft inertia matrix in kg*m^2
- `tau` is the applied total body torque in N*m
- `cross(omega, I @ omega)` is the gyroscopic coupling term
- `dt` is the discrete propagation timestep in seconds

Magnetorquer torque is:

```python
tau_mag = cross(m_cmd, B_body)
```

where:

- `m_cmd` is the commanded dipole moment in A*m^2
- `B_body` is the magnetic field vector in Tesla expressed in body axes

The total propagated torque is `tau = tau_mag + tau_disturbance`.

## What is inside

- `spacecraft_dynamics/plant.py`
  - `cubesat_1u_inertia_matrix(...)` returns a representative 1U CubeSat inertia tensor.
  - `MagnetorquerTorqueMapper` computes `m x B` and optionally clips dipole commands.
  - `RigidBodyPlant.propagate_omega(...)` advances one dynamics step.
  - `RigidBodyPlant.step(...)` and `RigidBodyPlant.__call__(...)` provide the callable plant interface.
  - `RigidBodyPlant.simulate(...)` runs a trajectory using constant vectors, time profiles, or callbacks.
- `spacecraft_dynamics/verification.py`
  - Runs the requested verification cases and saves plots + JSON/Markdown summaries.
- `scripts/run_verification.py`
  - One-command entry point for regenerating all deliverables.
- `tests/test_plant.py`
  - Pytest coverage for inertia validity, torque mapping, one-step propagation, free-tumble behavior, callback inputs, and timestep sensitivity.

## Verification cases covered

- Zero-torque free motion test
- Constant dipole input response
- Random disturbance torque analysis
- Time-step sensitivity analysis
- Torque-vs-dipole mapping plot generation
- Magnetic underactuation demonstration

## Generate plots and report

From `D:\icarus`:

```powershell
python .\scripts\run_verification.py
```

This writes:

- `outputs/free_tumbling_omega.png`
- `outputs/free_tumbling_energy.png`
- `outputs/constant_dipole_omega.png`
- `outputs/constant_dipole_energy.png`
- `outputs/random_disturbance_omega.png`
- `outputs/random_disturbance_torque.png`
- `outputs/torque_vs_dipole_mapping.png`
- `outputs/magnetic_underactuation.png`
- `outputs/timestep_sensitivity.png`
- `outputs/verification_summary.json`
- `outputs/verification_report.md`

## Run tests

```powershell
python -m pytest -q
```

## Example plant usage

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

print("omega_next =", omega_next)
print("magnetic torque =", diagnostics["magnetic_torque_n_m"])
```

## Interpretation notes

- In free tumble with zero external torque, the gyroscopic term redistributes angular velocity among axes because the inertia matrix is not perfectly spherical.
- Rotational kinetic energy and angular-momentum magnitude should stay nearly constant for a sufficiently small `dt`; drift grows as `dt` increases because the implemented update is explicit Euler, exactly as requested in the brief.
- Magnetorquer torque is always perpendicular to `B`, so there is no direct torque authority around the instantaneous magnetic-field direction. The underactuation plot demonstrates this by showing that `tau dot B_hat` remains numerically zero and `|tau|` goes to zero when `m` aligns with `B`.
