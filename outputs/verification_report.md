# Rotational Dynamics + Magnetorquer Verification Report

## What was built
- A discrete-time spacecraft rigid-body rotational plant implementing
  `omega[k+1] = omega[k] + inv(I) @ (tau[k] - omega[k] x (I @ omega[k])) * dt`.
- A magnetorquer torque mapper implementing `tau_mag = m x B` with optional dipole saturation.
- A callable step interface (`RigidBodyPlant.step` / `RigidBodyPlant.__call__`) and a trajectory simulator (`RigidBodyPlant.simulate`).
- Verification scripts and plots covering zero-torque free motion, constant dipole excitation,
  random disturbance injection, time-step sensitivity, and magnetic underactuation.

## Generated figures
- ![Free tumble angular velocity](D:/icarus/outputs/free_tumbling_omega.png)
- ![Free tumble kinetic energy](D:/icarus/outputs/free_tumbling_energy.png)
- ![Constant dipole angular velocity](D:/icarus/outputs/constant_dipole_omega.png)
- ![Constant dipole kinetic energy](D:/icarus/outputs/constant_dipole_energy.png)
- ![Torque vs dipole mapping](D:/icarus/outputs/torque_vs_dipole_mapping.png)
- ![Magnetic underactuation](D:/icarus/outputs/magnetic_underactuation.png)
- ![Random disturbance angular velocity](D:/icarus/outputs/random_disturbance_omega.png)
- ![Random disturbance torque](D:/icarus/outputs/random_disturbance_torque.png)
- ![Time-step sensitivity](D:/icarus/outputs/timestep_sensitivity.png)

## Numerical summary
```json
{
  "inertia_matrix_kg_m2": [
    [
      0.0025361160416666672,
      -1.2e-05,
      8e-06
    ],
    [
      -1.2e-05,
      0.0025361160416666672,
      -1e-05
    ],
    [
      8e-06,
      -1e-05,
      0.002216666666666667
    ]
  ],
  "free_tumbling": {
    "final_omega_norm_rad_s": 0.5898674532287851,
    "max_omega_norm_rad_s": 0.5898674532287851,
    "max_total_torque_n_m": 0.0,
    "relative_energy_change": 0.007573805973908956,
    "relative_angular_momentum_change": 0.003839290497630807
  },
  "constant_dipole": {
    "final_omega_norm_rad_s": 0.6937187042571694,
    "max_omega_norm_rad_s": 0.6940683801719754,
    "max_total_torque_n_m": 6.5535639159162846e-06,
    "relative_energy_change": 704.950300880575,
    "relative_angular_momentum_change": 26.344923731853797,
    "steady_dipole_a_m2": [
      0.18,
      -0.08,
      0.05
    ],
    "steady_magnetic_field_t": [
      2.8e-05,
      -1.2e-05,
      4.1e-05
    ],
    "steady_magnetic_torque_n_m": [
      -2.6799999999999998e-06,
      -5.9799999999999995e-06,
      8.000000000000008e-08
    ]
  },
  "random_disturbance": {
    "final_omega_norm_rad_s": 0.10253221467330578,
    "max_omega_norm_rad_s": 0.1030300004091869,
    "max_total_torque_n_m": 7.802902868885395e-07,
    "relative_energy_change": 0.0020194629966537145,
    "relative_angular_momentum_change": 0.0012065110099636661,
    "rms_disturbance_torque_n_m": 1.8655571675483912e-07
  },
  "torque_vs_dipole_mapping": {
    "b_field_norm_t": 5.1078371156488533e-05,
    "max_torque_norm_n_m": 1.241219158730641e-05
  },
  "magnetic_underactuation": {
    "b_field_norm_t": 5.1078371156488533e-05,
    "dipole_magnitude_a_m2": 0.2,
    "max_abs_tau_parallel_n_m": 8.470329472543003e-22,
    "peak_torque_n_m": 1.0215674231297708e-05
  },
  "timestep_sensitivity": {
    "reference_dt_s": 0.005,
    "dt_values_s": [
      0.5,
      0.2,
      0.1,
      0.05,
      0.02
    ],
    "final_omega_error_rad_s": [
      0.005644274777459474,
      0.0022149872930820403,
      0.0010777146015764911,
      0.0005101692204568541,
      0.0001699909828619165
    ],
    "relative_final_energy_error": [
      0.020779611438442952,
      0.008129936868678028,
      0.0039516781252946455,
      0.0018697042459293563,
      0.0006228063677239262
    ],
    "best_dt_s": 0.02,
    "worst_dt_s": 0.5
  },
  "generated_files": [
    "constant_dipole_energy.png",
    "constant_dipole_omega.png",
    "free_tumbling_energy.png",
    "free_tumbling_omega.png",
    "magnetic_underactuation.png",
    "random_disturbance_omega.png",
    "random_disturbance_torque.png",
    "timestep_sensitivity.png",
    "torque_vs_dipole_mapping.png",
    "verification_report.md",
    "verification_summary.json"
  ]
}
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
