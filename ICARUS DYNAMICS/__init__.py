"""Spacecraft rotational dynamics and magnetorquer plant models."""

from .plant import (
    MagnetorquerTorqueMapper,
    RigidBodyPlant,
    SimulationResult,
    cubesat_1u_inertia_matrix,
)

__all__ = [
    "MagnetorquerTorqueMapper",
    "RigidBodyPlant",
    "SimulationResult",
    "cubesat_1u_inertia_matrix",
]
