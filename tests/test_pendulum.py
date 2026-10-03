import numpy as np

from integrators import rk4
from models import pendulum

TIMESTEP = 0.01
NUMBER_OF_STEPS = 100


def simulate(params, initial_state):
    """Return the state trajectory, shape (2, NUMBER_OF_STEPS + 1)."""
    state_traj = np.zeros((2, NUMBER_OF_STEPS + 1))
    state_traj[:, 0] = initial_state
    for step in range(NUMBER_OF_STEPS):
        state_traj[:, step + 1] = rk4(
            pendulum.dynamics, step * TIMESTEP, state_traj[:, step], TIMESTEP, params
        )
    return state_traj


def total_energy(state_traj, params):
    kinetic, potential = pendulum.calculate_energy(state_traj, params)
    return kinetic + potential


def test_energy_conserved_without_torque_or_damping():
    params = pendulum.generate_params()
    params["torque"] = 0.0
    params["damping_coeff"] = 0.0
    initial_state = np.array([np.pi / 4, 0.0])

    state_traj = simulate(params, initial_state)
    energy = total_energy(state_traj, params)

    assert not np.allclose(state_traj[:, -1], initial_state)
    assert np.all(np.isclose(np.diff(energy), 0.0, atol=1e-6))


def test_torque_work_equals_energy_change():
    # if no damping, d(energy)/dt = torque * angular_velocity.
    # a constant torque adds energy equal to torque * the change in angle
    params = pendulum.generate_params()
    params["torque"] = 0.5
    params["damping_coeff"] = 0.0
    initial_state = np.array([np.pi / 4, 0.0])

    state_traj = simulate(params, initial_state)
    energy = total_energy(state_traj, params)
    torque_work = params["torque"] * (state_traj[0] - initial_state[0])

    assert np.all(np.isclose(energy - energy[0], torque_work, atol=1e-6))
    # Guard against a vacuous pass: the torque did meaningful work.
    assert abs(torque_work[-1]) > 0.1


def test_damping_dissipates_energy():
    # With no torque, d(energy)/dt = -damping * angular_velocity**2 <= 0.
    params = pendulum.generate_params()
    params["torque"] = 0.0
    params["damping_coeff"] = 0.5
    initial_state = np.array([np.pi / 4, 1.0])

    state_traj = simulate(params, initial_state)
    energy = total_energy(state_traj, params)

    # allows round-off where the angular velocity, and so the dissipation, is ~0.
    assert np.all(np.diff(energy) <= 1e-9)
    assert energy[-1] < energy[0] - 0.1
