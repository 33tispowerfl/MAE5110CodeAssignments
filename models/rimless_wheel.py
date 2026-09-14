"""Continuous and impact dynamics for the passive rimless wheel."""

import numpy as np


def dynamics(t, state, params):
    """Return the continuous state derivative between spoke impacts."""
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
        mass * gravity * length * np.sin(angle)
    ) / (mass * length**2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def generate_params():
    """Return the baseline physical and geometric parameters."""
    number_of_spokes = 8

    params = {
        "gravity": 9.81,  # gravity (m/s^2)
        "length": 1.0,  # spoke length (m)
        "mass": 1.0,  # point mass at hub (kg)
        "number_of_spokes": number_of_spokes,
        "alpha": np.pi / number_of_spokes,
        "slope": np.deg2rad(8.0),  # slope angle gamma (rad)
    }

    return params


def impact_guard(t, state, params):
    """Return zero when the next spoke contacts the ground."""
    angle = state[0]

    slope = params["slope"]
    alpha = params["alpha"]

    return angle - (slope + alpha)


def reset_dynamics(state, params):
    """Apply the stance-coordinate and angular-velocity impact reset."""
    angle = state[0]
    angular_velocity = state[1]

    alpha = params["alpha"]

    new_angle = angle - 2 * alpha

    new_angular_velocity = np.cos(2 * alpha) * angular_velocity

    return np.array([new_angle, new_angular_velocity])


impact_guard.direction = 1
impact_guard.terminal = True
