import numpy as np


def generate_params():
    return {
        "gravity": 9.81,
        "mass": 0.2,
        "restitution": 0.8,
    }


def dynamics(t, state, params):
    height, velocity = state
    return np.array([velocity, -params["gravity"]])


def handle_impact(state, params):
    state = state.copy()

    if state[0] <= 0 and state[1] < 0:
        state[0] = 0
        state[1] *= -params["restitution"]

    return state


def calculate_energy(state, params):
    height = np.maximum(state[0], 0)
    velocity = state[1]

    kinetic = 0.5 * params["mass"] * velocity**2
    potential = params["mass"] * params["gravity"] * height

    return kinetic, potential