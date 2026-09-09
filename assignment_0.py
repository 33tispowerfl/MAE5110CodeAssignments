import numpy as np
import matplotlib.pyplot as plt
import timeit

from models import pendulum as model
from integrators import explicit_euler as integrator
from integrators import explicit_euler, rk4

# Basic simulation of the pendulum

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 0.2,  # point mass at end of rod (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
}


# some set-up
initial_state = np.array([np.pi / 4, 0.0])

timestep = 0.000202359
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

# simulation loop
for step, t in enumerate(time_traj[:-1]):
    state_traj[:, step + 1] = integrator.step(
        model.dynamics,
        t,
        state_traj[:, step],
        timestep,
        params,
    )

# sanity check the energies: since there is no actuation, and no damping, total energy should stay
# constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
# a stand-still.

kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)


def simulate(method, dt):
    state = initial_state.copy()

    for step in range(int(sim_time / dt)):
        state = method.step(model.dynamics, step * dt, state, dt, params)

    return state


def is_accurate(method, dt, tolerance=0.01):
    state = initial_state.copy()
    kinetic, potential = model.calculate_energy(state, params)
    initial_energy = kinetic + potential
    energy_scale = max(abs(initial_energy), np.finfo(float).eps)

    for step in range(int(sim_time / dt)):
        state = method.step(model.dynamics, step * dt, state, dt, params)

        kinetic, potential = model.calculate_energy(state, params)
        energy = kinetic + potential
        error = abs(energy - initial_energy) / energy_scale

        if not np.isfinite(energy) or error > tolerance:
            return False

    return True


def largest_accurate_timestep(method):
    for candidate_timestep in np.logspace(-1, -5, 50):
        if is_accurate(method, candidate_timestep):
            return candidate_timestep

    raise RuntimeError("No tested timestep met the energy tolerance")


def measure_runtime(method, dt):
    return min(
        timeit.repeat(
            lambda: simulate(method, dt),
            number=1,
            repeat=3,
        )
    )


euler_timestep = largest_accurate_timestep(explicit_euler)
rk4_timestep = largest_accurate_timestep(rk4)
common_timestep = min(euler_timestep, rk4_timestep)

print(f"Largest accurate Euler timestep: {euler_timestep:.6g} s")
print(f"Largest accurate RK4 timestep:   {rk4_timestep:.6g} s")
print(f"\nRuntime at the same timestep ({common_timestep:.6g} s):")
print(f"Euler: {measure_runtime(explicit_euler, common_timestep):.6f} s")
print(f"RK4:   {measure_runtime(rk4, common_timestep):.6f} s")
print("\nRuntime at each method's largest accurate timestep:")
print(f"Euler: {measure_runtime(explicit_euler, euler_timestep):.6f} s")
print(f"RK4:   {measure_runtime(rk4, rk4_timestep):.6f} s")

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Pendulum energy")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(state_traj[0], state_traj[1])
plt.xlabel("Angle (rad)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Pendulum phase portrait")
plt.tight_layout()
plt.show()
