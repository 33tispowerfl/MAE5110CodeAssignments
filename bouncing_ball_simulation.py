import numpy as np
import matplotlib.pyplot as plt

from models import bouncing_ball as model
from integrators import rk4 as integrator

params = model.generate_params()
initial_state = np.array([2.0, 0.0])

timestep = 1e-3
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

for step, t in enumerate(time_traj[:-1]):
    next_state = integrator.step(
        model.dynamics,
        t,
        state_traj[:, step],
        timestep,
        params,
    )

    state_traj[:, step + 1] = model.handle_impact(
        next_state,
        params,
    )

assert np.all(np.isfinite(state_traj))
assert np.min(state_traj[0]) >= 0

kinetic, potential = model.calculate_energy(state_traj, params)
total_energy = kinetic + potential

plt.figure()
plt.plot(time_traj, state_traj[0])
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Bouncing ball height")
plt.tight_layout()

plt.figure()
plt.plot(time_traj, total_energy)
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Bouncing ball total energy")
plt.tight_layout()

plt.show()