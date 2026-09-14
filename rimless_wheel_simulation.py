import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from models import rimless_wheel as model
from integrators import rk4 as integrator

params = model.generate_params()

gravity = params["gravity"]
length = params["length"]
number_of_spokes = params["number_of_spokes"]
alpha = params["alpha"]
slope = params["slope"]

timestep = 0.001
sim_time = 8.0

initial_state = np.array([
    slope - alpha,  # theta = gamma - alpha
    1.0,            # initial angular velocity (rad/s)
])


# ------------------------------------------------------------
# store the output
# ------------------------------------------------------------
n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

contact_traj = np.zeros((2, n_timesteps))

impact_flags = np.zeros(n_timesteps, dtype=bool)

impact_count = 0


guard_angle = slope + alpha
reset_angle = slope - alpha
expected_speed_ratio = np.cos(2.0 * alpha)

print("RIMLESS WHEEL EXPECTED VALUES")
print("--------------------------------")
print(f"Guard angle (impact)      gamma + alpha = {np.rad2deg(guard_angle):.4f} deg")
print(f"Reset angle      gamma - alpha = {np.rad2deg(reset_angle):.4f} deg")
print(f"Velocity ratio   cos(2 alpha)  = {expected_speed_ratio:.6f}")
print()


# check logic
#if \(g(\theta)<0\), then \(\theta<\gamma+\alpha\): the next spoke has not hit yet
#if \(g(\theta)=0\), then \(\theta=\gamma+\alpha\): impact happens
#if \(g(\theta)>0\), then \(\theta>\gamma+\alpha\): numerically you have passed the impact point
# ------------------------------------------------------------
# Simulation loop
# ------------------------------------------------------------
for step, t in enumerate(time_traj[:-1]):

    current_state = state_traj[:, step]

    next_state = integrator.step(
        model.dynamics,
        t,
        current_state,
        timestep,
        params,
    )

    current_guard = model.impact_guard(t, current_state, params)
    next_guard = model.impact_guard(t + timestep, next_state, params)

    # Impact occurs when the guard crosses zero in the downhill direction.
    # here we use guard to check if the spoke contact the ground
    impact = (
        current_guard < 0.0
        and next_guard >= 0.0
        and next_state[1] > 0.0
    )

    if impact:
        impact_count += 1

        state_before = np.array([
            guard_angle,
            next_state[1],
        ])

        state_after = model.reset_dynamics(state_before, params)

        # --------------------------------------------------------
        # Move the stance-contact point to the next spoke.
        # --------------------------------------------------------
        old_contact = contact_traj[:, step]

        # Hub position at impact.
        hub_at_impact = old_contact + length * np.array([
            np.sin(state_before[0]),
            np.cos(state_before[0]),
        ])

        # The new stance spoke has angle theta+ after reset.
        new_contact = hub_at_impact - length * np.array([
            np.sin(state_after[0]),
            np.cos(state_after[0]),
        ])

        contact_traj[:, step + 1] = new_contact
        state_traj[:, step + 1] = state_after
        impact_flags[step + 1] = True


        print(f"IMPACT {impact_count} at t = {t + timestep:.4f} s")
        print(
            f"  before: theta = {np.rad2deg(state_before[0]):8.4f} deg, "
            f"omega = {state_before[1]:8.4f} rad/s"
        )
        print(
            f"  after : theta = {np.rad2deg(state_after[0]):8.4f} deg, "
            f"omega = {state_after[1]:8.4f} rad/s"
        )
        print()

    else:
        state_traj[:, step + 1] = next_state
        contact_traj[:, step + 1] = contact_traj[:, step]

    # Optional state printout every 0.5 seconds.
    print_interval = max(1, int(0.5 / timestep))
    if (step + 1) % print_interval == 0:
        state = state_traj[:, step + 1]
        print(
            f"t = {t + timestep:6.3f} s | "
            f"theta = {np.rad2deg(state[0]):8.3f} deg | "
            f"omega = {state[1]:8.3f} rad/s | "
            f"impacts = {impact_count}"
        )


# ------------------------------------------------------------
# Save the full state history so every state can be inspected.
# ------------------------------------------------------------
output = np.column_stack(
    (
        time_traj,
        state_traj[0],
        state_traj[1],
        contact_traj[0],
        contact_traj[1],
        impact_flags.astype(int),
    )
)


print()
print(f"Simulation finished with {impact_count} impacts.")


# ------------------------------------------------------------
# State plots
# ------------------------------------------------------------
plt.figure()
plt.plot(time_traj, np.rad2deg(state_traj[0]))
plt.axhline(
    np.rad2deg(guard_angle),
    linestyle="--",
    label="impact guard",
)
plt.axhline(
    np.rad2deg(reset_angle),
    linestyle=":",
    label="reset angle",
)
plt.xlabel("Time (s)")
plt.ylabel("Angle theta (deg)")
plt.title("Rimless wheel angle")
plt.legend()
plt.tight_layout()


plt.figure()
plt.plot(time_traj, state_traj[1])
plt.xlabel("Time (s)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Rimless wheel angular velocity")
plt.tight_layout()


plt.figure()
plt.plot(state_traj[0], state_traj[1])
plt.xlabel("Angle theta (rad)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Rimless wheel phase portrait")
plt.tight_layout()


# ------------------------------------------------------------
# Simple visual GUI / animation (to check if the simulation looks reasonable)
# ------------------------------------------------------------
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

frame_skip = max(1, int(0.02 / timestep))
frames = np.arange(0, n_timesteps, frame_skip)

all_contact_x = contact_traj[0]
xmin = np.min(all_contact_x) - 1.5 * length
xmax = np.max(all_contact_x) + 2.0 * length

ground_x = np.linspace(xmin, xmax, 500)
ground_y = -np.tan(slope) * ground_x

ax.plot(ground_x, ground_y, linewidth=2)
ax.set_aspect("equal")
ax.set_xlim(xmin, xmin + 6.0 * length)
ax.set_ylim(-2.0 * length, 2.5 * length)
ax.set_xlabel("Horizontal position (m)")
ax.set_ylabel("Vertical position (m)")
ax.set_title("Rimless wheel simulation")

spoke_lines = []
for _ in range(number_of_spokes):
    line, = ax.plot([], [], linewidth=2)
    spoke_lines.append(line)

hub_point, = ax.plot([], [], "o", markersize=9)
contact_point, = ax.plot([], [], "o", markersize=6)

state_text = ax.text(
    0.02,
    0.96,
    "",
    transform=ax.transAxes,
    va="top",
    family="monospace",
)


def update(frame_number):
    i = frames[frame_number]

    theta = state_traj[0, i]
    omega = state_traj[1, i]
    contact = contact_traj[:, i]

    hub = contact + length * np.array([
        np.sin(theta),
        np.cos(theta),
    ])

    # Angle of the hub-to-contact stance spoke in ordinary x-y coordinates.
    stance_direction = np.arctan2(
        contact[1] - hub[1],
        contact[0] - hub[0],
    )

    for k, line in enumerate(spoke_lines):
        phi = stance_direction + k * 2.0 * np.pi / number_of_spokes

        spoke_end = hub + length * np.array([
            np.cos(phi),
            np.sin(phi),
        ])

        line.set_data(
            [hub[0], spoke_end[0]],
            [hub[1], spoke_end[1]],
        )

    hub_point.set_data([hub[0]], [hub[1]])
    contact_point.set_data([contact[0]], [contact[1]])

    # Follow the wheel as it moves downhill.
    window_width = 6.0 * length
    ax.set_xlim(
        hub[0] - 2.0 * length,
        hub[0] - 2.0 * length + window_width,
    )

    impacts_so_far = np.count_nonzero(impact_flags[: i + 1])

    state_text.set_text(
        f"t     = {time_traj[i]:6.3f} s\n"
        f"theta = {np.rad2deg(theta):7.2f} deg\n"
        f"omega = {omega:7.3f} rad/s\n"
        f"impact count = {impacts_so_far}"
    )

    return spoke_lines + [hub_point, contact_point, state_text]


animation = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=20,
    blit=False,
    repeat=True,
)

plt.tight_layout()
plt.show()
