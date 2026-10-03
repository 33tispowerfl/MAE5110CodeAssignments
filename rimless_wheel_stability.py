"""Stability analysis for the passive rimless wheel.

The contact event is used as a Poincare section.  Just after every impact,
theta = gamma - alpha, so the return map has only one variable: omega.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from integrators import rk4
from models import rimless_wheel as model


# solve_ivp is preferred for event detection.  The assignment repository does
# not require SciPy, so the existing RK4 method is used when SciPy is absent.
try:
    from scipy.integrate import solve_ivp

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def make_params(slope_degrees=8.0, number_of_spokes=8):
    """Return model parameters for one choice of slope and spoke count."""
    params = model.generate_params()
    params["gravity"] = 9.81
    params["length"] = 1.0
    params["number_of_spokes"] = number_of_spokes
    params["alpha"] = np.pi / number_of_spokes
    params["slope"] = np.deg2rad(slope_degrees)
    return params


def integrate_to_impact(initial_state, params, maximum_time=10.0):
    """Integrate one continuous phase and return the pre-impact state.

    None is returned if the wheel reverses direction or does not reach the
    next spoke contact within maximum_time.
    """
    state = np.asarray(initial_state, dtype=float)
    guard_value = model.impact_guard(0.0, state, params)

    if state[1] <= 0.0:
        return None

    # An initial condition exactly on the guard is already at impact.
    if abs(guard_value) < 1e-12:
        return state.copy()

    # States beyond the next contact configuration are not physically valid
    # for the current stance spoke.
    if guard_value > 0.0:
        return None

    if SCIPY_AVAILABLE:
        return integrate_to_impact_scipy(state, params, maximum_time)

    return integrate_to_impact_rk4(state, params, maximum_time)


def integrate_to_impact_scipy(initial_state, params, maximum_time):
    """Event-based continuous integration using SciPy when available."""

    def impact_event(time, state):
        return model.impact_guard(time, state, params)

    impact_event.direction = model.impact_guard.direction
    impact_event.terminal = True

    def stopped_event(time, state):
        return state[1]

    stopped_event.direction = -1
    stopped_event.terminal = True

    solution = solve_ivp(
        lambda time, state: model.dynamics(time, state, params),
        (0.0, maximum_time),
        initial_state,
        events=(impact_event, stopped_event),
        rtol=1e-9,
        atol=1e-11,
        max_step=0.05,
    )

    if len(solution.t_events[0]) == 0:
        return None

    impact_state = solution.y_events[0][0]
    if impact_state[1] <= 0.0:
        return None
    return impact_state


def integrate_to_impact_rk4(initial_state, params, maximum_time):
    """RK4 fallback with bisection refinement of the guard crossing."""
    timestep = 0.02
    time = 0.0
    state = initial_state.copy()
    current_guard = model.impact_guard(time, state, params)

    while time < maximum_time:
        step_size = min(timestep, maximum_time - time)
        next_state = rk4(model.dynamics, time, state, step_size, params)
        next_guard = model.impact_guard(time + step_size, next_state, params)

        if current_guard < 0.0 and next_guard >= 0.0:
            # Refine the event time inside this RK4 step.  Reintegrating from
            # the left endpoint keeps the crossing much more accurate than
            # simply accepting the end of the fixed step.
            lower_time = 0.0
            upper_time = step_size

            for _ in range(20):
                middle_time = 0.5 * (lower_time + upper_time)
                middle_state = rk4(
                    model.dynamics, time, state, middle_time, params
                )
                middle_guard = model.impact_guard(
                    time + middle_time, middle_state, params
                )

                if middle_guard < 0.0:
                    lower_time = middle_time
                else:
                    upper_time = middle_time

            impact_state = rk4(
                model.dynamics, time, state, upper_time, params
            )
            impact_state[0] = params["slope"] + params["alpha"]

            if impact_state[1] <= 0.0:
                return None
            return impact_state

        # Reversal before contact means that forward rolling has failed.
        if next_state[1] <= 0.0:
            return None

        time += step_size
        state = next_state
        current_guard = next_guard

    return None


def one_step(omega, params):
    """Map one post-impact angular velocity to the next one.

    Returns (next_omega, success), where success is False if the wheel does
    not reach the next impact while rolling forward.
    """
    post_impact_angle = params["slope"] - params["alpha"]
    initial_state = np.array([post_impact_angle, omega])
    pre_impact_state = integrate_to_impact(initial_state, params)

    if pre_impact_state is None:
        return np.nan, False

    next_state = model.reset_dynamics(pre_impact_state, params)
    if next_state[1] <= 0.0:
        return np.nan, False

    return next_state[1], True


def evaluate_return_map(omega_values, params):
    """Evaluate P(omega) and use NaN where the next impact is not reached."""
    return_values = np.full(len(omega_values), np.nan)

    for index, omega in enumerate(omega_values):
        next_omega, success = one_step(omega, params)
        if success:
            return_values[index] = next_omega

    return return_values


def find_fixed_point(params, minimum_omega=0.01, maximum_omega=5.0):
    """Find P(omega) - omega = 0 using sampling followed by bisection."""
    test_omegas = np.linspace(minimum_omega, maximum_omega, 160)
    previous_omega = None
    previous_residual = None

    for omega in test_omegas:
        next_omega, success = one_step(omega, params)
        if not success:
            continue

        residual = next_omega - omega
        if abs(residual) < 1e-10:
            return omega

        if previous_residual is not None and residual * previous_residual < 0.0:
            lower_omega = previous_omega
            upper_omega = omega

            for _ in range(50):
                middle_omega = 0.5 * (lower_omega + upper_omega)
                middle_next, middle_success = one_step(middle_omega, params)

                if not middle_success:
                    lower_omega = middle_omega
                    continue

                middle_residual = middle_next - middle_omega
                if abs(middle_residual) < 1e-11:
                    return middle_omega

                if middle_residual * previous_residual > 0.0:
                    lower_omega = middle_omega
                    previous_residual = middle_residual
                else:
                    upper_omega = middle_omega

            return 0.5 * (lower_omega + upper_omega)

        previous_omega = omega
        previous_residual = residual

    return np.nan


def floquet_multiplier(fixed_point, params, perturbation=1e-3):
    """Estimate the slope of the numerical return map at its fixed point."""
    if not np.isfinite(fixed_point):
        return np.nan

    upper_value, upper_success = one_step(fixed_point + perturbation, params)
    lower_value, lower_success = one_step(fixed_point - perturbation, params)

    if not upper_success or not lower_success:
        return np.nan

    return (upper_value - lower_value) / (2.0 * perturbation)


def analytical_return_map(omega, params):
    """Closed-form map used only to check the numerical hybrid simulation."""
    gravity = params["gravity"]
    length = params["length"]
    slope = params["slope"]
    alpha = params["alpha"]
    speed_ratio = np.cos(2.0 * alpha)
    energy_gain = 4.0 * gravity * np.sin(slope) * np.sin(alpha) / length
    return speed_ratio * np.sqrt(omega**2 + energy_gain)


def iterate_return_map(initial_omega, params, number_of_steps=20):
    """Repeatedly apply the numerical return map."""
    omega_history = [initial_omega]
    omega = initial_omega

    for _ in range(number_of_steps):
        omega, success = one_step(omega, params)
        if not success:
            break
        omega_history.append(omega)

    return np.asarray(omega_history)


def sample_limit_cycle(fixed_point, params, timestep=0.002):
    """Sample one continuous step of the rolling limit cycle for plotting."""
    state = np.array([params["slope"] - params["alpha"], fixed_point])
    state_history = [state.copy()]
    time = 0.0

    while time < 10.0:
        next_state = rk4(model.dynamics, time, state, timestep, params)
        next_guard = model.impact_guard(time + timestep, next_state, params)

        if next_guard >= 0.0:
            impact_state = integrate_to_impact(state, params, timestep)
            if impact_state is not None:
                state_history.append(impact_state)
            break

        state_history.append(next_state.copy())
        state = next_state
        time += timestep

    return np.asarray(state_history)


def converges_to_limit_cycle(
    initial_state,
    params,
    fixed_point,
    maximum_impacts=35,
    minimum_impacts=8,
    recent_count=5,
    tolerance=5e-3,
):
    """Classify one state using only its recent post-impact velocities."""
    if not np.isfinite(fixed_point):
        return 0

    state = np.asarray(initial_state, dtype=float)
    recent_velocities = []

    for impact_count in range(1, maximum_impacts + 1):
        pre_impact_state = integrate_to_impact(state, params)
        if pre_impact_state is None:
            return 0

        state = model.reset_dynamics(pre_impact_state, params)
        if state[1] <= 0.0:
            return 0

        recent_velocities.append(state[1])
        if len(recent_velocities) > recent_count:
            recent_velocities.pop(0)

        if impact_count >= minimum_impacts and len(recent_velocities) == recent_count:
            errors = np.abs(np.asarray(recent_velocities) - fixed_point)
            if np.all(errors < tolerance):
                return 1

    return 0


def compute_roa(params, fixed_point, grid_size=41, maximum_omega=3.0):
    """Compute a brute-force region-of-attraction map on a uniform grid."""
    alpha = params["alpha"]
    slope = params["slope"]
    theta_values = np.linspace(-alpha, slope + alpha, grid_size)
    omega_values = np.linspace(0.0, maximum_omega, grid_size)
    roa_map = np.zeros((grid_size, grid_size), dtype=int)

    for omega_index, omega in enumerate(omega_values):
        for theta_index, theta in enumerate(theta_values):
            initial_state = np.array([theta, omega])
            roa_map[omega_index, theta_index] = converges_to_limit_cycle(
                initial_state, params, fixed_point
            )

    return theta_values, omega_values, roa_map


def sweep_slopes(slope_values, number_of_spokes=8, grid_size=17):
    """Compute fixed point, multiplier, and RoA fraction versus slope."""
    fixed_points = []
    multipliers = []
    roa_fractions = []

    for slope_degrees in slope_values:
        params = make_params(slope_degrees, number_of_spokes)
        fixed_point = find_fixed_point(params)
        multiplier = floquet_multiplier(fixed_point, params)
        _, _, roa_map = compute_roa(params, fixed_point, grid_size)

        fixed_points.append(fixed_point)
        multipliers.append(multiplier)
        roa_fractions.append(np.mean(roa_map))

        print(f"  gamma = {slope_degrees:4.1f} deg complete")

    return (
        np.asarray(fixed_points),
        np.asarray(multipliers),
        np.asarray(roa_fractions),
    )


def sweep_spoke_counts(spoke_counts, slope_degrees=8.0, grid_size=17):
    """Compute fixed point, multiplier, and RoA fraction versus spoke count."""
    fixed_points = []
    multipliers = []
    roa_fractions = []

    for number_of_spokes in spoke_counts:
        params = make_params(slope_degrees, number_of_spokes)
        fixed_point = find_fixed_point(params)
        multiplier = floquet_multiplier(fixed_point, params)
        _, _, roa_map = compute_roa(params, fixed_point, grid_size)

        fixed_points.append(fixed_point)
        multipliers.append(multiplier)
        roa_fractions.append(np.mean(roa_map))

        print(f"  N = {number_of_spokes:2d} complete")

    return (
        np.asarray(fixed_points),
        np.asarray(multipliers),
        np.asarray(roa_fractions),
    )


def plot_return_map(params, fixed_point):
    omega_values = np.linspace(0.05, 3.0, 180)
    return_values = evaluate_return_map(omega_values, params)

    figure = plt.figure(figsize=(6.5, 5.0))
    plt.plot(omega_values, return_values, label=r"Numerical $P(\omega_k)$")
    plt.plot(omega_values, omega_values, "k--", label=r"$\omega_{k+1}=\omega_k$")
    plt.plot(fixed_point, fixed_point, "ro", label=fr"Fixed point = {fixed_point:.3f}")
    plt.xlabel(r"Post-impact $\omega_k$ (rad/s)")
    plt.ylabel(r"Next post-impact $\omega_{k+1}$ (rad/s)")
    plt.title("Rimless-wheel Poincare return map")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return figure


def plot_limit_cycle_convergence(params, fixed_point):
    figure = plt.figure(figsize=(6.5, 5.0))

    for initial_omega in [0.9, 1.1, 2.0, 2.8]:
        history = iterate_return_map(initial_omega, params)
        plt.plot(range(len(history)), history, "o-", label=fr"$\omega_0={initial_omega}$")

    plt.axhline(fixed_point, color="black", linestyle="--", label=fr"$\omega^*={fixed_point:.3f}$")
    plt.xlabel("Step number k")
    plt.ylabel(r"Post-impact $\omega_k$ (rad/s)")
    plt.title("Convergence to the rolling limit cycle")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return figure


def plot_roa(theta_values, omega_values, roa_map, params):
    figure = plt.figure(figsize=(7.0, 5.0))
    colors = ListedColormap(["#d9d9d9", "#2b8cbe"])
    image = plt.pcolormesh(
        np.rad2deg(theta_values),
        omega_values,
        roa_map,
        shading="nearest",
        cmap=colors,
        vmin=0,
        vmax=1,
    )
    colorbar = plt.colorbar(image, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["Fails", "Converges"])

    fixed_point = find_fixed_point(params)
    limit_cycle = sample_limit_cycle(fixed_point, params)
    reset_angle = params["slope"] - params["alpha"]

    plt.plot(
        np.rad2deg(limit_cycle[:, 0]),
        limit_cycle[:, 1],
        color="#e66101",
        linewidth=2.5,
        label="Rolling limit cycle",
        zorder=3,
    )
    plt.plot(
        np.rad2deg([limit_cycle[-1, 0], limit_cycle[0, 0]]),
        [limit_cycle[-1, 1], limit_cycle[0, 1]],
        color="#e66101",
        linestyle=":",
        linewidth=2.0,
        label="Impact reset",
        zorder=3,
    )
    plt.plot(
        np.rad2deg(reset_angle),
        fixed_point,
        marker="*",
        color="black",
        markersize=12,
        linestyle="none",
        label="Poincare fixed point",
        zorder=4,
    )
    plt.plot(
        0.0,
        0.0,
        marker="x",
        color="black",
        markersize=8,
        linestyle="none",
        label="Unstable equilibrium",
        zorder=4,
    )
    plt.xlabel(r"Angle $\theta$ (deg)")
    plt.ylabel(r"Angular velocity $\omega$ (rad/s)")
    plt.title(
        "Rolling region of attraction\n"
        + fr"$\gamma={np.rad2deg(params['slope']):.0f}^\circ$, "
        + fr"$N={params['number_of_spokes']}$"
    )
    plt.legend(loc="upper left")
    plt.tight_layout()
    return figure


def plot_slope_sweep(slope_values, fixed_points, multipliers, roa_fractions):
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    axes[0].plot(slope_values, fixed_points, "o-")
    axes[0].set_ylabel(r"Fixed-point $\omega^*$ (rad/s)")

    axes[1].plot(slope_values, multipliers, "o-", label="Numerical")
    axes[1].axhline(1.0, color="black", linestyle="--", label=r"$|\lambda|=1$")
    axes[1].set_ylabel(r"Floquet multiplier $\lambda$")
    axes[1].legend()

    axes[2].plot(slope_values, roa_fractions, "o-")
    axes[2].set_ylabel("Convergent grid fraction")
    axes[2].set_ylim(-0.05, 1.05)

    for axis in axes:
        axis.set_xlabel(r"Slope $\gamma$ (deg)")
        axis.grid(alpha=0.3)

    figure.suptitle("Effect of slope inclination")
    figure.tight_layout()
    return figure


def plot_spoke_sweep(spoke_counts, fixed_points, multipliers, roa_fractions):
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    axes[0].plot(spoke_counts, fixed_points, "o-")
    axes[0].set_ylabel(r"Fixed-point $\omega^*$ (rad/s)")

    axes[1].plot(spoke_counts, multipliers, "o-", label="Numerical")
    axes[1].plot(
        spoke_counts,
        np.cos(2.0 * np.pi / spoke_counts) ** 2,
        "k--",
        label="Analytical",
    )
    axes[1].axhline(1.0, color="gray", linestyle=":", label=r"$|\lambda|=1$")
    axes[1].set_ylabel(r"Floquet multiplier $\lambda$")
    axes[1].legend()

    axes[2].plot(spoke_counts, roa_fractions, "o-")
    axes[2].set_ylabel("Convergent grid fraction")
    axes[2].set_ylim(-0.05, 1.05)

    for axis in axes:
        axis.set_xlabel("Number of spokes N")
        axis.grid(alpha=0.3)

    figure.suptitle("Effect of the number of spokes")
    figure.tight_layout()
    return figure


def save_figures(figures):
    """Save all report figures beside this script in a dedicated folder."""
    output_directory = Path(__file__).resolve().parent / "rimless_wheel_stability_plots"
    output_directory.mkdir(exist_ok=True)

    filenames = [
        "poincare_return_map.png",
        "limit_cycle_convergence.png",
        "region_of_attraction.png",
        "slope_sweep.png",
        "spoke_count_sweep.png",
    ]

    for figure, filename in zip(figures, filenames):
        figure.savefig(output_directory / filename, dpi=300, bbox_inches="tight")

    return output_directory


def main():
    baseline_params = make_params(slope_degrees=8.0, number_of_spokes=8)
    fixed_point = find_fixed_point(baseline_params)

    if not np.isfinite(fixed_point):
        raise RuntimeError("No rolling fixed point was found for the baseline parameters.")

    numerical_multiplier = floquet_multiplier(fixed_point, baseline_params)
    analytical_multiplier = np.cos(2.0 * baseline_params["alpha"]) ** 2

    analytical_at_fixed_point = analytical_return_map(fixed_point, baseline_params)
    map_error = abs(analytical_at_fixed_point - fixed_point)

    print("Computing baseline region of attraction ...")
    theta_values, omega_values, roa_map = compute_roa(
        baseline_params, fixed_point, grid_size=41
    )
    roa_fraction = np.mean(roa_map)

    figures = [
        plot_return_map(baseline_params, fixed_point),
        plot_limit_cycle_convergence(baseline_params, fixed_point),
        plot_roa(theta_values, omega_values, roa_map, baseline_params),
    ]

    slope_values = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    print("Computing slope sweep ...")
    slope_results = sweep_slopes(slope_values, number_of_spokes=8)
    figures.append(plot_slope_sweep(slope_values, *slope_results))

    spoke_counts = np.arange(6, 13)
    print("Computing spoke-count sweep ...")
    spoke_results = sweep_spoke_counts(spoke_counts, slope_degrees=8.0)
    figures.append(plot_spoke_sweep(spoke_counts, *spoke_results))

    output_directory = save_figures(figures)

    print("\nSLOPE SWEEP RESULTS")
    print("gamma (deg)    omega* (rad/s)    multiplier    RoA fraction")
    for slope_degrees, omega, multiplier, fraction in zip(
        slope_values, *slope_results
    ):
        print(f"{slope_degrees:11.1f}    {omega:15.6f}    {multiplier:10.6f}    {fraction:12.3f}")

    print("\nSPOKE-COUNT SWEEP RESULTS")
    print("N              omega* (rad/s)    multiplier    RoA fraction")
    for spoke_count, omega, multiplier, fraction in zip(
        spoke_counts, *spoke_results
    ):
        print(
            f"{spoke_count:1d}              {omega:15.6f}    "
            f"{multiplier:10.6f}    {fraction:12.3f}"
        )

    print("\nBASELINE STABILITY SUMMARY")
    print("--------------------------")
    print(f"Integration method: {'solve_ivp' if SCIPY_AVAILABLE else 'RK4 fallback'}")
    print(f"gamma = {np.rad2deg(baseline_params['slope']):.1f} deg")
    print(f"N = {baseline_params['number_of_spokes']}")
    print(f"alpha = {np.rad2deg(baseline_params['alpha']):.4f} deg")
    print(f"omega* = {fixed_point:.6f} rad/s")
    print(f"Floquet multiplier (numerical)  = {numerical_multiplier:.6f}")
    print(f"Floquet multiplier (analytical) = {analytical_multiplier:.6f}")
    print(f"Numerical/analytical map error at omega* = {map_error:.3e} rad/s")
    print(f"Locally stable? {abs(numerical_multiplier) < 1.0}")
    print(f"RoA fraction = {roa_fraction:.3f}")
    print(f"Plots saved to: {output_directory}")
    print("\nClose the figure windows to end the program.")

    plt.show()


if __name__ == "__main__":
    main()
