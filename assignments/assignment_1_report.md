# Assignment 1: Rimless-Wheel Stability Analysis

## Code review guide

With the project environment active, run these commands from the repository
root:

```console
python rimless_wheel_simulation.py
python rimless_wheel_stability.py
```

For review, the relevant files are:

1. `models/rimless_wheel.py` contains the continuous dynamics, impact guard,
   reset map, and baseline parameters.
2. `rimless_wheel_simulation.py` runs and animates the hybrid simulation.
3. `rimless_wheel_stability.py` computes the return map, Floquet multiplier,
   regions of attraction, and parameter sweeps.

The stability script also saves all report figures at 300 DPI in
`rimless_wheel_stability_plots/`.

## Model and baseline parameters

The baseline uses Earth gravity, a 1 m spoke length, 8 spokes, and an 8-degree
slope. The half-angle between adjacent spokes is:

```text
alpha = pi / N = 22.5 degrees
```

Between impacts, the stance spoke follows the inverted-pendulum dynamics:

```text
theta_dot = omega
omega_dot = (g / length) sin(theta)
```

The next spoke contacts the ground when:

```text
theta = gamma + alpha
```

The impact reset is:

```text
theta_after = theta_before - 2 alpha
omega_after = cos(2 alpha) omega_before
```

## Sanity checks

Before running the simulation, I expected the following baseline values:

| Check | Expected value | Observed value |
|---|---:|---:|
| Impact angle, `gamma + alpha` | 30.5000 degrees | 30.5000 degrees |
| Reset angle, `gamma - alpha` | -14.5000 degrees | -14.5000 degrees |
| Impact velocity ratio, `cos(2 alpha)` | 0.707107 | 0.707107 |

The 8-second simulation produced 13 impacts. Each reported impact occurred at
the expected guard angle and reset to the expected post-impact angle. The
angular velocities approached a repeatable step-to-step value.

I also compared the numerical return map with the analytical energy result:

```text
P(omega) = cos(2 alpha)
           sqrt(omega^2 + (4 g / length) sin(gamma) sin(alpha))
```

At the numerical fixed point, the numerical and analytical map values differed
by only `5.95e-9 rad/s`. The numerical Floquet multiplier was `0.500006`, while
the analytical result was `cos^2(2 alpha) = 0.500000`. These comparisons verify
the continuous integration, contact detection, and reset map together.

SciPy is optional. When available, the analysis uses `solve_ivp` event
detection. For these results, the existing fourth-order Runge--Kutta integrator
was used with bisection refinement of each guard crossing.

## State-space region of attraction

The baseline region of attraction was estimated on a 41 x 41 grid over:

```text
-alpha <= initial theta <= gamma + alpha
0 <= initial omega <= 3 rad/s
```

Each state was simulated for at most 35 impacts. It was classified as converged
when its five most recent post-impact velocities were all within `0.005 rad/s`
of the return-map fixed point. It was classified as failed if forward motion
stopped, another impact was not reached within 10 seconds, or convergence was
not reached within the impact limit.

![State-space region of attraction with the rolling limit cycle, impact reset, Poincare fixed point, and unstable equilibrium.](../rimless_wheel_stability_plots/region_of_attraction.png)

The blue region converges to the rolling limit cycle and covers `0.857` of the
baseline grid. The orange curve shows the continuous part of one cycle, and the
dotted orange line shows the instantaneous impact reset. The star marks the
fixed point on the post-impact Poincare section.

The equilibrium at `(theta, omega) = (0, 0)` is unstable, as expected for an
inverted pendulum. Therefore, the rolling limit cycle is the only stable
attractor in the modeled forward-rolling state space. Gray points are failed
rolling trajectories rather than a second attractor because the model does not
include fallen-wheel or additional ground-contact dynamics.

The curved basin boundary is physically reasonable. An initial state needs
enough energy to reach the first impact and retain enough velocity after the
plastic collision to complete later steps.

## One-dimensional return map and local convergence

Immediately after every impact, `theta = gamma - alpha`. Therefore, the
Poincare section is one-dimensional:

```text
omega_(k+1) = P(omega_k)
```

![One-dimensional Poincare return map with the identity line and fixed point.](../rimless_wheel_stability_plots/poincare_return_map.png)

The numerical return map crosses the identity line once:

```text
omega* = 1.445646 rad/s
```

The local slope was estimated with a centered perturbation of
`delta = 0.001 rad/s`:

```text
lambda = [P(omega* + delta) - P(omega* - delta)] / (2 delta)
       = 0.500006
```

Because `abs(lambda) < 1`, the rolling limit cycle is locally stable. A
multiplier close to 0.5 means that a small velocity error is approximately
halved after each impact. Iterations from four initial velocities show this
convergence.

![Post-impact velocity convergence from four initial conditions.](../rimless_wheel_stability_plots/limit_cycle_convergence.png)

## Effect of slope

The slope sweep holds `N = 8` fixed. A smaller 17 x 17 grid was used for each
RoA estimate to keep the full brute-force sweep fast.

| Slope | Fixed-point speed (rad/s) | Floquet multiplier | RoA fraction |
|---:|---:|---:|---:|
| 2 degrees | no feasible gait | -- | 0.000 |
| 4 degrees | 1.023473 | 0.499989 | 0.751 |
| 6 degrees | 1.252857 | 0.500004 | 0.810 |
| 8 degrees | 1.445646 | 0.500006 | 0.830 |
| 10 degrees | 1.614803 | 0.500001 | 0.841 |
| 12 degrees | 1.766948 | 0.500008 | 0.855 |

![Fixed-point velocity, Floquet multiplier, and RoA fraction versus slope.](../rimless_wheel_stability_plots/slope_sweep.png)

The fixed-point speed increases with slope because the wheel gains more
gravitational energy per step. The RoA fraction also generally increases
because more initial states can maintain forward rolling.

The multiplier stays near 0.5 because the analytical multiplier for this ideal
model is `cos^2(2 alpha)`. It depends on spoke geometry but not on slope.
Therefore, changing the slope changes the rolling speed and basin size without
changing the local convergence rate for fixed `N`.

At 2 degrees, the algebraic return-map equation has a formal solution, but that
speed is too low for the wheel to pass the inverted-pendulum energy barrier and
reach the next contact. The numerical hybrid simulation therefore finds no
physically reachable rolling fixed point.

The baseline RoA fraction is `0.857` on the 41 x 41 grid, whereas the 8-degree
sweep reports `0.830` on its 17 x 17 grid. This difference is due to grid
resolution, not a change in the dynamics.

## Effect of the number of spokes

This sweep fixes the slope at 8 degrees and varies `N` from 6 through 12.

| Spokes | Fixed-point speed (rad/s) | Floquet multiplier | RoA fraction |
|---:|---:|---:|---:|
| 6 | no feasible gait | -- | 0.000 |
| 7 | 1.227567 | 0.388737 | 0.761 |
| 8 | 1.445646 | 0.500006 | 0.830 |
| 9 | 1.628750 | 0.586850 | 0.855 |
| 10 | 1.788020 | 0.654507 | 0.869 |
| 11 | 1.930095 | 0.707721 | 0.882 |
| 12 | 2.059211 | 0.750010 | 0.893 |

![Fixed-point velocity, numerical and analytical Floquet multiplier, and RoA fraction versus spoke count.](../rimless_wheel_stability_plots/spoke_count_sweep.png)

More spokes reduce the angle between contacts and reduce the fractional speed
loss at impact. The fixed-point speed and estimated RoA fraction increase over
the feasible cases. However, the Floquet multiplier also moves toward one.
Thus, more spokes increase the measured basin fraction but cause slower local
step-to-step convergence. All reported multipliers remain below one, so the
corresponding rolling cycles remain locally stable.

For `N = 6`, adjacent spokes are separated by 60 degrees, and the reset retains
only `cos(60 degrees) = 0.5` of the pre-impact velocity. At an 8-degree slope,
the candidate fixed-point speed cannot pass the upright configuration during
the next step, so no feasible rolling attractor is found.

The reported RoA fractions are fractions of the selected numerical grids, not
coordinate-independent measures of basin volume. Because the angular range
changes with `alpha = pi / N`, they should be interpreted as practical
comparisons over the stated initial-condition ranges.
