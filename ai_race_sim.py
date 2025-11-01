# AI Race System Dynamics + Game (SciPy-only) — Minimal Working Prototype
# ----------------------------------------------------------------------------------
# This single-file prototype implements a hybrid SD + game loop without PySD/BPTK.
# Dependencies: numpy, scipy, matplotlib (all standard). No internet needed.
#
# What it does:
# - Defines bloc states (US, CN, EU) with stocks: K (capital), E (power), S (safety),
#   T (trust), P (political capital), plus a global scarce resource R.
# - At each discrete step, computes bloc actions a = (aS, aV, aX) by a coarse
#   best-response grid search given current state (myopic).
# - Integrates continuous dynamics over that step via scipy.integrate.solve_ivp
#   with actions held constant during the step.
# - Logs and plots key outputs.
#
# Notes:
# - This is intentionally compact and readable to serve as a good starting skeleton.
# - Many functional forms are placeholders but smooth and well-behaved.
# - You can refine payoffs, constraints, and parameterization as you calibrate.
#
# License: Apache 2.0 (feel free to use/modify).
import math
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)


# ---------------------------
# Model parameters
# ---------------------------
@dataclass
class Params:
    # Time / integration
    dt: float = 0.25  # years per decision step (quarterly)
    T: float = 10.0  # total simulation years
    rel_tol: float = 1e-6
    abs_tol: float = 1e-8

    # Depreciation / decay
    delta_K: float = 0.08  # capital obsolescence
    delta_E: float = 0.03  # energy capacity retirement
    delta_S: float = 0.04  # safety knowledge depreciation
    delta_T: float = 0.05  # trust decay
    delta_R: float = 0.02  # resource depletion baseline

    # Investment/productivity baselines
    inv_scale_K: float = 1.5  # scale linking acceleration to capital invest
    inv_scale_S: float = 0.9  # scale linking safety effort to safety invest
    build_E_scale: float = 0.8  # scale for power buildout
    efficiency_growth: float = 0.25  # annualized efficiency trend (fraction)

    # Scarcity / price
    R0: float = 100.0  # initial/nominal resource stock (global)
    p0: float = 1.0  # base resource price
    eta: float = 1.2  # scarcity curvature (>1 => convex)
    rho_base: float = 4.0  # baseline replenishment (resource units / yr)
    coop_build_bonus: float = 0.8  # how much co-op boosts replenishment

    # Spillovers
    phi_spill: float = 0.12  # strength of safety spillover via verification

    # Politics / backlash weights (must add up to 1)
    psi_success: float = 0.15
    psi_incident: float = 0.6
    psi_grid_stress: float = 0.25

    # Payoff weights
    beta_first_mover: float = 2.0
    kappa_risk_cost: float = 3.0
    sigma_spill_payoff: float = 0.8

    # Risk model params
    base_hazard: float = 0.02  # baseline hazard
    hazard_cap_elastic: float = 0.7
    hazard_safety_elastic: float = 1.0
    hazard_verif_factor: float = 0.6  # multiplicative reduction when verified

    # Political / budget caps (soft)
    invest_cap_scale: float = 2.0  # investment cap proportional to P
    power_build_lag: float = 0.75  # slows build E (appears inside f)
    grid_stress_threshold: float = 1.0  # demand / supply threshold

    # Lead probability softmax temperature
    lead_temp: float = 0.5

    # Resource usage per compute (scaling constant)
    resource_per_compute: float = 0.08

    # Decision grid resolution
    grid_aS: int = 5
    grid_aX: int = 5
    grid_aV: int = 2  # 0 or 1 for now


# ---------------------------
# State representation
# ---------------------------
# Order of blocs
BLOCS = ["US", "CN", "EU"]
NB = len(BLOCS)


# State vector y layout:
# For each bloc i: [K_i, E_i, S_i, T_i, P_i]  => 5 * NB entries
# Then global resource R at the end
def pack_state(K, E, S, T, P, R):
    return np.concatenate([K, E, S, T, P, np.array([R])])


def unpack_state(y):
    K = y[0:NB]
    E = y[NB : 2 * NB]
    S = y[2 * NB : 3 * NB]
    T = y[3 * NB : 4 * NB]
    P = y[4 * NB : 5 * NB]
    R = y[-1]
    return K, E, S, T, P, R


# ---------------------------
# Helper functions
# ---------------------------
def resource_price(R, p0, R0, eta):
    # Convex scarcity price
    R_eff = max(R, 1e-6)
    return p0 * (R0 / R_eff) ** eta


def compute_capacity(K, E, R, params: Params):
    """Effective compute capacity per bloc constrained by (capital, energy, resource)."""
    # Capital-to-compute map with efficiency growth (handled externally via trend factor)
    fK = np.sqrt(np.maximum(K, 0.0))  # concave in capital
    # Energy bottleneck
    fE = np.sqrt(np.maximum(E, 1e-9))
    # Resource bottleneck shared -> scale everyone by resource tightness
    r_factor = max(min(R / params.R0, 1.0), 1e-3)  # clamp [1e-3, 1]
    fR = r_factor
    C = np.minimum(fK, fE) * fR
    return C


def efficiency_multiplier(t_years, g):
    # Continuous compounding approximation to performance/efficiency trend
    return math.exp(g * t_years)


def lead_prob(C_eff, aX, temp):
    # Softmax over aggressive compute * acceleration
    score = C_eff * (0.5 + 0.5 * aX)  # ensure >0
    # softmax with temperature
    ex = np.exp(score / max(temp, 1e-6))
    return ex / np.sum(ex)


def hazard_rate(C_eff_i, aS_i, S_i, verif_level, params: Params):
    # Simple hazard declines with safety effort & stock, and with verification
    base = params.base_hazard
    cap_term = C_eff_i**params.hazard_cap_elastic
    safety_term = 1.0 / (1.0 + aS_i + S_i) ** params.hazard_safety_elastic
    verif_term = params.hazard_verif_factor if verif_level > 0.5 else 1.0
    return base * cap_term * safety_term * verif_term


def political_drift(success_i, incident_i, grid_stress_i, params: Params):
    return (
        params.psi_success * success_i
        - params.psi_incident * incident_i
        - params.psi_grid_stress
        * max(0.0, grid_stress_i - params.grid_stress_threshold)
    )


# ---------------------------
# Game / action selection
# ---------------------------
def payoff_for_bloc(i, actions, state, t_years, params: Params):
    """Compute per-bloc instantaneous payoff given all actions (myopic)."""
    K, E, S, T, P, R = unpack_state(state)

    aS = actions[:, 0]
    aV = actions[:, 1]
    aX = actions[:, 2]

    # Effective compute with efficiency trend
    C_raw = compute_capacity(K, E, R, params)
    C_eff = C_raw * efficiency_multiplier(t_years, params.efficiency_growth)

    # First-mover probability
    p_lead = lead_prob(C_eff, aX, params.lead_temp)

    # Risk construction
    verif_level = np.mean(aV)  # crude: global verification climate
    hazards = np.array(
        [hazard_rate(C_eff[j], aS[j], S[j], verif_level, params) for j in range(NB)]
    )
    # Systemic incident probability approximation
    p_incident = 1.0 - np.exp(-np.sum(hazards))

    # Scarcity price
    pR = resource_price(R, params.p0, params.R0, params.eta)
    use_i = (
        params.resource_per_compute * C_eff[i] * (0.5 + 0.5 * aX[i])
    )  # usage grows with aggressiveness

    # Spillover benefit from verification
    spill_i = (
        params.sigma_spill_payoff
        * aV[i]
        * np.mean(S[np.arange(NB) != i])
        * params.phi_spill
    )

    # Domestic political grid stress proxy (actual energy usage vs. capacity)
    energy_usage_i = C_eff[i] * (0.5 + 0.5 * aX[i])  # actual energy demand
    energy_capacity_i = max(E[i], 1e-6)
    grid_stress_i = energy_usage_i / energy_capacity_i

    payoff = (
        params.beta_first_mover * p_lead[i]
        - params.kappa_risk_cost * p_incident
        - pR * use_i
        - grid_stress_i * 0.05
        + spill_i
    )
    return payoff


def choose_actions_myopic(state, t_years, params: Params):
    """
    Coarse global best-response search on a grid for all players simultaneously.
    For speed and simplicity in this prototype, do a naive joint search by
    evaluating a small grid and taking the argmax of the sum of payoffs.
    (You can swap for iterative BR if desired.)
    """
    grid_S = np.linspace(0.0, 1.0, params.grid_aS)
    grid_X = np.linspace(0.0, 1.0, params.grid_aX)
    grid_V = np.linspace(0.0, 1.0, params.grid_aV)  # {0,1} if grid_aV=2

    # Build candidate actions per bloc
    cand = np.array([(s, v, x) for s in grid_S for v in grid_V for x in grid_X])
    # To keep combinatorics tractable, we tie blocs to identical candidates (symmetry heuristic).
    # You can remove this and do per-bloc product search or iterative BR later.
    best_actions = None
    best_value = -1e18

    for idx in range(cand.shape[0]):
        a = np.vstack([cand[idx]] * NB)  # same action across blocs (fast prototype)
        # Value = sum of payoffs (utilitarian selection); replace with Nash search later
        value = sum(payoff_for_bloc(i, a, state, t_years, params) for i in range(NB))
        if value > best_value:
            best_value = value
            best_actions = a.copy()

    return best_actions  # shape (NB, 3): columns = [aS, aV, aX]


def choose_actions_iterative_best_response(
    state, t_years, params: Params, max_iterations: int = 10, tolerance: float = 1e-4
):
    """
    Iterative best-response dynamics: each bloc sequentially optimizes their action
    given the other blocs' current actions, repeating until convergence.

    Args:
        state: Current state vector
        t_years: Current time in years
        params: Model parameters
        max_iterations: Maximum number of iterations through all blocs
        tolerance: Convergence threshold (max action change)

    Returns:
        actions: Array of shape (NB, 3) with columns [aS, aV, aX]
    """
    grid_S = np.linspace(0.0, 1.0, params.grid_aS)
    grid_X = np.linspace(0.0, 1.0, params.grid_aX)
    grid_V = np.linspace(0.0, 1.0, params.grid_aV)

    # Build candidate actions (all combinations)
    cand = np.array([(s, v, x) for s in grid_S for v in grid_V for x in grid_X])

    # Initialize with middle-of-grid actions
    actions = np.ones((NB, 3)) * 0.5

    for iteration in range(max_iterations):
        actions_old = actions.copy()

        # Each bloc takes a turn optimizing
        for i in range(NB):
            best_action_i = actions[i].copy()
            best_payoff_i = -1e18

            # Try each candidate action for bloc i
            for idx in range(cand.shape[0]):
                # Create trial action profile: others unchanged, bloc i tries candidate
                trial_actions = actions.copy()
                trial_actions[i] = cand[idx]

                # Compute payoff for bloc i under this action profile
                payoff_i = payoff_for_bloc(i, trial_actions, state, t_years, params)

                if payoff_i > best_payoff_i:
                    best_payoff_i = payoff_i
                    best_action_i = cand[idx].copy()

            # Update bloc i's action to their best response
            actions[i] = best_action_i

        # Check convergence: max absolute change across all actions
        max_change = np.max(np.abs(actions - actions_old))
        if max_change < tolerance:
            break

    return actions


# ---------------------------
# Continuous dynamics
# ---------------------------
def rhs_continuous(t, y, actions, params: Params):
    """RHS for solve_ivp over one decision interval with fixed actions."""
    K, E, S, T, P, R = unpack_state(y)

    aS = actions[:, 0]
    aV = actions[:, 1]
    aX = actions[:, 2]

    # Investment caps via political capital (soft saturation)
    invest_cap = params.invest_cap_scale * np.maximum(P, 0.0)
    I_K = np.minimum(invest_cap, params.inv_scale_K * aX * (1.0 + P))
    I_S = np.minimum(invest_cap, params.inv_scale_S * aS * (1.0 + P))

    # Power buildout slowed by lag and politics
    B_E = (params.build_E_scale * (0.5 + 0.5 * aX) * (1.0 + 0.5 * P)) / (
        1.0 + params.power_build_lag
    )

    # Effective compute for resource usage estimates
    C_raw = compute_capacity(K, E, R, params)
    C_eff = C_raw * efficiency_multiplier(t, params.efficiency_growth)
    use = params.resource_per_compute * C_eff * (0.5 + 0.5 * aX)

    # Resource replenishment boosted by cooperation (sum aV)
    rho = params.rho_base + params.coop_build_bonus * np.sum(aV)

    dK = I_K - params.delta_K * K
    dE = B_E - params.delta_E * E
    dS = (
        I_S
        + params.phi_spill * (aV[:, None] * S).sum(axis=0) / max(NB - 1, 1)
        - params.delta_S * S
    )
    dT = 0.2 * aV - params.delta_T * T  # trust builds with verification

    # Prevent stocks from going negative (non-negativity constraints)
    # If a stock is at/near zero and its derivative is negative, clamp to zero
    dK = np.where((K <= 1e-6) & (dK < 0), 0.0, dK)
    dE = np.where((E <= 1e-6) & (dE < 0), 0.0, dE)
    dS = np.where((S <= 1e-6) & (dS < 0), 0.0, dS)
    # Incidents proxy (not Poisson here; we use hazard sum to push P)
    verif_level = np.mean(aV)
    hazards = np.array(
        [hazard_rate(C_eff[j], aS[j], S[j], verif_level, params) for j in range(NB)]
    )
    p_incident = 1.0 - np.exp(-np.sum(hazards) * 0.25)  # scaled in interval
    # Grid stress proxy (actual energy usage vs. capacity)
    energy_usage = C_eff * (0.5 + 0.5 * aX)  # actual energy demand
    grid_stress = energy_usage / np.maximum(E, 1e-6)

    dP = np.array(
        [
            political_drift(
                success_i=C_eff[i],
                incident_i=p_incident,
                grid_stress_i=grid_stress[i],
                params=params,
            )
            for i in range(NB)
        ]
    )

    dR = rho - np.sum(use) - params.delta_R * R

    # Prevent global resource from going negative
    if R <= 1e-6 and dR < 0:
        dR = 0.0

    return np.concatenate([dK, dE, dS, dT, dP, np.array([dR])])


# ---------------------------
# Simulation driver
# ---------------------------
def simulate(
    initial_state: np.ndarray,
    params: Params,
    seed: int = 0,
    use_iterative_br: bool = False,
):
    """
    Run simulation with discrete action selection and continuous integration.

    Args:
        initial_state: Initial state vector
        params: Model parameters
        seed: Random seed (for future stochastic extensions)
        use_iterative_br: If True, use iterative best-response; if False, use myopic joint search
    """
    rng = np.random.default_rng(seed)
    horizon_steps = int(params.T / params.dt)

    # Logs
    times = [0.0]
    states = [initial_state.copy()]
    actions_log = []

    y = initial_state.copy()
    t = 0.0

    for step in range(horizon_steps):
        # Choose actions given current state
        if use_iterative_br:
            a = choose_actions_iterative_best_response(y, t, params)
        else:
            a = choose_actions_myopic(y, t, params)
        actions_log.append(a.copy())

        # Integrate over [t, t+dt] with actions held constant
        sol = solve_ivp(
            fun=lambda tt, yy: rhs_continuous(tt, yy, a, params),
            t_span=(t, t + params.dt),
            y0=y,
            rtol=params.rel_tol,
            atol=params.abs_tol,
            method="RK45",
            max_step=params.dt / 8.0,
        )
        y = sol.y[:, -1]
        t = t + params.dt
        times.append(t)
        states.append(y.copy())

    times = np.array(times)
    states = np.vstack(states)
    actions_arr = np.stack(actions_log, axis=0)  # [steps, NB, 3]
    return times, states, actions_arr


# ---------------------------
# Demo run
# ---------------------------
def default_initial_state():
    # Initialize blocs with slightly different strengths
    # US, CN, EU
    K0 = np.array([12.0, 9.0, 7.0])  # capital
    E0 = np.array([10.0, 8.0, 7.0])  # power capacity
    S0 = np.array([2.0, 1.2, 1.5])  # safety
    T0 = np.array([0.8, 0.4, 0.9])  # trust
    P0 = np.array([1.5, 1.2, 1.1])  # political capital
    R0 = 100.0  # global resource
    return pack_state(K0, E0, S0, T0, P0, R0)


def plot_results(times, states, actions, params: Params):
    K, E, S, T, P, R = unpack_state(states.T)

    # Recompute time-series C_eff for plotting
    C_eff_series = []
    R_series = states[:, -1]
    for idx, t in enumerate(times):
        Kt, Et, St, Tt, Pt, Rt = unpack_state(states[idx])
        C_raw_t = compute_capacity(Kt, Et, Rt, params)
        C_eff_t = C_raw_t * efficiency_multiplier(t, params.efficiency_growth)
        C_eff_series.append(C_eff_t)
    C_eff_series = np.array(C_eff_series)  # shape [T, NB]

    # Actions by component
    aS = actions[:, :, 0]
    aV = actions[:, :, 1]
    aX = actions[:, :, 2]

    # 1) Plot compute capacity time series (one figure)
    plt.figure()
    for i, bloc in enumerate(BLOCS):
        plt.plot(times, C_eff_series[:, i], label=f"{bloc}")
    plt.title("Effective Compute (constrained)")
    plt.xlabel("Years")
    plt.ylabel("C_eff (arb units)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/fig_compute.png", dpi=160)

    # 2) Plot resource stock R
    plt.figure()
    plt.plot(times, R_series)
    plt.title("Global Scarce Resource R")
    plt.xlabel("Years")
    plt.ylabel("R")
    plt.tight_layout()
    plt.savefig("plots/fig_resource.png", dpi=160)

    # 3) Safety effort
    plt.figure()
    for i, bloc in enumerate(BLOCS):
        plt.step(times[:-1], aS[:, i], where="post", label=f"{bloc}")
    plt.title("Safety Effort aS")
    plt.xlabel("Years")
    plt.ylabel("aS")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/fig_aS.png", dpi=160)

    # 4) Verification
    plt.figure()
    for i, bloc in enumerate(BLOCS):
        plt.step(times[:-1], aV[:, i], where="post", label=f"{bloc}")
    plt.title("Verification / Cooperation aV")
    plt.xlabel("Years")
    plt.ylabel("aV")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/fig_aV.png", dpi=160)

    # 5) Acceleration
    plt.figure()
    for i, bloc in enumerate(BLOCS):
        plt.step(times[:-1], aX[:, i], where="post", label=f"{bloc}")
    plt.title("Acceleration aX")
    plt.xlabel("Years")
    plt.ylabel("aX")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/fig_aX.png", dpi=160)


def main():
    use_iterative_br = True
    params = Params()
    y0 = default_initial_state()
    times, states, actions = simulate(
        y0, params, seed=42, use_iterative_br=use_iterative_br
    )

    # Save numpy outputs if needed
    np.save("times.npy", times)
    np.save("states.npy", states)
    np.save("actions.npy", actions)

    # Plots
    plot_results(times, states, actions, params)


if __name__ == "__main__":
    main()
