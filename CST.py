from dataclasses import dataclass
import math
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

###############################################################################
# 1. Model configuration
###############################################################################

# We'll index players as 0,1,2 = [US, CN, EU] for convenience.
N_PLAYERS = 3
US, CN, EU = 0, 1, 2  # useful named indices


@dataclass
class Params:
    # Capability growth efficiency
    alpha: float  # dK/dt contribution from acceleration effort

    # Safety growth efficiency
    gamma: float  # dS/dt contribution from safety effort

    # Spillover strength for safety via trust
    eta: float    # multiplier on T * avg_other_S

    # Trust formation / decay
    beta: float      # how fast joint verification effort builds trust
    delta_T: float   # trust decay rate

    # Safety effectiveness for diagnostic debt
    theta: float  # how much safety cancels capability in the debt metric

    # Two-phase capability growth (recursive self-improvement)
    K_threshold: float  # capability level where recursive self-improvement kicks in
    beta_dim: float     # diminishing returns strength before threshold (higher = more diminishing)

    # Do agents *care* about debt in their payoff? (kept for future)
    lam: float = 0.5  # lambda=0 means they ignore debt when choosing actions


@dataclass
class State:
    # State vector:
    # K[i] = capability stock for bloc i
    # S[i] = safety stock for bloc i
    # T    = global trust/verification stock
    K: np.ndarray  # shape (3,)
    S: np.ndarray  # shape (3,)
    T: float       # scalar


@dataclass
class Controls:
    # Per-bloc effort allocations at a given time t
    # aX[i] = acceleration effort for bloc i (0..1)
    # aS[i] = safety effort for bloc i (0..1)
    # aV[i] = verification/coop effort for bloc i (0..1)
    aX: np.ndarray  # shape (3,)
    aS: np.ndarray  # shape (3,)
    aV: np.ndarray  # shape (3,)


###############################################################################
# 2. Control policy
###############################################################################
# For now, policy is just a function of time t (and maybe state later).
# Later you can implement best-response: aX_i(t) = argmax payoff_i(...)
#
# Signature: policy_fn(t, y_vector) -> Controls
#
# y_vector is the flattened state we'll pass to the ODE solver. We'll provide
# a helper to unpack that into (K,S,T) if you want state-feedback policies.


def unpack_state(y: np.ndarray) -> State:
    """
    y is a flat array of length 3(K) + 3(S) + 1(T) = 7
    ordering: [K_US, K_CN, K_EU, S_US, S_CN, S_EU, T]
    """
    K = y[0:3]
    S = y[3:6]
    T = y[6]
    return State(K=K, S=S, T=T)


def simple_scenario_policy_builder(
    mode: str = "arms_race"
) -> Callable[[float, np.ndarray], Controls]:
    """
    Returns a policy function.
    - 'arms_race': high accel, low safety, low verification
    - 'treaty': moderate accel, higher safety, decent verification
    You can extend this to depend on current state for adaptive behavior.
    """

    def policy_fn(t: float, y: np.ndarray) -> Controls:
        if mode == "arms_race":
            aX = np.array([0.9, 0.9, 0.9])
            aS = np.array([0.1, 0.1, 0.1])
            aV = np.array([0.05, 0.05, 0.05])

        elif mode == "treaty":
            aX = np.array([0.6, 0.6, 0.6])
            aS = np.array([0.5, 0.5, 0.5])
            aV = np.array([0.5, 0.5, 0.5])

        else:
            # default fallback (you can raise instead)
            aX = np.array([0.8, 0.8, 0.8])
            aS = np.array([0.2, 0.2, 0.2])
            aV = np.array([0.1, 0.1, 0.1])

        return Controls(aX=aX, aS=aS, aV=aV)

    return policy_fn


###############################################################################
# 3. System dynamics (ODE right-hand side)
###############################################################################
# Continuous-time approximation of:
#
# Two-phase capability growth (recursive self-improvement):
#   If K_i < K_threshold:
#     dK_i/dt = alpha * aX_i / (1 + beta_dim * K_i)  [diminishing returns]
#   If K_i >= K_threshold:
#     dK_i/dt = alpha * aX_i * K_i  [exponential/compounding growth]
#
# dS_i/dt = gamma * aS_i
#           + eta * T * avg_other_S_i
#
# dT/dt   = beta * mean(aV_i) - delta_T * T
#
# avg_other_S_i = (sum_j S_j - S_i)/(N_PLAYERS-1)
#
# No decay in K or S (can add later).
# No resource depletion in K.
# No feedback from "risk" into behavior (lam=0 baseline).
#
# We'll feed this to solve_ivp.


def rhs_ode(
    t: float,
    y: np.ndarray,
    params: Params,
    policy_fn: Callable[[float, np.ndarray], Controls],
) -> np.ndarray:
    """
    Compute dy/dt for the flattened state y.
    """
    st = unpack_state(y)
    ctrl = policy_fn(t, y)

    dK = np.zeros(N_PLAYERS)
    dS = np.zeros(N_PLAYERS)

    # --- Capability dynamics (two-phase growth)
    # Phase 1 (K < K_threshold): Diminishing returns
    #   dK_i/dt = alpha * aX_i / (1 + beta_dim * K_i)
    # Phase 2 (K >= K_threshold): Recursive self-improvement (compounding)
    #   dK_i/dt = alpha * aX_i * K_i
    for i in range(N_PLAYERS):
        if st.K[i] < params.K_threshold:
            # Diminishing returns phase
            dK[i] = params.alpha * ctrl.aX[i] / (1.0 + params.beta_dim * st.K[i])
        else:
            # Recursive self-improvement phase (exponential growth)
            dK[i] = params.alpha * ctrl.aX[i] * st.K[i]

    # --- Safety dynamics
    # dS_i/dt = gamma * aS_i + eta * T * avg_other_S_i
    S_sum = np.sum(st.S)
    for i in range(N_PLAYERS):
        avg_other_S = (S_sum - st.S[i]) / (N_PLAYERS - 1)
        spillover = params.eta * st.T * avg_other_S
        dS[i] = params.gamma * ctrl.aS[i] + spillover

    # --- Trust dynamics
    # dT/dt = beta * mean(aV_i) - delta_T * T
    mean_aV = np.mean(ctrl.aV)
    dT = params.beta * mean_aV - params.delta_T * st.T

    # pack derivative
    dydt = np.zeros_like(y)
    dydt[0:3] = dK
    dydt[3:6] = dS
    dydt[6] = dT

    return dydt


###############################################################################
# 4. Diagnostics
###############################################################################
# Safety debt / overhang D_i(t) = max(0, K_i - theta * S_i)
# This is NOT fed back into dynamics yet (lam=0).
# You can compute it after integration to see how dangerous each bloc is.


def safety_debt(state: State, params: Params) -> np.ndarray:
    debt = state.K - params.theta * state.S
    debt = np.maximum(debt, 0.0)
    return debt


###############################################################################
# 5. Simulation helper
###############################################################################


def simulate(
    t_span: tuple[float, float],
    y0: np.ndarray,
    params: Params,
    policy_fn: Callable[[float, np.ndarray], Controls],
    t_eval: np.ndarray = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
):
    """
    Wrapper around solve_ivp.
    - t_span: (t0, tf)
    - y0: initial state vector [K0_US, K0_CN, K0_EU, S0_US, S0_CN, S0_EU, T0]
    - params: Params(...)
    - policy_fn: decides aX, aS, aV
    - t_eval: times at which to sample solution (np.linspace recommended)
    """

    sol = solve_ivp(
        fun=lambda t, y: rhs_ode(t, y, params, policy_fn),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        vectorized=False,
        rtol=rtol,
        atol=atol,
    )
    return sol


def plot_results(t_eval, K_path):
    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, K_path[US], label="US K")
    plt.plot(t_eval, K_path[CN], label="CN K")
    plt.plot(t_eval, K_path[EU], label="EU K")
    plt.xlabel("Time")
    plt.ylabel("Capability (K)")
    plt.title("Capability Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/CST.png")
    plt.close()


def plot_actions(t_eval, aX_path, aS_path, aV_path):
    """
    Plot actions over time using step plots.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot aX (acceleration)
    axes[0].step(t_eval, aX_path[US], label="US", where='post', linewidth=2)
    axes[0].step(t_eval, aX_path[CN], label="CN", where='post', linewidth=2)
    axes[0].step(t_eval, aX_path[EU], label="EU", where='post', linewidth=2)
    axes[0].set_ylabel("Acceleration Effort (aX)")
    axes[0].set_title("Action Evolution Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-0.05, 1.05])

    # Plot aS (safety)
    axes[1].step(t_eval, aS_path[US], label="US", where='post', linewidth=2)
    axes[1].step(t_eval, aS_path[CN], label="CN", where='post', linewidth=2)
    axes[1].step(t_eval, aS_path[EU], label="EU", where='post', linewidth=2)
    axes[1].set_ylabel("Safety Effort (aS)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-0.05, 1.05])

    # Plot aV (verification)
    axes[2].step(t_eval, aV_path[US], label="US", where='post', linewidth=2)
    axes[2].step(t_eval, aV_path[CN], label="CN", where='post', linewidth=2)
    axes[2].step(t_eval, aV_path[EU], label="EU", where='post', linewidth=2)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Verification Effort (aV)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig("plots/CST_actions.png", dpi=150)
    plt.close()
    print("Saved action plot to plots/CST_actions.png")

###############################################################################
# 6. Nash / best response implementation
###############################################################################


def compute_next_state_single_step(
    state: State,
    controls: Controls,
    params: Params,
    dt: float = 0.1,
) -> State:
    """
    Compute the next state after a small timestep dt using Euler integration.
    """
    dK = np.zeros(N_PLAYERS)
    dS = np.zeros(N_PLAYERS)

    # Capability dynamics
    for i in range(N_PLAYERS):
        if state.K[i] < params.K_threshold:
            dK[i] = params.alpha * controls.aX[i] / (1.0 + params.beta_dim * state.K[i])
        else:
            dK[i] = params.alpha * controls.aX[i] * state.K[i]

    # Safety dynamics
    S_sum = np.sum(state.S)
    for i in range(N_PLAYERS):
        avg_other_S = (S_sum - state.S[i]) / (N_PLAYERS - 1)
        spillover = params.eta * state.T * avg_other_S
        dS[i] = params.gamma * controls.aS[i] + spillover

    # Trust dynamics
    mean_aV = np.mean(controls.aV)
    dT = params.beta * mean_aV - params.delta_T * state.T

    # Euler step
    K_next = state.K + dK * dt
    S_next = state.S + dS * dt
    T_next = state.T + dT * dt

    return State(K=K_next, S=S_next, T=T_next)


def compute_payoff(
    bloc_i: int,
    state: State,
    params: Params,
) -> float:
    """
    Compute payoff for bloc i: U_i = K_i - lam * Debt_i
    where Debt_i = max(0, K_i - theta * S_i)
    """
    debt_i = max(0.0, state.K[bloc_i] - params.theta * state.S[bloc_i])
    payoff = state.K[bloc_i] - params.lam * debt_i
    return payoff


def best_response_for_bloc(
    bloc_i: int,
    state: State,
    other_controls: Controls,
    params: Params,
    dt: float = 0.1,
    budget_constraint: bool = True,
    action_grid: list = None,
) -> tuple[float, float, float]:
    """
    Find best response actions for bloc i, given other blocs' actions.
    Returns (aX_i, aS_i, aV_i) that maximize bloc i's payoff.
    Uses discrete grid search for speed.
    """
    if action_grid is None:
        # Coarse grid for speed: only 5 values per action
        action_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    best_payoff = -np.inf
    best_actions = (0.5, 0.3, 0.2)

    # Grid search over all combinations
    for aX_i in action_grid:
        for aS_i in action_grid:
            for aV_i in action_grid:
                # Check budget constraint
                if budget_constraint and (not math.isclose(aX_i + aS_i + aV_i, 1.0)):
                    continue

                # Create full control vector
                aX_full = other_controls.aX.copy()
                aS_full = other_controls.aS.copy()
                aV_full = other_controls.aV.copy()

                aX_full[bloc_i] = aX_i
                aS_full[bloc_i] = aS_i
                aV_full[bloc_i] = aV_i

                controls = Controls(aX=aX_full, aS=aS_full, aV=aV_full)

                # Compute next state
                next_state = compute_next_state_single_step(state, controls, params, dt)

                # Compute payoff
                payoff = compute_payoff(bloc_i, next_state, params)

                if payoff > best_payoff:
                    best_payoff = payoff
                    best_actions = (aX_i, aS_i, aV_i)

    return best_actions


def best_response_policy_builder(
    params: Params,
    dt: float = 0.1,
    max_iterations: int = 5,
    budget_constraint: bool = True,
) -> Callable[[float, np.ndarray], Controls]:
    """
    Returns a best-response policy function that computes Nash equilibrium
    via iterated best responses.
    """
    # Cache for previous actions (warm start)
    cache = {
        'aX': np.array([0.5, 0.5, 0.5]),
        'aS': np.array([0.3, 0.3, 0.3]),
        'aV': np.array([0.2, 0.2, 0.2]),
    }

    def policy_fn(t: float, y: np.ndarray) -> Controls:
        state = unpack_state(y)

        # Initialize with cached actions
        aX = cache['aX'].copy()
        aS = cache['aS'].copy()
        aV = cache['aV'].copy()

        # Iterative best response
        for iteration in range(max_iterations):
            aX_new = aX.copy()
            aS_new = aS.copy()
            aV_new = aV.copy()

            # Each bloc computes best response given others' current actions
            for i in range(N_PLAYERS):
                current_controls = Controls(aX=aX, aS=aS, aV=aV)
                aX_i, aS_i, aV_i = best_response_for_bloc(
                    i, state, current_controls, params, dt, budget_constraint
                )
                aX_new[i] = aX_i
                aS_new[i] = aS_i
                aV_new[i] = aV_i

            # Check convergence
            change = (np.max(np.abs(aX_new - aX)) +
                     np.max(np.abs(aS_new - aS)) +
                     np.max(np.abs(aV_new - aV)))

            aX, aS, aV = aX_new, aS_new, aV_new

            if change < 1e-4:
                break

        # Update cache
        cache['aX'] = aX.copy()
        cache['aS'] = aS.copy()
        cache['aV'] = aV.copy()

        return Controls(aX=aX, aS=aS, aV=aV)

    return policy_fn


###############################################################################
# 7. Example usage / quick demo
###############################################################################
if __name__ == "__main__":
    # --- Define parameters
    params = Params(
        alpha=1.0,       # capability growth rate per accel effort
        gamma=0.5,       # safety growth rate per safety effort
        eta=0.2,         # spillover strength (trust -> shared safety)
        beta=0.3,        # trust build rate from verification effort
        delta_T=0.1,     # trust decay
        theta=0.8,       # how effective safety is at offsetting capability
        K_threshold=10.0,  # AGI threshold for recursive self-improvement
        beta_dim=0.3,    # diminishing returns strength (higher = more diminishing)
        lam=0.5,         # payoff weighting for safety debt (0=ignore, 1=full weight)
    )

    # --- Initial conditions:
    # Let's say everyone starts modest on capability, low-ish safety,
    # and almost no trust.
    K0 = np.array([12.0, 9.0, 7.0])   # starting capability levels
    S0 = np.array([0.2, 0.12, 0.15])   # starting safety levels
    T0 = 0.1                         # low verification regime
    y0 = np.concatenate([K0, S0, [T0]])

    # --- Time horizon
    t0, tf = 0.0, 20.0  # Shorter time for faster best response computation
    t_eval = np.linspace(t0, tf, 101)  # Fewer points for faster computation

    # --- Choose policy
    use_best_response = True  # Set to False to use simple scenario policy

    if use_best_response:
        print("Using best response policy (Nash equilibrium)...")
        print("Using discrete action grid with 5 values per action")
        policy_fn = best_response_policy_builder(
            params=params,
            dt=0.1,
            max_iterations=3,  # Reduced for speed
            budget_constraint=True,  # Enforce aX + aS + aV <= 1
        )
    else:
        print("Using fixed scenario policy...")
        mode = "arms_race"
        policy_fn = simple_scenario_policy_builder(mode=mode)

    # --- Run simulation and track actions
    print("Running simulation...")
    sol = simulate(
        t_span=(t0, tf),
        y0=y0,
        params=params,
        policy_fn=policy_fn,
        t_eval=t_eval,
    )

    # --- Unpack results for convenience
    K_path = sol.y[0:3, :]   # shape (3, len(t_eval))
    S_path = sol.y[3:6, :]
    T_path = sol.y[6, :]

    # --- Reconstruct actions at each timestep
    print("Reconstructing actions...")
    aX_path = np.zeros((N_PLAYERS, len(t_eval)))
    aS_path = np.zeros((N_PLAYERS, len(t_eval)))
    aV_path = np.zeros((N_PLAYERS, len(t_eval)))

    for ti, t in enumerate(t_eval):
        if ti % 20 == 0:
            print(f"  Progress: {ti}/{len(t_eval)}")
        ctrl = policy_fn(t, sol.y[:, ti])
        aX_path[:, ti] = ctrl.aX
        aS_path[:, ti] = ctrl.aS
        aV_path[:, ti] = ctrl.aV
    print(f"  Progress: {len(t_eval)}/{len(t_eval)} - Done!")

    # --- Compute safety debt over time for each bloc
    debt_path = np.zeros_like(K_path)
    for ti in range(len(t_eval)):
        st_t = State(
            K=K_path[:, ti],
            S=S_path[:, ti],
            T=T_path[ti],
        )
        debt_path[:, ti] = safety_debt(st_t, params)

    # --- Print some summary stats
    print("\n=== Final State ===")
    print("Final capabilities K:", K_path[:, -1])
    print("Final safety S:", S_path[:, -1])
    print("Final trust T:", T_path[-1])
    print("Final safety debt:", debt_path[:, -1])
    print("\n=== Final Actions ===")
    print("Final aX (acceleration):", aX_path[:, -1])
    print("Final aS (safety):", aS_path[:, -1])
    print("Final aV (verification):", aV_path[:, -1])
    print("Final effort sum:", aX_path[:, -1] + aS_path[:, -1] + aV_path[:, -1])

    # --- Plot results
    print("\nGenerating plots...")
    plot_results(t_eval, K_path)
    plot_actions(t_eval, aX_path, aS_path, aV_path)
    print("Done!")
