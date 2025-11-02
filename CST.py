from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from CST_lib import plot_results, plot_actions, load_calibration_from_json

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

    # Safety growth efficiency (bloc-dependent: [US, CN, EU])
    gamma: np.ndarray  # dS/dt contribution from safety effort

    # Spillover strength for safety via trust
    eta: float  # multiplier on T * avg_other_S

    # Trust formation / decay
    beta: float  # how fast joint verification effort builds trust
    delta_T: float  # trust decay rate

    # Safety effectiveness for diagnostic debt
    theta: float  # how much safety cancels capability in the debt metric

    # Two-phase capability growth (recursive self-improvement)
    K_threshold: float  # capability level where recursive self-improvement kicks in
    beta_dim: float  # diminishing returns strength before threshold (higher = more diminishing)
    transition_width: float  # width of smooth transition zone (superhuman coder → superhuman AI researcher)

    # Do agents *care* about debt in their payoff? (kept for future)
    lam: float = 0.5  # lambda=0 means they ignore debt when choosing actions

    # Direct value of trust/cooperation in payoff
    omega: float = 1.1  # weight on trust benefit in utility function


@dataclass
class State:
    # State vector:
    # K[i] = capability stock for bloc i
    # S[i] = safety stock for bloc i
    # T    = global trust/verification stock
    K: np.ndarray  # shape (3,)
    S: np.ndarray  # shape (3,)
    T: float  # scalar


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
    mode: str = "arms_race",
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


def compute_capability_derivatives(
    K: np.ndarray, aX: np.ndarray, params: Params, force_phase1: bool = False
) -> np.ndarray:
    """
    Compute dK/dt for all players using smooth two-phase growth.

    Phase 1 (K << K_threshold): Diminishing returns (pre-superhuman coder)
      dK_i/dt = alpha * aX_i / (1 + beta_dim * K_i)
    Phase 2 (K >> K_threshold): Recursive self-improvement (superhuman AI researcher)
      dK_i/dt = alpha * aX_i * K_i
    Transition: Smooth blend using tanh centered at K_threshold
      Physically: transition from "superhuman coder" to "superhuman AI researcher"

    Args:
        K: Capability levels for all players (shape N_PLAYERS)
        aX: Acceleration efforts for all players (shape N_PLAYERS)
        params: Model parameters
        force_phase1: If True, only use phase 1 (diminishing returns) dynamics,
                     ignoring the transition to exponential growth

    Returns:
        dK: Capability derivatives for all players (shape N_PLAYERS)
    """
    dK = np.zeros(N_PLAYERS)
    for i in range(N_PLAYERS):
        # Phase 1: diminishing returns
        phase1 = params.alpha * aX[i] / (1.0 + params.beta_dim * K[i])

        if force_phase1:
            # Only use phase 1 dynamics (simulate uncertainty about threshold)
            dK[i] = phase1
        else:
            # Phase 2: exponential (recursive self-improvement)
            phase2 = params.alpha * aX[i] * K[i]
            # Smooth weight: 0 at K << K_threshold, 1 at K >> K_threshold
            weight = 0.5 * (
                1.0 + np.tanh((K[i] - params.K_threshold) / params.transition_width)
            )
            # Blend the two phases
            dK[i] = (1.0 - weight) * phase1 + weight * phase2
    return dK


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

    # --- Capability dynamics
    dK = compute_capability_derivatives(st.K, ctrl.aX, params)

    dS = np.zeros(N_PLAYERS)

    # --- Safety dynamics
    # dS_i/dt = gamma[i] * aS_i + eta * T * avg_other_S_i
    S_sum = np.sum(st.S)
    for i in range(N_PLAYERS):
        avg_other_S = (S_sum - st.S[i]) / (N_PLAYERS - 1)
        spillover = params.eta * st.T * avg_other_S
        dS[i] = params.gamma[i] * ctrl.aS[i] + spillover

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


###############################################################################
# 6. Nash / best response implementation
###############################################################################


def compute_next_state_single_step(
    state: State,
    controls: Controls,
    params: Params,
    dt: float = 0.1,
    force_phase1: bool = False,
) -> State:
    """
    Compute the next state after a small timestep dt using Euler integration.

    Args:
        force_phase1: If True, only use phase 1 dynamics for capability growth
    """
    # Capability dynamics
    dK = compute_capability_derivatives(state.K, controls.aX, params, force_phase1=force_phase1)

    dS = np.zeros(N_PLAYERS)

    # Safety dynamics
    S_sum = np.sum(state.S)
    for i in range(N_PLAYERS):
        avg_other_S = (S_sum - state.S[i]) / (N_PLAYERS - 1)
        spillover = params.eta * state.T * avg_other_S
        dS[i] = params.gamma[i] * controls.aS[i] + spillover

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
    Compute payoff for bloc i: U_i = K_i - lam * Debt_i + omega * T
    where Debt_i = max(0, K_i - theta * S_i)
    and omega * T represents the direct value of trust/cooperation
    """
    debt_i = max(0.0, state.K[bloc_i] - params.theta * state.S[bloc_i])
    trust_benefit = params.omega * state.T
    payoff = state.K[bloc_i] - params.lam * debt_i + trust_benefit
    return payoff


def compute_lookahead_payoff(
    bloc_i: int,
    state: State,
    controls: Controls,
    params: Params,
    lookahead_years: float = 2.0,
    discount_rate: float = 0.2,
    n_steps: int = 5,
    phase1_only_lookahead: bool = False,
) -> float:
    """
    Compute discounted cumulative payoff by simulating forward lookahead_years.
    More strategic than single-step payoff.

    Args:
        phase1_only_lookahead: If True and bloc_i's current K < K_threshold,
                              use only phase 1 dynamics in lookahead simulation.
                              This simulates agents not knowing when exponential
                              growth will begin.

    Returns: integral of exp(-discount_rate * t) * payoff(t) dt from 0 to lookahead_years
    """
    dt = lookahead_years / n_steps
    current_state = State(K=state.K.copy(), S=state.S.copy(), T=state.T)

    # Determine if we should force phase 1 dynamics in lookahead
    force_phase1 = phase1_only_lookahead and (state.K[bloc_i] < params.K_threshold)

    total_payoff = 0.0
    for step in range(n_steps):
        t = step * dt
        # Compute instantaneous payoff
        instant_payoff = compute_payoff(bloc_i, current_state, params)
        # Add discounted payoff
        total_payoff += instant_payoff * np.exp(-discount_rate * t) * dt
        # Step forward
        current_state = compute_next_state_single_step(
            current_state, controls, params, dt, force_phase1=force_phase1
        )

    return total_payoff


def best_response_for_bloc(
    bloc_i: int,
    state: State,
    other_controls: Controls,
    params: Params,
    dt: float = 0.1,
    budget_constraint: bool = True,
    action_grid: list = None,
    use_lookahead: bool = True,
    lookahead_years: float = 2.0,
    discount_rate: float = 0.2,
    phase1_only_lookahead: bool = False,
) -> tuple[float, float, float]:
    """
    Find best response actions for bloc i, given other blocs' actions.
    Returns (aX_i, aS_i, aV_i) that maximize bloc i's payoff.
    Uses discrete grid search for speed.

    Args:
        use_lookahead: If True, use forward simulation with discounting for strategic behavior
        lookahead_years: How far to simulate forward (typically 2 years)
        discount_rate: Exponential discount rate for future payoffs
        phase1_only_lookahead: If True, agents below K_threshold will use only phase 1
                              dynamics in their lookahead, simulating uncertainty about
                              when exponential growth begins
    """
    if action_grid is None:
        # Coarse grid for speed
        action_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        # action_grid = [i * 0.2 for i in range(6)]

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

                # Compute payoff
                if use_lookahead:
                    payoff = compute_lookahead_payoff(
                        bloc_i, state, controls, params,
                        lookahead_years=lookahead_years,
                        discount_rate=discount_rate,
                        phase1_only_lookahead=phase1_only_lookahead,
                    )
                else:
                    # Original single-step payoff
                    next_state = compute_next_state_single_step(state, controls, params, dt)
                    payoff = compute_payoff(bloc_i, next_state, params)

                if payoff > best_payoff:
                    best_payoff = payoff
                    best_actions = (aX_i, aS_i, aV_i)

    return best_actions


def bostrom_minimal_info_policy_builder(
    params: Params,
    budget_constraint: bool = True,
    action_grid: list = None,
) -> Callable[[float, np.ndarray], Controls]:
    """
    Returns a policy function implementing Bostrom's "minimal information" scenario
    from "Racing to the Precipice" (Bostrom, 2014).

    Key features:
    - Agents only observe their own current state (K_i, S_i, T)
    - No knowledge of other agents' payoffs or strategies
    - No lookahead - only instantaneous/myopic optimization
    - Agents maximize immediate payoff gradient without modeling future consequences

    This represents the most informationally limited rational agents - they know
    their own utility function but can't predict others' actions or their own
    future states.
    """
    if action_grid is None:
        action_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    def policy_fn(t: float, y: np.ndarray) -> Controls:
        state = unpack_state(y)

        # Each bloc independently optimizes based only on current state
        aX = np.zeros(N_PLAYERS)
        aS = np.zeros(N_PLAYERS)
        aV = np.zeros(N_PLAYERS)

        for bloc_i in range(N_PLAYERS):
            best_payoff = -np.inf
            best_actions = (0.5, 0.3, 0.2)

            # Grid search over own actions only (no modeling of others)
            for aX_i in action_grid:
                for aS_i in action_grid:
                    for aV_i in action_grid:
                        # Check budget constraint
                        if budget_constraint and (not math.isclose(aX_i + aS_i + aV_i, 1.0)):
                            continue

                        # Compute instantaneous payoff at current state
                        # (not future state - myopic optimization)
                        # We evaluate how good this action would be RIGHT NOW

                        # Current payoff (baseline)
                        current_payoff = compute_payoff(bloc_i, state, params)

                        # Estimate instantaneous payoff gradient:
                        # How much does my payoff increase if I take this action
                        # for one instant, given current state?

                        # For myopic agents, we care about immediate changes:
                        # - aX_i increases K_i immediately
                        # - aS_i decreases debt_i immediately
                        # - aV_i increases T immediately (through mean_aV)

                        # Marginal utility from each action:
                        # dU/dt = dK_i/dt * (1 - lam * d(debt)/dK) + omega * dT/dt

                        # Capability gain rate
                        aX_temp = np.zeros(N_PLAYERS)
                        aX_temp[bloc_i] = aX_i
                        dK_i = compute_capability_derivatives(state.K, aX_temp, params)[bloc_i]

                        # Safety gain rate (ignoring spillover for minimal info)
                        dS_i = params.gamma[bloc_i] * aS_i

                        # Trust gain rate (assuming others' aV stays at current mean)
                        # Under minimal info, agent doesn't know others' future actions,
                        # so assumes they continue current behavior
                        dT = params.beta * (aV_i / N_PLAYERS) - params.delta_T * state.T

                        # Instantaneous payoff rate:
                        # dU_i/dt = dK_i/dt - lam * d(debt_i)/dt + omega * dT/dt
                        # where d(debt_i)/dt = dK_i/dt - theta * dS_i/dt

                        ddebti_dt = dK_i - params.theta * dS_i
                        # Only count debt derivative if currently in debt
                        if state.K[bloc_i] > params.theta * state.S[bloc_i]:
                            ddebti_dt = max(0, ddebti_dt)
                        else:
                            ddebti_dt = 0

                        payoff_rate = dK_i - params.lam * ddebti_dt + params.omega * dT

                        if float(payoff_rate) > best_payoff:
                            best_payoff = float(payoff_rate)
                            best_actions = (aX_i, aS_i, aV_i)

            aX[bloc_i], aS[bloc_i], aV[bloc_i] = best_actions

        return Controls(aX=aX, aS=aS, aV=aV)

    return policy_fn


def best_response_policy_builder(
    params: Params,
    dt: float = 0.1,
    max_iterations: int = 5,
    budget_constraint: bool = True,
    seed: int = 42,
    action_grid: list = None,
    use_lookahead: bool = True,
    lookahead_years: float = 2.0,
    discount_rate: float = 0.2,
    phase1_only_lookahead: bool = False,
) -> Callable[[float, np.ndarray], Controls]:
    """
    Returns a best-response policy function that computes Nash equilibrium
    via iterated best responses.

    Args:
        seed: Random seed for deterministic player ordering in best response iteration
        action_grid: Discrete action values to search over (finer = smoother but slower)
        use_lookahead: Use forward-looking discounted payoff (more strategic)
        lookahead_years: Horizon for forward simulation
        discount_rate: Discount rate for future payoffs
        phase1_only_lookahead: If True, agents below K_threshold use only phase 1
                              dynamics in lookahead (simulates threshold uncertainty)
    """
    # Cache for previous actions (warm start)
    cache = {
        "aX": np.array([0.5, 0.5, 0.5]),
        "aS": np.array([0.3, 0.3, 0.3]),
        "aV": np.array([0.2, 0.2, 0.2]),
    }

    # Create random number generator with fixed seed for reproducibility
    rng = np.random.default_rng(seed)

    def policy_fn(t: float, y: np.ndarray) -> Controls:
        state = unpack_state(y)

        # Initialize with cached actions
        aX = cache["aX"].copy()
        aS = cache["aS"].copy()
        aV = cache["aV"].copy()

        # Iterative best response
        for iteration in range(max_iterations):
            aX_new = aX.copy()
            aS_new = aS.copy()
            aV_new = aV.copy()

            # Randomize player order to avoid bias and improve Nash convergence
            player_order = rng.permutation(N_PLAYERS)

            # Each bloc computes best response given others' current actions
            for i in player_order:
                current_controls = Controls(aX=aX, aS=aS, aV=aV)
                aX_i, aS_i, aV_i = best_response_for_bloc(
                    i, state, current_controls, params, dt, budget_constraint,
                    action_grid=action_grid,
                    use_lookahead=use_lookahead,
                    lookahead_years=lookahead_years,
                    discount_rate=discount_rate,
                    phase1_only_lookahead=phase1_only_lookahead,
                )
                aX_new[i] = aX_i
                aS_new[i] = aS_i
                aV_new[i] = aV_i

            # Check convergence
            change = (
                np.max(np.abs(aX_new - aX))
                + np.max(np.abs(aS_new - aS))
                + np.max(np.abs(aV_new - aV))
            )

            aX, aS, aV = aX_new, aS_new, aV_new

            if change < 1e-4:
                break

        # Update cache
        cache["aX"] = aX.copy()
        cache["aS"] = aS.copy()
        cache["aV"] = aV.copy()

        return Controls(aX=aX, aS=aS, aV=aV)

    return policy_fn


###############################################################################
# 7. Example usage / quick demo
###############################################################################
if __name__ == "__main__":
    # --- Option to load from JSON calibration file or use hardcoded values
    use_calibration_file = True  # Set to False to use hardcoded params below
    calibration_file = "calibration_from_real_data.json"

    if use_calibration_file:
        print(f"Loading parameters and initial conditions from {calibration_file}...")
        params, y0 = load_calibration_from_json(calibration_file, Params)
        params.K_threshold = 14.5
        # params.delta_T = 0  # no trust decay
        # > 1 so that we have the effect of AI danger exceeds the benefit when
        # exponential growth starts to kick in
        params.lam = 1.1
        # safety growth rate per safety effort [US, CN, EU]
        params.gamma = np.array([0.15, 0.025, 0.10])  
    else:
        print("Using hardcoded parameters and initial conditions...")
        # --- Define parameters
        params = Params(
            alpha=1.0,  # capability growth rate per accel effort
            gamma=np.array([0.15, 0.025, 0.10]),  # safety growth rate per safety effort [US, CN, EU]
            eta=0.2,  # spillover strength (trust -> shared safety)
            beta=0.3,  # trust build rate from verification effort
            delta_T=0.1,  # trust decay
            theta=0.8,  # how effective safety is at offsetting capability
            K_threshold=10.0,  # AGI threshold for recursive self-improvement
            beta_dim=0.3,  # diminishing returns strength (higher = more diminishing)
            transition_width=1.5,  # smooth transition zone (superhuman coder → superhuman AI researcher)
            lam=0.5,  # payoff weighting for safety debt (0=ignore, 1=full weight)
        )

        # --- Initial conditions:
        # Let's say everyone starts modest on capability, low-ish safety,
        # and almost no trust.
        # starting capability levels
        K0_US = 5e26  # Grok 4
        K0 = (
            np.array(
                [
                    K0_US,
                    1.5e25,  # Qwen 3 Max
                    1.8e24,  # Mistral Large 2
                ]
            )
            / K0_US
        )
        S0 = np.array([0.2, 0.12, 0.15])  # starting safety levels
        T0 = 0.1  # low verification regime
        y0 = np.concatenate([K0, S0, [T0]])

    # --- Time horizon
    t0, tf = 0.0, 15.0  # Shorter time for faster best response computation
    t_eval = np.linspace(t0, tf, 101)  # Fewer points for faster computation

    # --- Choose policy
    # Options: "best_response", "bostrom_minimal", "scenario"
    policy_mode = "best_response"
    policy_mode = "bostrom_minimal"

    if policy_mode == "best_response":
        print("Using best response policy (approximate Nash equilibrium)...")
        # Finer action grid for smoother trajectories (10 values: 0.0, 0.1, 0.2, ..., 0.9, 1.0)
        action_grid = [i * 0.1 for i in range(11)]
        action_grid = None
        print("Using lookahead with exponential discounting (with discount rate)")

        # Set to True to make agents ignore exponential growth in their lookahead
        # This simulates uncertainty about when the threshold will be reached
        phase1_only_lookahead = False
        if phase1_only_lookahead:
            print("Phase 1 only lookahead: Agents below threshold ignore exponential growth in projections")

        policy_fn = best_response_policy_builder(
            params=params,
            dt=0.25,
            max_iterations=4,  # Reduced for speed
            budget_constraint=True,  # Enforce aX + aS + aV = 1
            action_grid=action_grid,
            use_lookahead=True,
            lookahead_years=1,
            discount_rate=0.3,
            phase1_only_lookahead=phase1_only_lookahead,
        )
    elif policy_mode == "bostrom_minimal":
        print("Using Bostrom minimal information policy (myopic optimization)...")
        print("Agents have no lookahead and no knowledge of others' payoffs")
        action_grid = [i * 0.1 for i in range(11)]  # Finer grid for smoother actions
        policy_fn = bostrom_minimal_info_policy_builder(
            params=params,
            budget_constraint=True,  # Enforce aX + aS + aV = 1
            action_grid=action_grid,
        )
    else:  # "scenario"
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
    K_path = sol.y[0:3, :]  # shape (3, len(t_eval))
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
    plot_results(t_eval, K_path, params, suffix=policy_mode)
    plot_actions(t_eval, aX_path, aS_path, aV_path, suffix=policy_mode)
    print("Done!")
