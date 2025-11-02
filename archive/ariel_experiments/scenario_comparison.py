import numpy as np
import matplotlib.pyplot as plt
from CST_ag import Params, simulate, simple_scenario_policy_builder, unpack_state, State

# Simulation setup
t0, tf = 0.0, 50.0
t_eval = np.linspace(t0, tf, 501)

# Win threshold
WIN_THRESHOLD = 50.0
player_names = {0: 'US', 1: 'CN', 2: 'EU'}

def analyze_trajectory(sol, params, scenario_name):
    """
    Analyze full trajectory: who wins, when, and what's the risk profile over time.
    """
    K_path = sol.y[0:3, :]
    S_path = sol.y[3:6, :]
    T_path = sol.y[6, :]
    t = sol.t
    
    # Find winner
    winner = None
    win_time = None
    win_idx = None
    
    for i, ti in enumerate(t):
        for player_idx in range(3):
            if K_path[player_idx, i] >= WIN_THRESHOLD:
                winner = player_idx
                win_time = ti
                win_idx = i
                break
        if winner is not None:
            break
    
    if winner is None:
        final_K = K_path[:, -1]
        winner = np.argmax(final_K)
        win_time = t[-1]
        win_idx = -1
    
    win_K = K_path[winner, win_idx]
    win_S = S_path[winner, win_idx]
    win_safety_ratio = win_S / win_K if win_K > 0 else 0
    win_debt = max(0, win_K - params.theta * win_S)
    
    # Calculate total system debt over time
    total_debt = np.zeros_like(t)
    for i in range(len(t)):
        st = State(K=K_path[:, i], S=S_path[:, i], T=T_path[i])
        debt = st.K - params.theta * st.S
        debt = np.maximum(debt, 0.0)
        total_debt[i] = np.sum(debt)
    
    return {
        'name': scenario_name,
        'winner': winner,
        'win_time': win_time,
        'win_K': win_K,
        'win_S': win_S,
        'win_safety_ratio': win_safety_ratio,
        'win_debt': win_debt,
        'K_path': K_path,
        'S_path': S_path,
        'T_path': T_path,
        'total_debt': total_debt,
        't': t
    }

# Define scenarios to compare
scenarios = {}

# SCENARIO 1: Baseline Arms Race
print("Running Scenario 1: Baseline Arms Race")
scenarios['baseline'] = {
    'name': 'Baseline Arms Race',
    'K0': np.array([5.0, 4.0, 3.0]),  # US ahead, all pre-AGI
    'S0': np.array([0.5, 0.4, 0.3]),  # ~10% safety coverage
    'T0': 0.1,
    'params': Params(
        alpha=1.0,
        gamma=0.5,
        eta=0.2,
        beta=0.3,
        delta_T=0.1,
        theta=0.8,
        K_threshold=8.0,
        beta_dim=0.3,
        lam=0.0
    ),
    'mode': 'arms_race'
}

# SCENARIO 2: China Compute Constrained (starts further behind)
print("Running Scenario 2: China Compute Constrained")
scenarios['china_constrained'] = {
    'name': 'China Compute-Constrained',
    'K0': np.array([5.0, 2.5, 3.0]),  # CN much further behind
    'S0': np.array([0.5, 0.25, 0.3]),  # Proportional safety
    'T0': 0.1,
    'params': Params(
        alpha=1.0,
        gamma=0.5,
        eta=0.2,
        beta=0.3,
        delta_T=0.1,
        theta=0.8,
        K_threshold=8.0,
        beta_dim=0.3,
        lam=0.0
    ),
    'mode': 'arms_race'
}

# SCENARIO 3: Harder Alignment Problem (higher AGI threshold)
print("Running Scenario 3: Harder Alignment (Higher Threshold)")
scenarios['hard_alignment'] = {
    'name': 'Harder Alignment Problem',
    'K0': np.array([5.0, 4.0, 3.0]),
    'S0': np.array([0.5, 0.4, 0.3]),
    'T0': 0.1,
    'params': Params(
        alpha=1.0,
        gamma=0.5,
        eta=0.2,
        beta=0.3,
        delta_T=0.1,
        theta=0.8,
        K_threshold=12.0,  # Higher threshold - exponential growth delayed
        beta_dim=0.3,
        lam=0.0
    ),
    'mode': 'arms_race'
}

# SCENARIO 4: Treaty/Cooperation
print("Running Scenario 4: International Treaty")
scenarios['treaty'] = {
    'name': 'International Treaty',
    'K0': np.array([5.0, 4.0, 3.0]),
    'S0': np.array([0.5, 0.4, 0.3]),
    'T0': 0.1,
    'params': Params(
        alpha=1.0,
        gamma=0.5,
        eta=0.2,
        beta=0.3,
        delta_T=0.1,
        theta=0.8,
        K_threshold=8.0,
        beta_dim=0.3,
        lam=0.0
    ),
    'mode': 'treaty'  # Higher safety investment
}

# SCENARIO 5: Slowed AI Progress (lower alpha)
print("Running Scenario 5: Slowed AI Progress")
scenarios['slow_progress'] = {
    'name': 'Slowed AI Progress',
    'K0': np.array([5.0, 4.0, 3.0]),
    'S0': np.array([0.5, 0.4, 0.3]),
    'T0': 0.1,
    'params': Params(
        alpha=0.6,  # Slower capability growth
        gamma=0.5,
        eta=0.2,
        beta=0.3,
        delta_T=0.1,
        theta=0.8,
        K_threshold=8.0,
        beta_dim=0.3,
        lam=0.0
    ),
    'mode': 'arms_race'
}

# SCENARIO 6: High Safety Spillover (better knowledge sharing)
print("Running Scenario 6: High Safety Spillover")
scenarios['high_spillover'] = {
    'name': 'High Safety Spillover',
    'K0': np.array([5.0, 4.0, 3.0]),
    'S0': np.array([0.5, 0.4, 0.3]),
    'T0': 0.3,  # Start with more trust
    'params': Params(
        alpha=1.0,
        gamma=0.5,
        eta=0.6,  # Much stronger spillover
        beta=0.5,  # Easier to build trust
        delta_T=0.05,  # Slower trust decay
        theta=0.8,
        K_threshold=8.0,
        beta_dim=0.3,
        lam=0.0
    ),
    'mode': 'arms_race'
}

# Run all scenarios
results = {}
for key, scenario in scenarios.items():
    y0 = np.concatenate([scenario['K0'], scenario['S0'], [scenario['T0']]])
    policy_fn = simple_scenario_policy_builder(mode=scenario['mode'])
    
    sol = simulate(
        t_span=(t0, tf),
        y0=y0,
        params=scenario['params'],
        policy_fn=policy_fn,
        t_eval=t_eval,
    )
    
    results[key] = analyze_trajectory(sol, scenario['params'], scenario['name'])

# Create comparison visualizations
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35, top=0.90, bottom=0.12, left=0.07, right=0.97)

colors = {'US': 'blue', 'CN': 'red', 'EU': 'green'}
scenario_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

# Plot 1: Who wins in each scenario (bar chart)
ax1 = fig.add_subplot(gs[0, :])
scenario_names = [results[k]['name'] for k in scenarios.keys()]
winners = [results[k]['winner'] for k in scenarios.keys()]
bar_colors = [scenario_colors[i] for i in range(len(scenarios))]

x_pos = np.arange(len(scenario_names))
ax1.bar(x_pos, [w + 1 for w in winners], color=bar_colors, alpha=0.75, edgecolor='black', linewidth=2)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(scenario_names, rotation=15, ha='right', fontsize=11)
ax1.set_yticks([1, 2, 3])
ax1.set_yticklabels(['US', 'CN', 'EU'], fontsize=12)
ax1.set_ylabel('Winner', fontsize=13, fontweight='bold')
ax1.set_title('Race Winner by Scenario', fontsize=15, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3, linewidth=0.8)
ax1.set_ylim(0.5, 3.5)

# Plot 2: Safety ratio at win
ax2 = fig.add_subplot(gs[1, 0])
safety_ratios = [100 * results[k]['win_safety_ratio'] for k in scenarios.keys()]
ax2.barh(x_pos, safety_ratios, color=bar_colors, alpha=0.75, edgecolor='black', linewidth=2)
ax2.set_yticks(x_pos)
ax2.set_yticklabels(scenario_names, fontsize=10)
ax2.set_xlabel('Safety/Capability Ratio (%) at Win', fontsize=11, fontweight='bold')
ax2.set_title('Safety State When Won', fontsize=12, fontweight='bold', pad=12)
ax2.axvline(x=5, color='darkred', linestyle='--', alpha=0.7, linewidth=2.5, label='5% (High Risk)')
ax2.axvline(x=10, color='darkorange', linestyle='--', alpha=0.7, linewidth=2.5, label='10% (Moderate)')
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(axis='x', alpha=0.3, linewidth=0.8)
ax2.tick_params(axis='both', labelsize=10)

# Plot 3: Time to win
ax3 = fig.add_subplot(gs[1, 1])
win_times = [results[k]['win_time'] for k in scenarios.keys()]
ax3.barh(x_pos, win_times, color=bar_colors, alpha=0.75, edgecolor='black', linewidth=2)
ax3.set_yticks(x_pos)
ax3.set_yticklabels(scenario_names, fontsize=10)
ax3.set_xlabel('Years to Win', fontsize=11, fontweight='bold')
ax3.set_title('Time Until Race Won', fontsize=12, fontweight='bold', pad=12)
ax3.grid(axis='x', alpha=0.3, linewidth=0.8)
ax3.tick_params(axis='both', labelsize=10)

# Plot 4: Safety debt at win
ax4 = fig.add_subplot(gs[1, 2])
win_debts = [results[k]['win_debt'] for k in scenarios.keys()]
ax4.barh(x_pos, win_debts, color=bar_colors, alpha=0.75, edgecolor='black', linewidth=2)
ax4.set_yticks(x_pos)
ax4.set_yticklabels(scenario_names, fontsize=10)
ax4.set_xlabel('Safety Debt at Win', fontsize=11, fontweight='bold')
ax4.set_title('Risk Level When Won', fontsize=12, fontweight='bold', pad=12)
ax4.grid(axis='x', alpha=0.3, linewidth=0.8)
ax4.tick_params(axis='both', labelsize=10)

# Plot 5: Total debt over time for each scenario
ax5 = fig.add_subplot(gs[2, :])
for idx, key in enumerate(scenarios.keys()):
    ax5.plot(results[key]['t'], results[key]['total_debt'], 
             color=scenario_colors[idx], linewidth=2.5, label=results[key]['name'], alpha=0.9)
ax5.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Total System Safety Debt', fontsize=12, fontweight='bold')
ax5.set_title('Total Risk Evolution Over Time', fontsize=13, fontweight='bold', pad=15)
ax5.legend(fontsize=10, loc='upper left', framealpha=0.95, edgecolor='black')
ax5.grid(True, alpha=0.3, linewidth=0.8)
ax5.set_xlim(0, 50)
ax5.tick_params(axis='both', labelsize=11)

# Add key assumptions text below the plots
assumptions_lines = [
    "Key Assumptions:",
    "• Win Threshold: K ≥ 50.0 capability units",
    "• Baseline Parameters: α=1.0 (capability growth), γ=0.5 (safety efficiency), η=0.2 (safety spillover), β=0.3 (trust building), δ_T=0.1 (trust decay), θ=0.8 (safety ratio), K_thresh=8.0 (AGI threshold)",
    "• Policy Modes: Arms Race (prioritize capabilities) vs. Treaty (higher safety investment)"
]
assumptions_text = '\n'.join(assumptions_lines)
fig.text(0.5, 0.04, assumptions_text, ha='center', va='center', fontsize=10, 
         wrap=True, color='#222222', linespacing=1.6,
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#f5f5dc', alpha=0.85, edgecolor='#888888', linewidth=1.5))

plt.savefig('plots/scenario_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved plot to plots/scenario_comparison.png")

# Print summary table
print("\n" + "="*100)
print("SCENARIO COMPARISON SUMMARY")
print("="*100)
print(f"{'Scenario':<30} {'Winner':<8} {'Time':<8} {'Safety%':<10} {'Debt':<12} {'Assessment':<20}")
print("-"*100)

for idx, key in enumerate(scenarios.keys()):
    r = results[key]
    safety_pct = 100 * r['win_safety_ratio']
    
    # Assess outcome
    if safety_pct > 10:
        assessment = "Relatively Safe"
    elif safety_pct > 5:
        assessment = "Moderate Risk"
    else:
        assessment = "High Risk"
    
    print(f"{r['name']:<30} {player_names[r['winner']]:<8} {r['win_time']:<8.1f} "
          f"{safety_pct:<10.2f} {r['win_debt']:<12.1f} {assessment:<20}")

print("="*100)

# Key insights
print("\nKEY INSIGHTS:")
print("-" * 100)

best_safety = max(scenarios.keys(), key=lambda k: results[k]['win_safety_ratio'])
print(f"SAFEST OUTCOME: {results[best_safety]['name']}")
print(f"  - Safety/Capability: {100*results[best_safety]['win_safety_ratio']:.2f}%")
print(f"  - Winner: {player_names[results[best_safety]['winner']]}")

slowest = max(scenarios.keys(), key=lambda k: results[k]['win_time'])
print(f"\nMOST TIME TO WIN: {results[slowest]['name']}")
print(f"  - Time: {results[slowest]['win_time']:.1f} years")
print(f"  - More time allows safety to catch up")

lowest_debt = min(scenarios.keys(), key=lambda k: results[k]['win_debt'])
print(f"\nLOWEST RISK AT WIN: {results[lowest_debt]['name']}")
print(f"  - Safety debt: {results[lowest_debt]['win_debt']:.1f}")

print("\n" + "="*100)

