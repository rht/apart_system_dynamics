import numpy as np
import matplotlib.pyplot as plt
from CST_ag import Params, simulate, simple_scenario_policy_builder, State

# Simulation setup
t0, tf = 0.0, 30.0  # Shorter time to avoid exponential explosion
t_eval = np.linspace(t0, tf, 301)

player_names = {0: 'US', 1: 'CN', 2: 'EU'}

def analyze_outcomes(sol, params, scenario_name, framing='race'):
    """
    Analyze outcomes from either race or cooperation framing.
    
    Race framing: Who wins, when, at what safety level?
    Cooperation framing: Max risk, final collective state, safety maintenance
    """
    K_path = sol.y[0:3, :]
    S_path = sol.y[3:6, :]
    T_path = sol.y[6, :]
    t = sol.t
    
    # Calculate system-wide metrics over time
    total_capability = np.sum(K_path, axis=0)
    total_safety = np.sum(S_path, axis=0)
    system_safety_ratio = total_safety / total_capability
    
    # Calculate risk (debt) over time for each player
    debt_path = np.zeros_like(K_path)
    for i in range(len(t)):
        debt_path[:, i] = np.maximum(K_path[:, i] - params.theta * S_path[:, i], 0)
    total_debt = np.sum(debt_path, axis=0)
    max_debt = np.max(total_debt)
    
    # Individual safety ratios
    safety_ratios = S_path / (K_path + 1e-10)
    min_safety_ratio_over_time = np.min(safety_ratios, axis=0)
    
    results = {
        'name': scenario_name,
        't': t,
        'K_path': K_path,
        'S_path': S_path,
        'T_path': T_path,
        'total_capability': total_capability,
        'total_safety': total_safety,
        'system_safety_ratio': system_safety_ratio,
        'total_debt': total_debt,
        'max_debt': max_debt,
        'min_safety_ratio': np.min(min_safety_ratio_over_time),
        'final_system_safety_ratio': system_safety_ratio[-1],
        'final_total_capability': total_capability[-1],
        'final_total_safety': total_safety[-1],
        'final_debt': total_debt[-1],
    }
    
    # Race-specific metrics
    if framing == 'race':
        WIN_THRESHOLD = 50.0
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
        
        results['winner'] = winner
        results['win_time'] = win_time
        results['win_K'] = win_K
        results['win_S'] = win_S
        results['win_safety_ratio'] = win_safety_ratio
    
    return results

# Define scenarios
scenarios = {}

# Baseline Arms Race
scenarios['arms_race'] = {
    'name': 'Arms Race',
    'K0': np.array([5.0, 4.0, 3.0]),
    'S0': np.array([0.5, 0.4, 0.3]),
    'T0': 0.1,
    'params': Params(alpha=1.0, gamma=0.5, eta=0.2, beta=0.3, delta_T=0.1,
                    theta=0.8, K_threshold=8.0, beta_dim=0.3, lam=0.0),
    'mode': 'arms_race',
    'framing': 'race'
}

# International Cooperation
scenarios['cooperation'] = {
    'name': 'International Cooperation',
    'K0': np.array([5.0, 4.0, 3.0]),
    'S0': np.array([0.5, 0.4, 0.3]),
    'T0': 0.3,  # Start with more trust
    'params': Params(alpha=1.0, gamma=0.5, eta=0.6, beta=0.5, delta_T=0.05,
                    theta=0.8, K_threshold=8.0, beta_dim=0.3, lam=0.0),
    'mode': 'treaty',
    'framing': 'cooperation'
}

# Treaty (balanced approach)
scenarios['treaty'] = {
    'name': 'Limited Treaty',
    'K0': np.array([5.0, 4.0, 3.0]),
    'S0': np.array([0.5, 0.4, 0.3]),
    'T0': 0.1,
    'params': Params(alpha=1.0, gamma=0.5, eta=0.2, beta=0.3, delta_T=0.1,
                    theta=0.8, K_threshold=8.0, beta_dim=0.3, lam=0.0),
    'mode': 'treaty',
    'framing': 'cooperation'
}

# Run scenarios
print("Running scenario analysis...\n")
results = {}
for key, scenario in scenarios.items():
    print(f"Running: {scenario['name']}")
    y0 = np.concatenate([scenario['K0'], scenario['S0'], [scenario['T0']]])
    policy_fn = simple_scenario_policy_builder(mode=scenario['mode'])
    
    sol = simulate(
        t_span=(t0, tf),
        y0=y0,
        params=scenario['params'],
        policy_fn=policy_fn,
        t_eval=t_eval,
    )
    
    results[key] = analyze_outcomes(sol, scenario['params'], scenario['name'], 
                                    framing=scenario['framing'])

# Create comparison plot
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35, top=0.93, bottom=0.08)

colors_scenario = {'arms_race': '#e41a1c', 'cooperation': '#4daf4a', 'treaty': '#984ea3'}
colors_player = ['blue', 'red', 'green']
player_labels = ['US', 'CN', 'EU']

# Row 1: Individual capabilities over time
for idx, (key, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[0, idx])
    for i in range(3):
        ax.plot(res['t'], res['K_path'][i, :], color=colors_player[i], 
                label=player_labels[i], linewidth=2)
    ax.axhline(y=8, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='AGI threshold')
    ax.set_ylabel('Capability (K)', fontsize=10)
    ax.set_title(res['name'], fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 100)

# Row 2: Individual safety levels
for idx, (key, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, idx])
    for i in range(3):
        ax.plot(res['t'], res['S_path'][i, :], color=colors_player[i], 
                label=player_labels[i], linewidth=2)
    ax.set_ylabel('Safety (S)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 100)

# Row 3: System-wide metrics
ax_sys = fig.add_subplot(gs[2, :])
for key, res in results.items():
    ax_sys.plot(res['t'], 100 * res['system_safety_ratio'], 
                color=colors_scenario[key], linewidth=3, label=res['name'])
ax_sys.set_xlabel('Time (years)', fontsize=11)
ax_sys.set_ylabel('System-Wide Safety/Capability Ratio (%)', fontsize=11)
ax_sys.set_title('Collective Safety State Over Time', fontsize=12, fontweight='bold')
ax_sys.axhline(y=10, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='10% threshold')
ax_sys.axhline(y=5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='5% threshold')
ax_sys.legend(fontsize=10, loc='best')
ax_sys.grid(True, alpha=0.3)
ax_sys.set_xlim(0, 30)
ax_sys.set_ylim(0, 120)

plt.savefig('plots/scenario_analysis_revised.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to plots/scenario_analysis_revised.png")

# Print comparison table
print("\n" + "="*100)
print("SCENARIO COMPARISON: Race vs Cooperation Framing")
print("="*100)

print("\n" + "-"*100)
print("RACE FRAMING (if applicable)")
print("-"*100)
print(f"{'Scenario':<30} {'Winner':<8} {'Time':<10} {'Win K':<10} {'Win S':<10} {'Safety%':<10}")
print("-"*100)
for key, res in results.items():
    if 'winner' in res:
        print(f"{res['name']:<30} {player_names[res['winner']]:<8} {res['win_time']:<10.1f} "
              f"{res['win_K']:<10.1f} {res['win_S']:<10.2f} {100*res['win_safety_ratio']:<10.2f}")

print("\n" + "-"*100)
print("COOPERATION FRAMING (system-wide outcomes)")
print("-"*100)
print(f"{'Scenario':<30} {'Final System':<15} {'Final System':<15} {'Max Risk':<12} {'Min Safety%':<12}")
print(f"{'':30} {'Capability':<15} {'Safety':<15} {'(Debt)':<12} {'Any Player':<12}")
print("-"*100)
for key, res in results.items():
    print(f"{res['name']:<30} {res['final_total_capability']:<15.1f} {res['final_total_safety']:<15.2f} "
          f"{res['max_debt']:<12.1f} {100*res['min_safety_ratio']:<12.2f}")

print("="*100)

# Key insights
print("\nKEY INSIGHTS:")
print("-" * 100)

best_system_safety = max(results.keys(), key=lambda k: results[k]['final_system_safety_ratio'])
print(f"BEST COLLECTIVE SAFETY: {results[best_system_safety]['name']}")
print(f"  - Final system safety ratio: {100*results[best_system_safety]['final_system_safety_ratio']:.2f}%")
print(f"  - Maximum risk encountered: {results[best_system_safety]['max_debt']:.1f}")

lowest_risk = min(results.keys(), key=lambda k: results[k]['max_debt'])
print(f"\nLOWEST MAXIMUM RISK: {results[lowest_risk]['name']}")
print(f"  - Max debt over all time: {results[lowest_risk]['max_debt']:.1f}")
print(f"  - Final system safety: {100*results[lowest_risk]['final_system_safety_ratio']:.2f}%")

print("\n" + "="*100)
print("\nNOTE: In cooperation scenarios, 'winning' is less relevant than collective safety outcomes.")
print("Focus on: system-wide safety ratios, maximum risk exposure, and whether all players develop safely.")
print("="*100)

