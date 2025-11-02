import numpy as np
import matplotlib.pyplot as plt
from CST_ag import Params, simulate, simple_scenario_policy_builder, safety_debt, State

# Configuration
t0, tf = 0.0, 50.0
t_eval = np.linspace(t0, tf, 501)
K0 = np.array([12.0, 9.0, 7.0])
S0 = np.array([0.2, 0.12, 0.15])
T0 = 0.1
y0 = np.concatenate([K0, S0, [T0]])
mode = "arms_race"

# Define "winning" threshold - first to reach this capability level wins
WIN_THRESHOLD = 50.0

# Baseline parameters
baseline_params = {
    'alpha': 1.0,
    'gamma': 0.5,
    'eta': 0.2,
    'beta': 0.3,
    'delta_T': 0.1,
    'theta': 0.8,
    'K_threshold': 10.0,
    'beta_dim': 0.3,
    'lam': 0.0
}

# Select just a few key parameters to test
param_ranges = {
    'alpha': np.linspace(0.6, 1.4, 8),       # capability growth rate
    'gamma': np.linspace(0.2, 0.8, 8),       # safety growth rate
    'K_threshold': np.linspace(8.0, 12.0, 8) # AGI threshold for takeoff
}

player_names = {0: 'US', 1: 'CN', 2: 'EU'}

def analyze_race_outcome(sol, params):
    """
    Determine who wins, when, and at what capability/safety state.
    """
    K_path = sol.y[0:3, :]
    S_path = sol.y[3:6, :]
    t = sol.t
    
    # Find first player to cross WIN_THRESHOLD
    winner = None
    win_time = None
    win_capability = None
    win_safety = None
    win_debt = None
    
    for i, ti in enumerate(t):
        for player_idx in range(3):
            if K_path[player_idx, i] >= WIN_THRESHOLD:
                winner = player_idx
                win_time = ti
                win_capability = K_path[player_idx, i]
                win_safety = S_path[player_idx, i]
                win_debt = max(0, win_capability - params.theta * win_safety)
                break
        if winner is not None:
            break
    
    # If no one wins by end of simulation
    if winner is None:
        # Choose leader at end
        final_K = K_path[:, -1]
        winner = np.argmax(final_K)
        win_time = t[-1]
        win_capability = K_path[winner, -1]
        win_safety = S_path[winner, -1]
        win_debt = max(0, win_capability - params.theta * win_safety)
    
    # Calculate safety ratio at win (higher = safer)
    safety_ratio = win_safety / win_capability if win_capability > 0 else 0
    
    return {
        'winner': winner,
        'win_time': win_time,
        'win_capability': win_capability,
        'win_safety': win_safety,
        'win_debt': win_debt,
        'safety_ratio': safety_ratio
    }

def run_single_sim(param_dict):
    """Run a single simulation with given parameters."""
    params = Params(**param_dict)
    policy_fn = simple_scenario_policy_builder(mode=mode)
    sol = simulate(
        t_span=(t0, tf),
        y0=y0,
        params=params,
        policy_fn=policy_fn,
        t_eval=t_eval,
    )
    
    return analyze_race_outcome(sol, params)

# Run sensitivity analysis
print("Running focused sensitivity analysis...")
print(f"Win threshold: First to reach K = {WIN_THRESHOLD}")
print(f"Scenario: {mode}\n")

results = {}

for param_name, param_values in param_ranges.items():
    print(f"Testing {param_name}...")
    results[param_name] = {
        'values': param_values,
        'winner': [],
        'win_time': [],
        'win_capability': [],
        'win_safety': [],
        'win_debt': [],
        'safety_ratio': []
    }
    
    for val in param_values:
        test_params = baseline_params.copy()
        test_params[param_name] = val
        
        res = run_single_sim(test_params)
        results[param_name]['winner'].append(res['winner'])
        results[param_name]['win_time'].append(res['win_time'])
        results[param_name]['win_capability'].append(res['win_capability'])
        results[param_name]['win_safety'].append(res['win_safety'])
        results[param_name]['win_debt'].append(res['win_debt'])
        results[param_name]['safety_ratio'].append(res['safety_ratio'])

# Create plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'Sensitivity Analysis: Race to K={WIN_THRESHOLD} ({mode.replace("_", " ").title()})', 
             fontsize=14, fontweight='bold')

param_labels = {
    'alpha': 'Capability Growth Rate',
    'gamma': 'Safety Growth Rate',
    'K_threshold': 'AGI Threshold'
}

colors = {0: 'blue', 1: 'red', 2: 'green'}
markers = {0: 'o', 1: 's', 2: '^'}

for idx, (param_name, data) in enumerate(results.items()):
    col = idx
    
    # Plot 1: Winner
    ax1 = axes[0, col]
    for pval, winner in zip(data['values'], data['winner']):
        ax1.scatter(pval, winner, color=colors[winner], marker=markers[winner], s=100, alpha=0.7)
    ax1.set_xlabel(param_labels[param_name], fontsize=10)
    ax1.set_ylabel('Winner', fontsize=10)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['US', 'CN', 'EU'])
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Race Winner')
    
    # Plot 2: Safety Ratio at Win (safety/capability)
    ax2 = axes[1, col]
    # Color by winner
    for i, (pval, ratio, winner) in enumerate(zip(data['values'], data['safety_ratio'], data['winner'])):
        ax2.scatter(pval, ratio, color=colors[winner], marker=markers[winner], s=100, alpha=0.7)
    ax2.set_xlabel(param_labels[param_name], fontsize=10)
    ax2.set_ylabel('Safety/Capability Ratio at Win', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Safety State When Race Won')
    ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='10% safety')
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='5% safety')
    ax2.legend(fontsize=8)

# Add legend for winners
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', label='US wins'),
    Patch(facecolor='red', label='CN wins'),
    Patch(facecolor='green', label='EU wins')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
           bbox_to_anchor=(0.5, 0.98), fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('plots/sensitivity_analysis_v2.png', dpi=150, bbox_inches='tight')
print("\nSaved plot to plots/sensitivity_analysis_v2.png")

# Print detailed findings
print("\n" + "="*80)
print("KEY FINDINGS: WHO WINS AND IN WHAT CONDITION?")
print("="*80)

for param_name, data in results.items():
    print(f"\n{'='*80}")
    print(f"{param_labels[param_name].upper()}: {data['values'][0]:.2f} to {data['values'][-1]:.2f}")
    print(f"{'='*80}")
    
    # Count wins
    win_counts = {0: 0, 1: 0, 2: 0}
    for w in data['winner']:
        win_counts[w] += 1
    
    print(f"\nWin Distribution:")
    for player_idx in range(3):
        pct = 100 * win_counts[player_idx] / len(data['winner'])
        print(f"  {player_names[player_idx]}: {win_counts[player_idx]}/{len(data['winner'])} ({pct:.0f}%)")
    
    # Find safest and most dangerous wins
    safest_idx = np.argmax(data['safety_ratio'])
    most_dangerous_idx = np.argmin(data['safety_ratio'])
    
    print(f"\nSafest Outcome:")
    print(f"  {param_name} = {data['values'][safest_idx]:.2f}")
    print(f"  Winner: {player_names[data['winner'][safest_idx]]}")
    print(f"  Safety/Capability ratio: {data['safety_ratio'][safest_idx]:.3f} ({100*data['safety_ratio'][safest_idx]:.1f}%)")
    print(f"  Capability at win: {data['win_capability'][safest_idx]:.1f}")
    print(f"  Safety at win: {data['win_safety'][safest_idx]:.2f}")
    print(f"  Safety debt at win: {data['win_debt'][safest_idx]:.1f}")
    
    print(f"\nMost Dangerous Outcome:")
    print(f"  {param_name} = {data['values'][most_dangerous_idx]:.2f}")
    print(f"  Winner: {player_names[data['winner'][most_dangerous_idx]]}")
    print(f"  Safety/Capability ratio: {data['safety_ratio'][most_dangerous_idx]:.3f} ({100*data['safety_ratio'][most_dangerous_idx]:.1f}%)")
    print(f"  Capability at win: {data['win_capability'][most_dangerous_idx]:.1f}")
    print(f"  Safety at win: {data['win_safety'][most_dangerous_idx]:.2f}")
    print(f"  Safety debt at win: {data['win_debt'][most_dangerous_idx]:.1f}")

print("\n" + "="*80)

