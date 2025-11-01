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

# Parameters to test
param_ranges = {
    'alpha': np.linspace(0.5, 2.0, 10),      # capability growth efficiency
    'gamma': np.linspace(0.1, 1.0, 10),      # safety growth efficiency
    'eta': np.linspace(0.0, 0.5, 10),        # safety spillover strength
    'theta': np.linspace(0.4, 1.2, 10),      # safety effectiveness
    'K_threshold': np.linspace(5.0, 15.0, 10) # AGI threshold
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
    
    K_path = sol.y[0:3, :]
    S_path = sol.y[3:6, :]
    T_path = sol.y[6, :]
    
    # Compute final safety debt
    st_final = State(K=K_path[:, -1], S=S_path[:, -1], T=T_path[-1])
    debt_final = safety_debt(st_final, params)
    
    return {
        'K_final': K_path[:, -1],
        'S_final': S_path[:, -1],
        'T_final': T_path[-1],
        'debt_final': debt_final,
        'K_max': np.max(K_path),
        'total_debt': np.sum(debt_final),
    }

# Run sensitivity analysis
print("Running sensitivity analysis...")
results = {}

for param_name, param_values in param_ranges.items():
    print(f"Testing {param_name}...")
    results[param_name] = {
        'values': param_values,
        'K_final_US': [],
        'K_final_CN': [],
        'K_final_EU': [],
        'S_final_US': [],
        'S_final_CN': [],
        'S_final_EU': [],
        'total_debt': [],
        'max_capability': []
    }
    
    for val in param_values:
        test_params = baseline_params.copy()
        test_params[param_name] = val
        
        res = run_single_sim(test_params)
        results[param_name]['K_final_US'].append(res['K_final'][0])
        results[param_name]['K_final_CN'].append(res['K_final'][1])
        results[param_name]['K_final_EU'].append(res['K_final'][2])
        results[param_name]['S_final_US'].append(res['S_final'][0])
        results[param_name]['S_final_CN'].append(res['S_final'][1])
        results[param_name]['S_final_EU'].append(res['S_final'][2])
        results[param_name]['total_debt'].append(res['total_debt'])
        results[param_name]['max_capability'].append(res['K_max'])

# Create plots
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle('Sensitivity Analysis - Arms Race Scenario', fontsize=16, fontweight='bold')

param_labels = {
    'alpha': 'Capability Growth Rate (alpha)',
    'gamma': 'Safety Growth Rate (gamma)',
    'eta': 'Safety Spillover (eta)',
    'theta': 'Safety Effectiveness (theta)',
    'K_threshold': 'AGI Threshold'
}

for idx, (param_name, data) in enumerate(results.items()):
    col = idx
    
    # Plot 1: Final capabilities by player
    ax1 = axes[0, col]
    ax1.plot(data['values'], data['K_final_US'], 'b-o', label='US', markersize=4)
    ax1.plot(data['values'], data['K_final_CN'], 'r-s', label='CN', markersize=4)
    ax1.plot(data['values'], data['K_final_EU'], 'g-^', label='EU', markersize=4)
    ax1.set_xlabel(param_labels[param_name])
    ax1.set_ylabel('Final Capability')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final safety by player
    ax2 = axes[1, col]
    ax2.plot(data['values'], data['S_final_US'], 'b-o', label='US', markersize=4)
    ax2.plot(data['values'], data['S_final_CN'], 'r-s', label='CN', markersize=4)
    ax2.plot(data['values'], data['S_final_EU'], 'g-^', label='EU', markersize=4)
    ax2.set_xlabel(param_labels[param_name])
    ax2.set_ylabel('Final Safety')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Total safety debt
    ax3 = axes[2, col]
    ax3.plot(data['values'], data['total_debt'], 'k-o', markersize=4)
    ax3.set_xlabel(param_labels[param_name])
    ax3.set_ylabel('Total Safety Debt')
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(data['values'], 0, data['total_debt'], alpha=0.3, color='red')

plt.tight_layout()
plt.savefig('plots/sensitivity_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved sensitivity analysis plot to plots/sensitivity_analysis.png")

# Print some key findings
print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

for param_name, data in results.items():
    print(f"\n{param_labels[param_name]}:")
    
    # Find min/max outcomes
    min_idx = np.argmin(data['total_debt'])
    max_idx = np.argmax(data['total_debt'])
    
    print(f"  Range tested: {data['values'][0]:.2f} to {data['values'][-1]:.2f}")
    print(f"  Min total debt: {data['total_debt'][min_idx]:.2f} (at {param_name}={data['values'][min_idx]:.2f})")
    print(f"  Max total debt: {data['total_debt'][max_idx]:.2f} (at {param_name}={data['values'][max_idx]:.2f})")
    
    # Correlation with debt
    corr = np.corrcoef(data['values'], data['total_debt'])[0, 1]
    print(f"  Correlation with total debt: {corr:.3f}")

print("\n" + "="*70)

