AI Race Sim (SciPy-only) — Quick Start
--------------------------------------
Run:  python ai_race_sim.py
Dependencies: numpy, scipy, matplotlib

Files produced:
 - times.npy, states.npy, actions.npy (optional outputs)
 - fig_compute.png, fig_resource.png, fig_aS.png, fig_aV.png, fig_aX.png

How to extend:
 - Replace choose_actions_myopic() with iterative best-responses or Nash search.
 - Refine payoff_for_bloc() functional forms and calibrate parameters.
 - Add additional stocks (talent, chips), and explicit export-control knobs.
 - Add scenario loader (YAML/JSON) and sweep experiments.
