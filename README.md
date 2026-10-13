# Chasing Light, Breaking the Ship — simulation code

`iris_model.py` reproduces every number and figure in the paper (it imports `exact_elasticity.py`).

    pip install -r requirements.txt
    python iris_model.py --out .        # full study, ~4-5 min: self-tests -> results.json + figures/
    python iris_model.py --quick        # fast smoke test
    python iris_model.py --selftest     # physics/numerics checks only

The run stops if any self-test fails: relativistic rocket equation vs. direct integration of
lab-frame energy-momentum conservation, classical limit of the jet power, shield mass = areal
density x frontal area, fixed-point residual, Lambda = eps*Pi/(1+Pi) (and vs. finite differences)
on every feasible design among 3,000 random samples, Pi monotone in R, and Model A kinematic
ceiling = Model C kinematic ceiling.

Model: macroscopic energy-shielding model. Transit times are cruise times T = L/(beta c);
headline_facts() also integrates the one-year boost (+0.64 yr -> 25.71 yr at the 0.1743c design).

Figures used in the paper:
    Figure 1  fig_elasticity_exact.png   (exact Pi <= 10 Pareto front, from exact_elasticity.py)
    Figure 2  fig_models.png             (Models A, B, C)
    Figure 3  fig_sensitivity.png        (geometry and ISM density)
    Figure 4  fig_tornado.png            (one-at-a-time sensitivity)
Extra (not in the paper): fig_propulsion.png, fig_elasticity_nsga_binned.png.
The circular-dependency diagram in Section 2.3 is hand-drawn.
