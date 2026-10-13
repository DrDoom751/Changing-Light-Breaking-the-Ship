"""Exact (deterministic) Pareto front and marginal cost elasticity, Pi <= 10.
For each cruise speed beta, minimise launch mass M0 over specific impulse (R follows from the
rocket equation); T = L/(beta c), so E = |d ln M0 / d ln T| = |d ln M0 / d ln beta|."""
import json, sys, numpy as np
from scipy.optimize import minimize_scalar
import iris_model as m
B = m.BASE; C = m.C; G0 = m.G0

def best_M0(beta, p=B):
    def f(isp):
        u = isp * G0; R = np.exp(np.arctanh(beta) * C / u)
        if R < p.R_min or R > p.R_max: return 1e300
        d = m.design(isp, R, p)
        if not d.get("feasible") or d["Pi"] > p.pi_max: return 1e300
        return d["M0"]
    grid = np.geomspace(p.isp_min, p.isp_max, 200)
    vals = np.array([f(i) for i in grid]); j = int(np.argmin(vals))
    if vals[j] >= 1e300: return None, None
    lo, hi = grid[max(j-1, 0)], grid[min(j+1, len(grid)-1)]
    r = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
    return (r.fun, r.x) if r.fun < vals[j] else (vals[j], grid[j])

def run(out_png=None):
    ach = m.ceiling(B, B.pi_max)["beta"]
    betas = np.unique(np.concatenate([np.linspace(0.011, ach * (1 - 1e-6), 400), ach * (1 - np.geomspace(1e-2, 1e-6, 40))]))
    M0 = np.array([best_M0(b)[0] for b in betas], float)
    ok = np.isfinite(M0); betas, M0 = betas[ok], M0[ok]
    E = np.abs(np.gradient(np.log(M0), np.log(betas)))
    betas, M0, E = betas[1:], M0[1:], E[1:]          # drop the one-sided end point at low speed
    first = lambda lv: float(betas[np.argmax(E > lv)]) if np.any(E > lv) else None
    res = dict(beta_E_gt_1=first(1), beta_E_gt_10=first(10), beta_E_gt_30=first(30),
               E_at_0p172=float(np.interp(0.172, betas, E)), E_at_ceiling=float(E[-1]), beta_ceiling=float(ach),
               M0_at_ceiling=float(M0[-1]))
    if out_png:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        m._style()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.semilogy(betas, E, color=m.BLUE, lw=2)
        ax.axhline(1, color="0.4", ls="--", lw=1); ax.text(0.015, 1.15, "E = 1", fontsize=9, color="0.3")
        ax.axvline(ach, color="0.2", ls=":", lw=1.2)
        ax.text(ach - 0.002, 0.2, f"achievable ceiling\n{ach:.4f}c", ha="right", fontsize=9)
        ax.set_xlabel("Cruise speed β (fraction of c)"); ax.set_ylabel("E (% more launch mass per 1% shorter trip time)")
        ax.set_title("Marginal cost elasticity along the exact Π ≤ 10 Pareto front", loc="left", fontsize=11.5)
        fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
    return res

if __name__ == "__main__":
    r = run(sys.argv[1] if len(sys.argv) > 1 else None); print(json.dumps(r, indent=1))
