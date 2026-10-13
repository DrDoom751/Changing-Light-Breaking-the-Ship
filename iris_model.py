"""
Chasing Light, Breaking the Ship -- coupled shielding / propulsion model (version 4).

Written from scratch. One file reproduces every number, table, and figure in the paper:

    python iris_model.py            # full run (~5 min): self-tests, results.json, figures/
    python iris_model.py --quick    # smoke test with small optimizer budgets
    python iris_model.py --selftest # physics self-tests only

PHYSICS (all exact within the stated model assumptions)
  * Relativistic rocket (Ackeret): cruise rapidity  phi = (u/c) ln R,  beta = tanh(phi),
    where u is the exhaust speed in the rocket's instantaneous rest frame and R = M0/Mf.
    Isp is defined as u / g0.
  * Relativistic jet power. Converting rocket rest mass dM into exhaust of speed u gives
    exhaust kinetic energy (1 - 1/gamma_u) c^2 per unit dM consumed, so
        P_jet = (1 - 1/gamma_u) c^2 |dM/dtau|      (-> 1/2 u^2 |dM/dtau| when u << c).
  * Cruise: the full distance L is flown at the constant cruise speed beta (the acceleration
    phase is not part of the objective). Cruise time T = L/(beta c); ship proper time T/gamma.
    headline_facts() integrates the one-year boost separately to report the total trip time.
  * Fluence on the shield (column density rho*L is frame-invariant):
        q_dep = rho_eff * eta(beta) (gamma - 1) c^2 L,
        rho_eff = rho_dust + chi_g rho_gas,   eta(beta) = eta0 exp(-beta / eta_scale)
  * Shield areal density  sigma = q_dep / [E_rem (1 - s_frac)].
  * Mass budget  Mf = payload + shield + tankage + engine,
        tankage = f_t (R-1) Mf,  engine = alpha_eng P_jet = beta_eng Mf,
        beta_eng = alpha_eng (R-1)(1 - 1/gamma_u) c^2 / t_b,   k = 1 - f_t (R-1) - beta_eng
  * Coupled frontal area  A = A_ref (M0 / M_ref)^eps  -> fixed point
        Mf = [P + sigma A_ref (R Mf / M_ref)^eps] / k
    solved exactly (closed form for eps = 0 or 1, bracketed Brent root otherwise).
    Local contraction factor Lambda = F'(Mf*) = eps * Pi / (1 + Pi), Pi = M_shield / M_payload.
    (The geometry exponent eps is stored in the field `gamma_geom`; it is NOT the Lorentz factor.)

MODELS   A: no shield   B: shield with fixed area A_ref   C: coupled area (paper's model)
OBJECTIVES  constant-velocity cruise time to Alpha Centauri, log10 launch mass.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, replace, asdict

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore", category=RuntimeWarning)   # overflow in infeasible trial designs is expected

# ------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------
C = 299_792_458.0                 # m/s
G0 = 9.80665                      # m/s^2
LY = 9.460730472e15               # m (Julian light-year)
YEAR = 365.25 * 86400.0           # s (Julian year)
AU = 1.495978707e11               # m
TNT_J_PER_KG = 4.184e6



@dataclass(frozen=True)
class Params:
    distance_ly: float = 4.37     # Alpha Centauri
    payload: float = 1000.0       # kg
    rho_gas: float = 1.67e-22     # kg/m^3 (n ~ 0.1 cm^-3)
    rho_dust: float = 1.67e-24    # kg/m^3 (dust-to-gas 1:100)
    chi_g: float = 0.1            # gas damage multiplier
    eta0: float = 0.5             # punch-through prefactor
    eta_scale: float = 0.5        # eta = eta0 exp(-beta/eta_scale) = 0.5 exp(-2 beta)
    e_rem: float = 3.3e7          # J/kg, Be effective removal energy
    s_frac: float = 0.10          # surviving shield fraction (safety margin)
    rho_be: float = 1850.0        # kg/m^3, beryllium (for thickness reporting)
    f_t: float = 0.05             # tankage / propellant mass
    alpha_eng: float = 1e-8       # kg/W engine specific mass
    t_burn_yr: float = 1.0        # proper burn time
    A_ref: float = 100.0          # m^2
    M_ref: float = 5.0e4          # kg
    gamma_geom: float = 2.0 / 3.0 # A ~ M0^gamma
    pi_max: float = 10.0          # shield/payload viability limit
    isp_min: float = 1.0e5        # s
    isp_max: float = 3.0e6        # s  (u_max = 0.0981 c)
    R_min: float = 1.1
    R_max: float = 100.0
    beta_min: float = 0.01
    model: str = "C"

    @property
    def L(self): return self.distance_ly * LY
    @property
    def t_b(self): return self.t_burn_yr * YEAR
    @property
    def rho_eff(self): return self.rho_dust + self.chi_g * self.rho_gas


BASE = Params()

# ------------------------------------------------------------------------------------
# Physics
# ------------------------------------------------------------------------------------
def jet_energy_per_mass(u):
    """(1 - 1/gamma_u) c^2, written to stay accurate for u << c."""
    b2 = (u / C) ** 2
    return C**2 * b2 / (1.0 + np.sqrt(1.0 - b2))


def eta(beta, p: Params):
    return p.eta0 * np.exp(-beta / p.eta_scale)


def gm1(beta):
    """gamma - 1, accurate for small beta."""
    b2 = beta * beta
    s = np.sqrt(1.0 - b2)
    return b2 / (s * (1.0 + s))


def k_factor(u, R, p: Params):
    beta_eng = p.alpha_eng * (R - 1.0) * jet_energy_per_mass(u) / p.t_b
    return 1.0 - p.f_t * (R - 1.0) - beta_eng, beta_eng


def R_kzero(u, p: Params):
    """Mass ratio at which k = 0 (engine + tanks consume the whole final mass)."""
    return 1.0 + 1.0 / (p.f_t + p.alpha_eng * jet_energy_per_mass(u) / p.t_b)


def trajectory(u, R, p: Params):
    """Cruise kinematics and fluence per unit rho_eff (independent of the absolute mass scale)."""
    beta = np.tanh((u / C) * np.log(R))
    g = 1.0 + gm1(beta)
    T = p.L / (beta * C) / YEAR if beta > 0 else np.inf
    return dict(beta=beta, gamma=g, T=T, tau=T / g, fluence=eta(beta, p) * gm1(beta) * C**2 * p.L)


def _fixed_point(P, sigma, R, k, p: Params, gamma):
    """Smallest root of h(M) = M - [P + sigma A_ref (R M/M_ref)^gamma]/k. None if no root."""
    a = P / k
    b = sigma * p.A_ref * (R / p.M_ref) ** gamma / k     # h(M) = M - a - b M^gamma
    if gamma == 0.0:
        return a + sigma * p.A_ref / k
    if gamma == 1.0:
        return a / (1.0 - b) if b < 1.0 else None
    h = lambda M: M - a - b * M**gamma
    if gamma < 1.0:                                       # unique root, h(a) < 0
        hi = 2.0 * a
        while h(hi) < 0.0:
            hi *= 2.0
            if hi > 1e60:
                return None
        return brentq(h, a, hi, xtol=1e-300, rtol=1e-15, maxiter=500)
    # gamma > 1: h concave, maximum where b gamma M^(gamma-1) = 1
    M_star = (1.0 / (b * gamma)) ** (1.0 / (gamma - 1.0))
    if M_star <= a or h(M_star) < 0.0:
        return None
    return brentq(h, a, M_star, xtol=1e-300, rtol=1e-15, maxiter=500)


def design(isp, R, p: Params = BASE, model: str | None = None):
    """Evaluate one design. Returns a dict; feasible=False carries 'violation' > 0."""
    model = model or p.model
    u = isp * G0
    out = dict(isp=isp, R=R, u=u, feasible=False, violation=0.0, reason="")
    if u >= C:
        out.update(violation=u / C - 1.0 + 1e-9, reason="u>=c"); return out
    tr = trajectory(u, R, p)
    out.update(tr)
    k, beta_eng = k_factor(u, R, p)
    out.update(k=k, beta_eng=beta_eng)
    if tr["beta"] <= p.beta_min:
        out.update(violation=p.beta_min - tr["beta"] + 1e-9, reason="beta<=beta_min"); return out
    if k <= 0.0:
        out.update(violation=-k + 1e-9, reason="k<=0"); return out

    q_dep = p.rho_eff * tr["fluence"]
    sigma = q_dep / (p.e_rem * (1.0 - p.s_frac))
    P = p.payload
    if model == "A":
        Mf, S, Lam, area = P / k, 0.0, 0.0, 0.0
    else:
        g = 0.0 if model == "B" else p.gamma_geom
        Mf = _fixed_point(P, sigma, R, k, p, g)
        if Mf is None:
            out.update(violation=1.0, reason="no fixed point"); return out
        S = Mf * k - P
        area = p.A_ref * (R * Mf / p.M_ref) ** g
        Lam = 0.0 if g == 0.0 else g * sigma * p.A_ref * R**g * Mf ** (g - 1.0) / (k * p.M_ref**g)
        if Lam >= 1.0:
            out.update(violation=Lam - 1.0 + 1e-9, reason="Lambda>=1"); return out
    M0 = R * Mf
    out.update(feasible=True, q_dep=q_dep, sigma=sigma, Mf=Mf, M0=M0, shield=S, Pi=S / P,
               tank=p.f_t * (R - 1.0) * Mf, engine=beta_eng * Mf, propellant=(R - 1.0) * Mf,
               area=area, Lam=Lam, thickness_mm=sigma / p.rho_be * 1e3)
    return out


# ------------------------------------------------------------------------------------
# Exact ceilings (deterministic; no optimizer noise)
# ------------------------------------------------------------------------------------
def _R_low(u, p):
    return max(p.R_min, np.exp(np.arctanh(p.beta_min * 1.0001) / (u / C)))


def _R_edge(u, p, model="C"):
    """Supremum of feasible R at fixed u (feasibility is monotone in R)."""
    lo = _R_low(u, p)
    hi = min(R_kzero(u, p), p.R_max)
    if lo >= hi or not design(u / G0, lo, p, model)["feasible"]:
        return None
    top = lo + (hi - lo) * (1.0 - 1e-12)
    if design(u / G0, top, p, model)["feasible"]:
        return top
    a, b = lo, top
    for _ in range(200):
        mid = 0.5 * (a + b)
        if design(u / G0, mid, p, model)["feasible"]: a = mid
        else: b = mid
        if b - a < 1e-13 * b: break
    return a


def _boundary(u, p, pi_max, model="C"):
    """Fastest admissible R at fixed u: Pi = pi_max, or the feasibility edge."""
    Re = _R_edge(u, p, model)
    if Re is None:
        return None
    if pi_max is None or model == "A":
        return Re
    lo = _R_low(u, p)
    Pi = lambda r: design(u / G0, r, p, model)["Pi"] - pi_max
    if Pi(lo) > 0: return None
    if Pi(Re) <= 0: return Re
    return brentq(Pi, lo, Re, xtol=1e-300, rtol=1e-13, maxiter=300)


def ceiling(p: Params = BASE, pi_max=None, u_max=None, model="C", n_grid=60):
    """Minimum cruise time (i.e. maximum cruise speed) over admissible designs.
    pi_max=None -> kinematic ceiling (shield/payload unrestricted, k->0 or fixed-point edge)."""
    u_lo = p.isp_min * G0
    u_hi = (u_max if u_max is not None else p.isp_max * G0)
    u_hi = min(u_hi, 0.999 * C)

    def T_of(u):
        R = _boundary(u, p, pi_max, model)
        return (np.inf, None) if R is None else (trajectory(u, R, p)["T"], R)

    grid = np.linspace(u_lo, u_hi, n_grid)
    vals = [T_of(u)[0] for u in grid]
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
    r = minimize_scalar(lambda u: T_of(u)[0], bounds=(a, b), method="bounded",
                        options=dict(xatol=1e-6 * u_hi))
    cands = [(vals[i], grid[i]), (r.fun, r.x), (T_of(u_hi)[0], u_hi)]
    T, u = min(cands)
    R = T_of(u)[1]
    tr = trajectory(u, R, p)
    d = design(u / G0, R, p, model)
    return dict(T=T, beta=tr["beta"], tau=tr["tau"],
                isp=u / G0, u_over_c=u / C, R=R, at_isp_cap=bool(abs(u - u_hi) < 1e-6 * u_hi),
                Pi=d.get("Pi", np.inf) if d["feasible"] else float("inf"),
                M0=d.get("M0", np.inf) if d["feasible"] else float("inf"),
                design=d if d["feasible"] else None)


# ------------------------------------------------------------------------------------
# Multi-objective optimization (NSGA-II, constraint-handled)
# ------------------------------------------------------------------------------------
def _pymoo():
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    return ElementwiseProblem, NSGA2, minimize, NonDominatedSorting


def make_problem(p: Params, model="C", pi_max=None):
    ElementwiseProblem = _pymoo()[0]

    class Flight(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=2, n_obj=2, n_ieq_constr=2,
                             xl=np.array([p.isp_min, p.R_min]), xu=np.array([p.isp_max, p.R_max]))

        def _evaluate(self, x, out, *a, **kw):
            d = design(x[0], x[1], p, model)
            if d["feasible"]:
                out["F"] = [d["T"], np.log10(d["M0"])]
                out["G"] = [-1.0, (d["Pi"] - pi_max) if pi_max is not None else -1.0]
            else:
                out["F"] = [1e4, 30.0]
                out["G"] = [d["violation"], -1.0]
    return Flight()


def pareto_front(p: Params = BASE, model="C", pi_max=None, seeds=10, pop=100, gen=80):
    _, NSGA2, minimize, NDS = _pymoo()
    prob = make_problem(p, model, pi_max)
    Xs, Fs = [], []
    for s in range(1, seeds + 1):
        res = minimize(prob, NSGA2(pop_size=pop), ("n_gen", gen), seed=s, verbose=False)
        X, F, CV = np.atleast_2d(res.X), np.atleast_2d(res.F), np.atleast_1d(res.CV.ravel())
        ok = CV <= 0
        if ok.any():
            Xs.append(X[ok]); Fs.append(F[ok])
    X, F = np.vstack(Xs), np.vstack(Fs)
    nd = NDS().do(F, only_non_dominated_front=True)
    X, F = X[nd], F[nd]
    o = np.argsort(F[:, 0])
    return X[o], np.column_stack([F[o, 0], 10 ** F[o, 1]])      # (T [yr], M0 [kg])


def validate(p: Params = BASE, seeds=30, pop=100, gen=80, n_grid=160):
    """IGD / Hypervolume of NSGA-II vs a 160x160 = 25,600-design reference grid."""
    _, NSGA2, minimize, NDS = _pymoo()
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.hv import HV
    G = []
    for isp in np.linspace(p.isp_min, p.isp_max, n_grid):
        for R in np.geomspace(p.R_min, p.R_max, n_grid):
            d = design(isp, R, p)
            if d["feasible"]:
                G.append([d["T"], np.log10(d["M0"])])
    G = np.array(G)
    ref = G[NDS().do(G, only_non_dominated_front=True)]
    lo, hi = ref.min(0), ref.max(0)
    nz = lambda A: (A - lo) / (hi - lo)
    igd, hv = IGD(nz(ref)), HV(ref_point=np.array([1.1, 1.1]))
    prob = make_problem(p)
    I, H, best = [], [], []
    for s in range(1, seeds + 1):
        res = minimize(prob, NSGA2(pop_size=pop), ("n_gen", gen), seed=s, verbose=False)
        ok = res.CV.ravel() <= 0
        F = res.F[ok]
        I.append(igd(nz(F))); H.append(hv(nz(F))); best.append(F[:, 0].min())
    I, H = np.array(I), np.array(H)
    ci = lambda a: float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))
    return dict(grid_total=n_grid**2, grid_feasible=len(G), ref_front_points=len(ref),
                seeds=seeds, pop=pop, gen=gen,
                IGD_mean=float(I.mean()), IGD_ci95=ci(I), HV_mean=float(H.mean()), HV_ci95=ci(H),
                HV_grid=float(hv(nz(ref))), HV_ratio=float(H.mean() / hv(nz(ref))),
                grid_best_T=float(ref[:, 0].min()), nsga_best_T_mean=float(np.mean(best)))


# ------------------------------------------------------------------------------------
# Self-tests: physics identities and numerics (run before every full run)
# ------------------------------------------------------------------------------------
def selftest(p: Params = BASE, verbose=True):
    res = {}
    rng = np.random.default_rng(42)

    # 1. Relativistic rocket equation vs direct integration of lab-frame energy and momentum
    #    conservation: each proper-time step the rocket emits exhaust with 4-momentum
    #    (E, p) = (|dM|, -(u/c)|dM|) in its rest frame (units M0 = c = 1); boost to the lab
    #    frame and subtract. Independent of the closed form beta = tanh((u/c) ln R).
    u, R = 0.12 * C, 6.0; bu = u / C
    def rhs(tau, y):                       # y = [E, pz] of the rocket in the lab frame
        E, pz = y; M = np.sqrt(E * E - pz * pz); g, b = E / M, pz / E
        dM = -(1.0 - 1.0 / R) / p.t_b      # rest-mass loss rate (constant)
        Ep, pp = -dM, bu * dM              # exhaust energy / momentum rates in the rest frame
        return [-g * (Ep + b * pp), -g * (pp + b * Ep)]
    sol = solve_ivp(rhs, (0, p.t_b), [1.0, 0.0], rtol=1e-12, atol=1e-14)
    E_end, pz_end = sol.y[:, -1]
    res["rocket_eq_rel_err"] = abs((pz_end / E_end) / np.tanh(bu * np.log(R)) - 1.0)
    res["rocket_final_mass_rel_err"] = abs(np.sqrt(E_end**2 - pz_end**2) * R - 1.0)

    # 2. Classical limit of jet power
    uu = 1e3
    res["jet_power_classical_limit_rel_err"] = abs(jet_energy_per_mass(uu) / (0.5 * uu**2) - 1.0)

    # 3-6. Shield mass = sigma x area, fixed-point residual, Lambda identity, finite-difference Lambda
    worst = dict(closure=0.0, residual=0.0, lam_identity=0.0, lam_fd=0.0)   # closure = shield vs sigma*A
    n_ok = 0
    for _ in range(3000):
        d = design(rng.uniform(p.isp_min, p.isp_max), rng.uniform(p.R_min, p.R_max), p)
        if not d["feasible"]: continue
        n_ok += 1
        worst["closure"] = max(worst["closure"], abs(d["shield"] / (d["sigma"] * d["area"]) - 1))
        g, sig, k, Rr = p.gamma_geom, d["sigma"], d["k"], d["R"]
        F = lambda M: (p.payload + sig * p.A_ref * (Rr * M / p.M_ref) ** g) / k
        worst["residual"] = max(worst["residual"], abs(F(d["Mf"]) / d["Mf"] - 1))
        worst["lam_identity"] = max(worst["lam_identity"], abs(d["Lam"] - g * d["Pi"] / (1 + d["Pi"])))
        h = 1e-6 * d["Mf"]
        fd = (F(d["Mf"] + h) - F(d["Mf"] - h)) / (2 * h)
        worst["lam_fd"] = max(worst["lam_fd"], abs(fd - d["Lam"]))
    res.update({f"{k}_max": v for k, v in worst.items()}, feasible_samples=n_ok)
    # consistency check (true by construction for eps < 1, kept as a guard against code edits)
    res["max_Lambda_seen"] = max(design(rng.uniform(p.isp_min, p.isp_max), rng.uniform(p.R_min, p.R_max), p).get("Lam", 0) or 0 for _ in range(2000))

    # 7. Pi strictly increasing in R (needed for the exact ceiling root-find)
    mono = True
    for isp in np.linspace(1e6, p.isp_max, 8):
        Re = _R_edge(isp * G0, p)
        if Re is None: continue
        Rs = np.linspace(_R_low(isp * G0, p), Re, 120)
        Pis = [design(isp, r, p)["Pi"] for r in Rs]
        mono &= bool(np.all(np.diff(Pis) > 0))
    res["Pi_monotone_in_R"] = mono

    # 8. Consistency check: the kinematic edge (k -> 0) must not depend on the shield model
    res["kinematic_A_minus_C_beta"] = ceiling(p, None, model="A")["beta"] - ceiling(p, None, model="C")["beta"]

    tol = dict(rocket_eq_rel_err=1e-9, rocket_final_mass_rel_err=1e-9, jet_power_classical_limit_rel_err=1e-9,
               closure_max=1e-12, residual_max=1e-12, lam_identity_max=1e-10, lam_fd_max=1e-6)
    ok = all(res[k] < v for k, v in tol.items()) and res["Pi_monotone_in_R"] \
        and abs(res["kinematic_A_minus_C_beta"]) < 1e-9 and res["max_Lambda_seen"] < p.gamma_geom
    res["ALL_PASS"] = bool(ok)
    if verbose:
        for k_, v in res.items():
            print(f"   {k_:38s} {v}")
    if not ok:
        raise AssertionError("self-test failed")
    return res


# ------------------------------------------------------------------------------------
# Figures (no 'Figure N' baked into images; numbering lives in the paper captions)
# ------------------------------------------------------------------------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "legend.frameon": False, "lines.linewidth": 2.0, "savefig.dpi": 300,
        "figure.facecolor": "white", "axes.facecolor": "white"})
    return plt


def elasticity(F, betas, nbins=60):
    """E = -d ln M0 / d ln T on a front binned in ln T (median per bin)."""
    lt, lm = np.log(F[:, 0]), np.log(F[:, 1])
    edges = np.linspace(lt.min(), lt.max(), nbins + 1)
    idx = np.clip(np.digitize(lt, edges) - 1, 0, nbins - 1)
    keep = [i for i in range(nbins) if np.any(idx == i)]
    bt = np.array([np.median(lt[idx == i]) for i in keep])
    bm = np.array([np.median(lm[idx == i]) for i in keep])
    bb = np.array([np.median(betas[idx == i]) for i in keep])
    E = -np.gradient(bm, bt)
    ok = np.isfinite(E) & (E > 0)
    o = np.argsort(bb[ok])
    return bb[ok][o], E[ok][o], np.exp(bt[ok][o])


def fig_elasticity(beta, E, beta_ach, path):
    plt = _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(beta, E, color=BLUE, lw=2, marker="o", ms=3.5)
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.text(0.012, 1.1, "E = 1", color=INK2, ha="left", va="bottom", fontsize=9)
    ax.axvline(beta_ach, color=INK, lw=1, ls=":")
    ax.text(beta_ach - 0.002, E.min() * 1.2, f"achievable ceiling\n{beta_ach:.4f}c", color=INK, ha="right", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("Cruise speed β (fraction of c)")
    ax.set_ylabel("E (% more launch mass per 1% shorter trip time)")
    ax.set_title("Marginal cost elasticity along the Π ≤ 10 Pareto front", color=INK, fontsize=11.5, loc="left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_models(fronts, path):
    plt = _style()
    from scipy.interpolate import interp1d
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    spec = [("A", "Model A: no shield", AQUA, ":"), ("B", "Model B: fixed-area shield", ORANGE, "--"),
            ("C", "Model C: coupled shield (this paper)", BLUE, "-")]
    m_c = np.logspace(3.2, 10, 600)
    for key, lab, col, ls in spec:
        base, lo_e, hi_e = fronts[key]
        ax.plot(base[:, 0], base[:, 1], color=col, ls=ls, lw=2.2, label=lab)
        if key != "A":
            ts = []
            for f in (base, lo_e, hi_e):
                u, i = np.unique(np.log10(f[:, 1]), return_index=True)
                ts.append(interp1d(u, f[i, 0], bounds_error=False)(np.log10(m_c)))
            ax.fill_betweenx(m_c, np.nanmin(ts, 0), np.nanmax(ts, 0), color=col, alpha=0.15, lw=0)
    ax.set_yscale("log"); ax.set_xlim(22.5, 60); ax.set_ylim(1e3, 1e10)
    ax.set_xlabel("Cruise time to Alpha Centauri (years)")
    ax.set_ylabel("Launch mass M₀ (kg)")
    ax.set_title(r"Pareto fronts: launch mass vs trip time (bands: $E_{\rm rem}$ ± 20%)", color=INK, fontsize=11.5, loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_sensitivity(fronts, path):
    plt = _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    spec = [("base", "Baseline (ε = 2/3)", BLUE, "-"), ("needle", "Needle geometry (ε = 1/2)", AQUA, "-."),
            ("linear", "Linear geometry (ε = 1)", ORANGE, "--"), ("harsh", "Harsh ISM (10× density)", MAGENTA, (0, (5, 1, 1, 1))),
            ("benign", "Benign ISM (0.1× density)", VIOLET, ":")]
    for key, lab, col, ls in spec:
        f = fronts[key]
        ax.plot(f[:, 0], f[:, 1], color=col, ls=ls, lw=2.2, label=lab)
    ax.set_yscale("log"); ax.set_xlim(22.5, 70); ax.set_ylim(1e3, 1e10)
    ax.set_xlabel("Cruise time to Alpha Centauri (years)")
    ax.set_ylabel("Launch mass M₀ (kg)")
    ax.set_title("Sensitivity of the Pareto front to geometry and ISM density", color=INK, fontsize=11.5, loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_propulsion(sweep, path):
    plt = _style()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    a = ax[0]
    a.plot(sweep["isp_caps"] / 1e6, sweep["cap_kin"], color=INK, lw=2.2, label="Kinematic ceiling (Π → ∞)")
    a.plot(sweep["isp_caps"] / 1e6, sweep["cap_ach"], color=BLUE, lw=2.2, ls="--", label="Achievable ceiling (Π ≤ 10)")
    a.axvline(BASE.isp_max / 1e6, color=INK2, lw=1, ls=":")
    a.text(BASE.isp_max / 1e6 + 0.1, min(sweep["cap_ach"]) + 0.005, "baseline\ncap", color=INK2, fontsize=9)
    a.set_xlabel(r"Upper bound on specific impulse, $I_{\rm sp}$ (10⁶ s)")
    a.set_ylabel("Cruise speed at the ceiling (c)")
    a.set_title(r"(a) $\alpha_{\rm eng}$ = 10⁻⁸ kg/W", color=INK, fontsize=11, loc="left")
    a.legend(loc="lower right")
    b = ax[1]
    b.semilogx(sweep["alphas"], sweep["alpha_kin"], color=INK, lw=2.2, label="Kinematic ceiling (Π → ∞)")
    b.semilogx(sweep["alphas"], sweep["alpha_ach"], color=BLUE, lw=2.2, ls="--", label="Achievable ceiling (Π ≤ 10)")
    b.axvline(BASE.alpha_eng, color=INK2, lw=1, ls=":")
    b.axhspan(0.3, 1.0, color=INK2, alpha=0.08, lw=0)
    b.text(sweep["alphas"][-1], 0.97, "shaded: β > 0.3, η(β) heuristic extrapolated", color=INK2,
           ha="right", va="top", fontsize=8.5)
    b.set_ylim(0, 1.0)
    b.set_xlabel(r"Engine specific mass $\alpha_{\rm eng}$ (kg/W)")
    b.set_ylabel("Cruise speed at the ceiling (c)")
    b.set_title("(b) exhaust speed optimized freely (u < c)", color=INK, fontsize=11, loc="left")
    b.legend(loc="center right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_tornado(tor, base_beta, path):
    plt = _style()
    rows = sorted(tor, key=lambda r: abs(r["hi_beta"] - r["lo_beta"]))
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(rows) + 1.3))
    for i, r in enumerate(rows):
        for val, col in ((r["lo_beta"], ORANGE), (r["hi_beta"], BLUE)):
            ax.barh(i, val - base_beta, left=base_beta, color=col, height=0.62, edgecolor="white", linewidth=1)
        ax.text(min(r["lo_beta"], r["hi_beta"]) - 0.002, i, r["lo_label"] if r["lo_beta"] < r["hi_beta"] else r["hi_label"],
                ha="right", va="center", fontsize=8, color=INK2)
        ax.text(max(r["lo_beta"], r["hi_beta"]) + 0.002, i, r["hi_label"] if r["hi_beta"] > r["lo_beta"] else r["lo_label"],
                ha="left", va="center", fontsize=8, color=INK2)
    ax.axvline(base_beta, color=INK, lw=1.2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["name"] for r in rows], fontsize=9.5)
    ax.set_xlabel("Achievable ceiling, Π ≤ 10 (cruise speed, c)")
    ax.grid(axis="y", visible=False)
    lo_x = min(min(r["lo_beta"], r["hi_beta"]) for r in rows); hi_x = max(max(r["lo_beta"], r["hi_beta"]) for r in rows)
    ax.set_xlim(lo_x - 0.03, hi_x + 0.03)
    ax.set_title(f"Achievable ceiling: one-at-a-time sensitivity (baseline {base_beta:.4f}c)",
                 color=INK, fontsize=11, loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="lower parameter value"), Patch(color=BLUE, label="higher parameter value")],
              loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ------------------------------------------------------------------------------------
# Full study
# ------------------------------------------------------------------------------------
TORNADO = [  # name, field, low, high, label formatter
    (r"Effective removal energy $E_{\rm rem}$", "e_rem", 3.3e7 * 0.8, 3.3e7 * 1.2, lambda v: f"{v/1e7:.2f}×10⁷"),
    ("ISM density (gas + dust)", "ism", 0.1, 10.0, lambda v: f"×{v:g}"),
    (r"Gas damage multiplier $\chi_g$", "chi_g", 0.05, 0.2, lambda v: f"{v:g}"),
    ("Punch-through prefactor η₀", "eta0", 0.25, 1.0, lambda v: f"{v:g}"),
    (r"Shield safety margin $s_{\rm frac}$", "s_frac", 0.0, 0.2, lambda v: f"{v:g}"),
    (r"Tankage fraction $f_t$", "f_t", 0.03, 0.10, lambda v: f"{v:g}"),
    (r"Engine specific mass $\alpha_{\rm eng}$", "alpha_eng", 1e-9, 1e-7, lambda v: {1e-9: "10⁻⁹", 1e-7: "10⁻⁷"}.get(v, f"{v:.0e}")),
    (r"Burn time $t_{\rm burn}$", "t_burn_yr", 0.5, 2.0, lambda v: f"{v:g} yr"),
    ("Geometry exponent ε", "gamma_geom", 0.5, 1.0, lambda v: f"{v:g}"),
    (r"Reference area $A_{\rm ref}$", "A_ref", 50.0, 200.0, lambda v: f"{v:g} m²"),
    ("Payload mass", "payload", 500.0, 2000.0, lambda v: f"{v:g} kg"),
    (r"Viability limit $\Pi_{\rm max}$", "pi_max", 5.0, 20.0, lambda v: f"{v:g}"),
    (r"$I_{\rm sp}$ upper bound", "isp_max", 2e6, 4e6, lambda v: f"{v/1e6:g}×10⁶ s"),
]


def _with(p, field, v):
    if field == "ism":
        return replace(p, rho_gas=p.rho_gas * v, rho_dust=p.rho_dust * v)
    return replace(p, **{field: v})


def headline_facts(p, ach):
    """Every illustrative number quoted in the paper's prose, computed here."""
    v_voy = 17.0e3
    dime = 17.91e-3; neptune_diam = 2 * 30.07 * AU
    E_ug = gm1(0.2) * 1e-9 * C**2
    grenade = 0.184 * 1.33 * TNT_J_PER_KG          # M67: 184 g Comp B, RE ~1.33
    d = ach["design"]
    swept_dust = p.rho_dust * p.L * d["area"]
    swept_gas = p.rho_gas * p.L * d["area"]
    E_dust = swept_dust * gm1(ach["beta"]) * C**2
    drag = (swept_dust + swept_gas) / d["Mf"]       # fractional momentum loss, upper bound
    u = ach["isp"] * G0
    # one-year boost at constant proper mass-loss rate, integrated in the lab frame
    tau = np.linspace(0.0, p.t_b, 200001); Rr = ach["R"]
    M_tau = Rr - (Rr - 1.0) * tau / p.t_b                     # in units of Mf
    b_tau = np.tanh((u / C) * np.log(Rr / M_tau)); g_tau = 1.0 / np.sqrt(1.0 - b_tau**2)
    t_boost = np.trapezoid(g_tau, tau); x_boost = np.trapezoid(g_tau * b_tau * C, tau)
    T_total = (t_boost + (p.L - x_boost) / (ach["beta"] * C)) / YEAR
    return dict(
        boost_lab_time_yr=t_boost / YEAR, boost_distance_ly=x_boost / LY,
        trip_time_incl_boost_yr=T_total, boost_added_yr=T_total - ach["T"],
        deposited_power_W_m2=d["q_dep"] / (ach["T"] * YEAR),
        voyager_years=p.L / v_voy / YEAR,
        dime_scaled_distance_m=p.L * dime / neptune_diam,
        microgram_02c_MJ=E_ug / 1e6, grenade_MJ=grenade / 1e6, microgram_over_grenade=E_ug / grenade,
        swept_dust_mg=swept_dust * 1e6, swept_gas_mg=swept_gas * 1e6, dust_energy_GJ=E_dust / 1e9,
        dust_energy_tTNT=E_dust / TNT_J_PER_KG / 1e3, area_m2=d["area"],
        shield_thickness_mm=d["thickness_mm"], shield_kg=d["shield"], q_dep_J_m2=d["q_dep"],
        sigma_kg_m2=d["sigma"], ram_drag_momentum_fraction=drag,
        jet_power_rel_correction=jet_energy_per_mass(u) / (0.5 * u * u) - 1.0,
        gamma_at_ceiling=1 + gm1(ach["beta"]), ship_proper_time_yr=ach["tau"],
        Lambda_at_ceiling=d["Lam"], M0_kg=d["M0"], Mf_kg=d["Mf"], engine_kg=d["engine"],
        tank_kg=d["tank"], propellant_kg=d["propellant"], k=d["k"], beta_eng=d["beta_eng"],
        jet_power_W=d["engine"] / p.alpha_eng,
        eta_at_ceiling=float(eta(ach["beta"], p)),
        u_max_over_c=p.isp_max * G0 / C)


def run(out_dir, quick=False):
    t0 = time.time()
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    S = dict(seeds=3, pop=40, gen=30) if quick else dict(seeds=10, pop=100, gen=80)
    R = dict(meta=dict(python=platform.python_version(), numpy=np.__version__,
                       params=asdict(BASE), quick=quick))
    import scipy, pymoo
    R["meta"].update(scipy=scipy.__version__, pymoo=pymoo.__version__)

    print("[1/8] self-tests"); R["selftest"] = selftest()

    print("[2/8] exact ceilings")
    strip = lambda c: {k: v for k, v in c.items() if k != "design"}
    kin, ach = ceiling(BASE, None), ceiling(BASE, BASE.pi_max)
    gkin = ceiling(BASE, None, u_max=0.999 * C); gach = ceiling(BASE, BASE.pi_max, u_max=0.999 * C)
    no_eng = ceiling(replace(BASE, alpha_eng=0.0), None)
    R["ceilings"] = dict(kinematic=strip(kin), achievable=strip(ach), global_kinematic=strip(gkin),
                         global_achievable=strip(gach), no_engine_penalty_kinematic=strip(no_eng),
                         model_A_kinematic=strip(ceiling(BASE, None, model="A")),
                         model_B_kinematic=strip(ceiling(BASE, None, model="B")),
                         model_B_achievable=strip(ceiling(BASE, BASE.pi_max, model="B")))
    R["facts"] = headline_facts(BASE, ach)
    R["facts"]["R_kzero_at_cap"] = R_kzero(BASE.isp_max * G0, BASE)
    R["facts"]["R_kzero_massless_engine"] = R_kzero(BASE.isp_max * G0, replace(BASE, alpha_eng=0.0))
    for k_, c in R["ceilings"].items():
        print(f"   {k_:28s} beta={c['beta']:.5f}  T={c['T']:.3f} yr  Isp={c['isp']:.4e}  R={c['R']:.4f}  Pi={c['Pi']}")

    print("[3/8] scenario ceilings")
    scen = dict(base=BASE, needle=replace(BASE, gamma_geom=0.5), linear=replace(BASE, gamma_geom=1.0),
                harsh=_with(BASE, "ism", 10.0), benign=_with(BASE, "ism", 0.1),
                dense3=_with(BASE, "ism", 3.0))   # local-cloud-like density (Section 3.6)
    R["scenarios"] = {k_: dict(kinematic=strip(ceiling(p_, None)), achievable=strip(ceiling(p_, p_.pi_max)))
                      for k_, p_ in scen.items()}
    for k_, v in R["scenarios"].items():
        print(f"   {k_:8s} kin={v['kinematic']['beta']:.4f} ach={v['achievable']['beta']:.4f}")

    print("[4/8] tornado")
    tor = []
    for name, field, lo, hi, fmt in TORNADO:
        b = []
        for v in (lo, hi):
            p_ = _with(BASE, field, v)
            b.append(ceiling(p_, p_.pi_max, n_grid=40)["beta"])
        tor.append(dict(name=name, field=field, lo=lo, hi=hi, lo_label=fmt(lo), hi_label=fmt(hi),
                        lo_beta=b[0], hi_beta=b[1]))
        print(f"   {name:34s} {b[0]:.4f} / {b[1]:.4f}")
    R["tornado"] = tor

    print("[5/8] propulsion sweeps")
    caps = np.linspace(1.0e6, 1.0e7, 31 if not quick else 8)
    sw = dict(isp_caps=caps,
              cap_kin=[ceiling(replace(BASE, isp_max=c), None, n_grid=40)["beta"] for c in caps],
              cap_ach=[ceiling(replace(BASE, isp_max=c), BASE.pi_max, n_grid=40)["beta"] for c in caps])
    alphas = np.logspace(-10, -6, 25 if not quick else 6)
    ak = [ceiling(replace(BASE, alpha_eng=a), None, u_max=0.999 * C, n_grid=60) for a in alphas]
    aa = [ceiling(replace(BASE, alpha_eng=a), BASE.pi_max, u_max=0.999 * C, n_grid=60) for a in alphas]
    sw.update(alphas=alphas, alpha_kin=[c["beta"] for c in ak], alpha_ach=[c["beta"] for c in aa],
              alpha_u_opt=[c["u_over_c"] for c in ak])
    R["propulsion_sweep"] = {k_: np.asarray(v).tolist() for k_, v in sw.items()}

    print("[6/8] Pareto fronts (NSGA-II)")
    fr = {}
    for m in "ABC":
        fr[m] = [pareto_front(replace(BASE, e_rem=e), m, None, **S)[1] for e in (3.3e7, 0.8 * 3.3e7, 1.2 * 3.3e7)]
    sens = {k_: pareto_front(p_, "C", None, **S)[1] for k_, p_ in scen.items() if k_ != "dense3"}
    Xc, Fc = pareto_front(BASE, "C", BASE.pi_max, seeds=20 if not quick else 3, pop=S["pop"], gen=S["gen"])
    R["fronts"] = dict(
        model_min_T={m: float(fr[m][0][:, 0].min()) for m in "ABC"},
        constrained_best_T=float(Fc[:, 0].min()),
        constrained_gap_to_exact=float(Fc[:, 0].min() / ach["T"] - 1.0),
        constrained_points=int(len(Fc)))
    np.savez(os.path.join(out_dir, "fronts.npz"), C_constrained=Fc, X_constrained=Xc,
             **{f"{m}_{i}": fr[m][i] for m in "ABC" for i in range(3)},
             **{f"sens_{k_}": v for k_, v in sens.items()})

    print("[7/8] validation (IGD / HV)")
    R["validation"] = validate(BASE, seeds=30 if not quick else 3, pop=S["pop"], gen=S["gen"],
                               n_grid=160 if not quick else 40)
    print("   ", R["validation"])

    print("[8/8] figures")
    figs = os.path.join(out_dir, "figures")
    betas_front = np.array([design(x[0], x[1], BASE)["beta"] for x in Xc])
    bE, E, TE = elasticity(Fc, betas_front)
    fig_elasticity(bE, E, ach["beta"], os.path.join(figs, "fig_elasticity_nsga_binned.png"))
    import exact_elasticity                                   # exact (deterministic) front -> Figure 1
    R["elasticity_exact"] = exact_elasticity.run(os.path.join(figs, "fig_elasticity_exact.png"))
    first = lambda lvl: float(bE[np.argmax(E > lvl)]) if np.any(E > lvl) else None
    R["elasticity"] = dict(beta_E_gt_1=first(1), beta_E_gt_10=first(10), E_at_fastest_bin=float(E[-1]),
                           beta_fastest_bin=float(bE[-1]), n_bins=int(len(E)))
    fig_models(fr, os.path.join(figs, "fig_models.png"))
    fig_sensitivity(sens, os.path.join(figs, "fig_sensitivity.png"))
    fig_propulsion(sw, os.path.join(figs, "fig_propulsion.png"))
    fig_tornado(tor, ach["beta"], os.path.join(figs, "fig_tornado.png"))

    R["runtime_s"] = time.time() - t0
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(R, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"done in {R['runtime_s']:.0f} s -> {out_dir}/results.json")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    run(a.out, quick=a.quick)
