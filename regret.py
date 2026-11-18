#!/usr/bin/env python3
"""
Step 6 -- HOLD against the paper's benchmark (dynamic regret), and the theory check.

    python3 regret.py            both scenarios at the default parameters, every slot (~15 min)
    python3 regret.py sweep      the "steady" scenario for several V, with alpha = 25 V (~15 min)
    python3 regret.py theory     the "steady" scenario for growing T, with fixed V and alpha and
                                 with the scaling of the paper's corollary (~15 min)

BENCHMARK  the paper's Y*[t]: the allocation that minimises the same cost L_t of slot t,
           under the same constraints (0 <= B <= B_max, x >= rho) and the flow constraint
           in every slot, g_i^t(Y*) <= 0 at every node. L_t is known once slot t is over (it
           depends on that slot's queues, capacities, gamma and lambda), so Y*[t] is computed
           afterwards: a convex problem in the 132 edge allocations, solved with scipy (SLSQP)
           with the gradient env.py reports. It is the strongest dynamic benchmark: it may
           change completely from one slot to the next.
REGRET     R_t = sum over tau <= t of  L_tau(Y[tau]) - L_tau(Y*[tau])
PATH       the benchmark's path lengths, as in the paper's definitions:
           P_B(T) = sum_i sum_t |B*_i[t] - B*_i[t-1]|,   P_x(T) = sum_i sum_t ||x*_i[t] - x*_i[t-1]||_1.
           If they grow like T^delta, the corollary needs V = O(sqrt T), alpha = O(T^(1 - delta/2))
           and then gives R_T = O(T^((1 + delta)/2)); the virtual queues ||Z[T+1]|| = O(T^(3/4)).
SCENARIOS  "current"  the environment of sim.py (random arrivals and jobs, background load,
                      busy periods)
           "steady"   the theory check: 32 jobs per slot, every job s = 2 and p = its mean,
                      no background load, no busy periods. The best allocation then hardly
                      moves, and regret / t and violation / t should both go to 0.

Writes results/regret/: current.csv, steady.csv, regret.png, sweep.csv, sweep.png,
theory.csv, theory.png
"""
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

import env
import sim
from hold import children
from tree import NODES, ROOT

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "regret")
EDGES = [(i, j) for i in NODES for j in children(i)]
SWEEP_V = (1, 2, 4, 8, 16, 32)          # with alpha = 25 V (the default V = 4, alpha = 100 is among them)
SWEEP_EVERY = 4                         # the sweep solves the benchmark in one slot out of 4
THEORY_T = (500, 1000, 2000, 4000)      # horizons of the theory check
THEORY_DELTA = 0.0                      # path budget of the benchmark: T^delta windows (0 = one static allocation)
SAMPLE = 200                            # slots used for the objective of one window of the windowed benchmark


def steady() -> None:
    """Switch env.py to the theory-check scenario (constant arrivals, jobs and capacity)."""
    env.SESSIONS = (2,)
    env.SPREAD = 0.0
    env.P_DISP = (0.2, 0.2)
    env.BACKGROUND = (0.0, 0.0)
    env.P_MEAN.update({"disp": 0.2, **{k: env.p_measured(k, 2) for k in env.MEASURED}})


SCENARIOS = {"current": dict(events=env.EVENTS, gamma=env.GAMMA),
             "steady": dict(events=[], gamma=(32, 32))}


# ── the benchmark of one slot ────────────────────────────────────────────────
def constraints(lam: dict, gamma: float):
    """Linear constraints as rows of A with lower and upper bounds."""
    k = {e: n for n, e in enumerate(EDGES)}
    rows, lo, hi = [], [], []
    for i, n in NODES.items():
        out = np.zeros(len(EDGES))
        for j in children(i):
            out[k[i, j]] = 1.0
        g = -out.copy()                                      # g_i(Y) <= 0
        if n.parent is not None:
            g[k[n.parent, i]] += lam[n.parent, i]
        rows.append(g), lo.append(-np.inf), hi.append(-(gamma if i == ROOT else 0.0))
        rows.append(out), lo.append(0.0), hi.append(sim.B_MAX)          # 0 <= B_i <= B_max
        if len(children(i)) > 1:
            for j in children(i):                            # Y_ij >= rho B_i
                r = -sim.RHO * out
                r[k[i, j]] += 1.0
                rows.append(r), lo.append(0.0), hi.append(np.inf)
    A, lo, hi = np.array(rows), np.array(lo), np.array(hi)
    up, dn = np.isfinite(hi), np.isfinite(lo)
    return [{"type": "ineq", "fun": lambda y: hi[up] - A[up] @ y, "jac": lambda y: -A[up]},
            {"type": "ineq", "fun": lambda y: A[dn] @ y - lo[dn], "jac": lambda y: A[dn]}]


def solve(job: tuple) -> list:
    """Benchmark cost of each slot of a chunk; each solution starts the next (warm start)."""
    cost_options, chunk = job
    e = env.Env(1, cost=cost_options)                        # exactly the cost HOLD was run with
    out, y = [], None
    for t, Y, Q0, cap, gamma, lam in chunk:
        e.lam = lam

        def f(v):
            cost, grad, _ = e.feedback({ed: v[n] for n, ed in enumerate(EDGES)}, Q0, cap, gamma)
            return sum(cost.values()), np.array([grad[ed] for ed in EDGES])
        start = np.array([Y[ed] for ed in EDGES]) if y is None else y
        r = minimize(f, start, jac=True, bounds=[(0.0, sim.B_MAX)] * len(EDGES),
                     constraints=constraints(lam, gamma), method="SLSQP",
                     options={"maxiter": 300, "ftol": 1e-9})
        y = r.x
        out.append((t, r.fun, bool(r.success), r.x))
    return out


def solve_window(job: tuple) -> tuple:
    """Benchmark with a variation budget: ONE allocation for all the slots of a window.

    It minimises the window's total cost and meets the flow constraint in every slot of the
    window. Since Y >= 0, that is: B_root >= the window's largest gamma, and B_j >= (largest
    lambda_ij) * Y_ij below the root. The objective is evaluated on at most SAMPLE slots of the
    window (spread evenly); the returned cost is the exact total over all its slots.
    """
    cost_options, recs = job
    e = env.Env(1, cost=cost_options)
    step = max(1, len(recs) // SAMPLE)
    sample = recs[::step]
    lam_max = {ed: max(r[5][ed] for r in recs) for ed in recs[0][5]}
    gamma_max = max(r[4] for r in recs)

    def total(v, rows):
        Y = {ed: v[n] for n, ed in enumerate(EDGES)}
        c_sum, g_sum = 0.0, np.zeros(len(EDGES))
        for t, _, Q0, cap, gamma, lam in rows:
            e.lam = lam
            cost, grad, _ = e.feedback(Y, Q0, cap, gamma)
            c_sum += sum(cost.values())
            g_sum += np.array([grad[ed] for ed in EDGES])
        return c_sum, g_sum

    def f(v):
        c, g = total(v, sample)
        return c / len(sample), g / len(sample)
    start = np.mean([[r[1][ed] for ed in EDGES] for r in recs], axis=0)
    r = minimize(f, start, jac=True, bounds=[(0.0, sim.B_MAX)] * len(EDGES),
                 constraints=constraints(lam_max, gamma_max), method="SLSQP",
                 options={"maxiter": 400, "ftol": 1e-10})
    return recs[0][0], total(r.x, recs)[0], bool(r.success), r.x


def windowed(rec: list, cost_options: dict, windows: int, pool=None) -> tuple:
    """Total cost of the best allocation sequence that changes at most `windows` times."""
    T = len(rec)
    parts = [rec[k * T // windows:(k + 1) * T // windows] for k in range(windows)]
    jobs = [(cost_options, p) for p in parts if p]
    done = pool.map(solve_window, jobs) if pool else [solve_window(j) for j in jobs]
    return sum(c for _, c, _, _ in done), sum(ok for _, _, ok, _ in done), len(done)


def run(scenario: str, V: float = sim.V, alpha: float = sim.ALPHA, every: int = 1, T: int = sim.T) -> dict:
    """One run of HOLD and the benchmark of its slots. Returns the series per slot."""
    rec, slots = [], []

    def on_slot(t, Y, fb):
        rec.append((t, dict(Y), fb["Q0"], fb["cap"], fb["gamma"], fb["lam"]))
    slots, log = sim.run(T=T, V=V, alpha=alpha, on_slot=on_slot, **SCENARIOS[scenario])
    picked = rec[::every]
    n = os.cpu_count()
    with Pool(n) as pool:
        cost_options = SCENARIOS[scenario].get("cost", env.COST)
        done = pool.map(solve, [(cost_options, picked[k * len(picked) // n:(k + 1) * len(picked) // n])
                                for k in range(n)])
    bench = {t: (c, ok, y) for part in done for t, c, ok, y in part}
    T = len(slots)
    viol = [max(log["viol"][i][t] for i in NODES) for t in range(T)]
    return {"t": [s["t"] for s in slots], "hold": [s["cost"] for s in slots],
            "bench": [bench[t][0] if t in bench else None for t in range(1, T + 1)],
            "ystar": {t: v[2] for t, v in bench.items()}, "every": every,
            "Znorm": float(np.sqrt(sum(log["Z"][i][-1] ** 2 for i in NODES))),
            "ok": sum(v[1] for v in bench.values()), "solved": len(bench), "viol": viol,
            "Z": [sum(log["Z"][i][t] for i in NODES) for t in range(T)],
            "backlog": [sum(log["Q"][i][t] for i in NODES) for t in range(T)]}


def path_lengths(r: dict) -> tuple:
    """Cumulative P_B(t) and P_x(t) of the benchmark (needs every slot solved)."""
    kids = {i: [EDGES.index((i, j)) for j in children(i)] for i in NODES}
    PB, Px, accB, accx, prev = [], [], 0.0, 0.0, None
    for t in sorted(r["ystar"]):
        y = r["ystar"][t]
        B = {i: sum(y[k] for k in ks) for i, ks in kids.items()}
        x = {i: [y[k] / B[i] if B[i] > 1e-9 else 1 / len(ks) for k in ks] for i, ks in kids.items()}
        if prev is not None:
            accB += sum(abs(B[i] - prev[0][i]) for i in NODES)
            accx += sum(sum(abs(a - b) for a, b in zip(x[i], prev[1][i])) for i in NODES)
        PB.append(accB)
        Px.append(accx)
        prev = (B, x)
    return PB, Px


def regret_curve(r: dict) -> list:
    acc, out = 0.0, []
    for h, b in zip(r["hold"], r["bench"]):
        acc += h - b
        out.append(acc)
    return out


def save(name: str, r: dict) -> None:
    R = regret_curve(r)
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "cost_hold", "cost_benchmark", "regret", "max_violation", "sum_Z", "backlog"])
        for k, t in enumerate(r["t"]):
            w.writerow([t, f"{r['hold'][k]:.3f}", f"{r['bench'][k]:.3f}", f"{R[k]:.1f}",
                        f"{r['viol'][k]:.2f}", f"{r['Z'][k]:.1f}", r["backlog"][k]])


def report(name: str, r: dict) -> None:
    R, T, W = regret_curve(r), len(r["t"]), sim.WARM
    gap = [h - b for h, b in zip(r["hold"], r["bench"])]
    print(f"\n{name}: {r['solved']} slots solved, {r['solved'] - r['ok']} not converged")
    print(f"  mean cost per slot after the warm-up: HOLD {np.mean(r['hold'][W:]):.1f}   benchmark "
          f"{np.mean(r['bench'][W:]):.1f}   ratio {np.mean(r['hold'][W:]) / np.mean(r['bench'][W:]):.2f}")
    print(f"  {'t':>6} {'regret R_t':>11} {'R_t / t':>8} {'gap per slot (last 200)':>24} {'max_i [Σg]+ / t':>16}")
    for t in (200, 400, 800, 1600, T):
        print(f"  {t:6d} {R[t - 1]:11.0f} {R[t - 1] / t:8.2f} {np.mean(gap[t - 200:t]):24.2f} {r['viol'][t - 1] / t:16.4f}")
    PB, Px = path_lengths(r)
    print(f"  benchmark path length  {'t':>6} {'P_B(t)':>10} {'P_x(t)':>9}   (growth from t/2 to t gives delta)")
    for t in (T // 4, T // 2, T):
        d = (f"   delta_B {np.log2(PB[t - 1] / PB[t // 2 - 1]):.2f}  delta_x {np.log2(Px[t - 1] / Px[t // 2 - 1]):.2f}"
             if t > T // 4 else "")
        print(f"  {'':22} {t:6d} {PB[t - 1]:10.0f} {Px[t - 1]:9.1f}{d}")


def plot(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W = sim.WARM

    def smooth(v, w=25):
        out, acc = [], 0.0
        for k, a in enumerate(v):
            acc += a - (v[k - w] if k >= w else 0.0)
            out.append(acc / min(k + 1, w))
        return out

    fig, axs = plt.subplots(4, len(results), figsize=(8 * len(results), 12), squeeze=False)
    for col, (name, r) in enumerate(results.items()):
        t, R = r["t"], regret_curve(r)
        ax = axs[0, col]
        h, b = smooth(r["hold"]), smooth(r["bench"])
        ax.plot(t, h, color="#2a5fd0", lw=1.4, label="HOLD")
        ax.plot(t, b, color="#222", lw=1.2, ls="--", label="benchmark Y* (best feasible allocation of each slot)")
        ax.set(title=f"{name}: cost L_t per slot (25-slot mean)", ylabel="cost", ylim=(0, 1.3 * max(h[W:])))
        ax.legend(frameon=False, fontsize=8)
        ax = axs[1, col]
        ax.plot(t, R, color="#2a5fd0", lw=1.6)
        ax.set(title=f"{name}: dynamic regret R_t = Σ [L(Y) − L(Y*)]", ylabel="regret")
        ax = axs[2, col]
        ax.plot(t, [x / k for k, x in enumerate(R, 1)], color="#2a5fd0", lw=1.6, label="R_t / t")
        ax.plot(t, smooth([x - y for x, y in zip(r["hold"], r["bench"])], 200), color="#e0812b", lw=1.2,
                label="gap per slot (200-slot mean)")
        ax.set(title=f"{name}: regret per slot", ylabel="cost per slot", ylim=(0, 1.3 * max(R[W:]) / W))
        ax.legend(frameon=False, fontsize=8)
        ax = axs[3, col]
        ax.plot(t, [v / k for k, v in enumerate(r["viol"], 1)], color="#2e9e6b", lw=1.6)
        ax.set(title=f"{name}: flow violation  max_i [Σ g_i]+ / t  (benchmark: 0)", ylabel="jobs per slot",
               xlabel="slot", ylim=(0, 0.1))
        for ax in axs[:, col]:
            ax.set_xlim(0, len(t))
            ax.spines[["top", "right"]].set_visible(False)
            ax.title.set_fontsize(10)
            for a, b_, _, _ in SCENARIOS[name]["events"]:
                ax.axvspan(a, b_, color="#000", alpha=0.06, lw=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "regret.png"), dpi=120)


def sweep() -> None:
    steady()
    rows = []
    for V in SWEEP_V:
        r = run("steady", V=V, alpha=25.0 * V, every=SWEEP_EVERY)
        T, W = len(r["t"]), sim.WARM
        pairs = [(h, b) for h, b in zip(r["hold"][W:], r["bench"][W:]) if b is not None]
        rows.append({"V": V, "alpha": 25.0 * V, "cost": np.mean([h for h, _ in pairs]),
                     "benchmark": np.mean([b for _, b in pairs]),
                     "gap": np.mean([h - b for h, b in pairs]), "violation": r["viol"][-1] / T,
                     "Z": np.mean(r["Z"][W:]), "backlog": np.mean(r["backlog"][W:])})
        print(f"  V {V:4g}  alpha {25 * V:5g}: cost {rows[-1]['cost']:7.1f}  benchmark {rows[-1]['benchmark']:7.1f}  "
              f"gap {rows[-1]['gap']:6.2f}  violation/T {rows[-1]['violation']:.4f}  mean ΣZ {rows[-1]['Z']:7.1f}  "
              f"backlog {rows[-1]['backlog']:.0f}")
    with open(os.path.join(OUT, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Vs = [r["V"] for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axs, ("gap", "violation", "Z"),
                              ("regret per slot (cost of HOLD − benchmark)", "flow violation  max_i [Σ g_i]+ / T",
                               "sum of the virtual queues Z (mean)")):
        ax.plot(Vs, [r[key] for r in rows], "o-", color="#2a5fd0", lw=1.6)
        ax.set(xscale="log", xlabel="V  (α = 25 V)", title=title, ylim=(0, None))
        ax.set_xticks(Vs, labels=[f"{v:g}" for v in Vs])
        ax.minorticks_off()
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("steady scenario: the trade-off of V", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sweep.png"), dpi=120)


def theory_run(job: tuple) -> dict:
    """One horizon and one parameter setting of the theory check (runs in a worker)."""
    T, name, V, alpha = job
    steady()
    rec = []
    slots, log = sim.run(T=T, V=V, alpha=alpha, events=[], gamma=(32, 32),
                         on_slot=lambda t, Y, fb: rec.append((t, dict(Y), fb["Q0"], fb["cap"], fb["gamma"], fb["lam"])))
    windows = max(1, round(T ** THEORY_DELTA)) if THEORY_DELTA > 0 else 1
    bench, ok, n = windowed(rec, env.COST, windows)
    hold_total = sum(s["cost"] for s in slots)
    Znorm = float(np.sqrt(sum(log["Z"][i][-1] ** 2 for i in NODES)))
    viol = max(log["viol"][i][-1] for i in NODES)
    return {"setting": name, "T": T, "V": V, "alpha": alpha, "windows": windows, "converged": f"{ok}/{n}",
            "regret": hold_total - bench, "regret_per_slot": (hold_total - bench) / T,
            "hold_per_slot": hold_total / T, "benchmark_per_slot": bench / T,
            "violation_per_slot": viol / T, "Z_norm": Znorm, "Z_norm_over_T34": Znorm / T ** 0.75}


def theory() -> None:
    """The corollary: with a benchmark of path length O(T^delta), V = sqrt T and
    alpha = T^(1 - delta/2), R_T / T should fall like T^(-(1 - delta)/2); with V and alpha fixed
    the theorem allows a constant gap per slot."""
    jobs = [(T, name, V, alpha) for T in THEORY_T
            for name, V, alpha in (("fixed", sim.V, sim.ALPHA),
                                   ("corollary", T ** 0.5, T ** (1 - THEORY_DELTA / 2)))]
    with Pool(os.cpu_count()) as pool:
        rows = pool.map(theory_run, sorted(jobs, key=lambda j: -j[0]), chunksize=1)
    rows.sort(key=lambda r: (r["setting"], r["T"]))
    for r in rows:
        print(f"  {r['setting']:9} T {r['T']:5d}  V {r['V']:6.1f}  alpha {r['alpha']:7.1f}  benchmark windows {r['windows']:3d} "
              f"({r['converged']} converged):  cost per slot HOLD {r['hold_per_slot']:7.1f}  benchmark {r['benchmark_per_slot']:7.1f}  "
              f"R_T/T {r['regret_per_slot']:7.2f}  violation/T {r['violation_per_slot']:.4f}  ||Z[T]|| {r['Z_norm']:7.1f}  "
              f"||Z||/T^0.75 {r['Z_norm_over_T34']:.3f}")
    with open(os.path.join(OUT, "theory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, color in (("fixed", "#e0812b"), ("corollary", "#2a5fd0")):
        rs = [r for r in rows if r["setting"] == name]
        Ts = [r["T"] for r in rs]
        label = "V = 4, α = 100 (fixed)" if name == "fixed" else f"V = √T, α = T^(1−δ/2), δ = {THEORY_DELTA}"
        for ax, key in zip(axs, ("regret_per_slot", "violation_per_slot", "Z_norm")):
            ax.plot(Ts, [r[key] for r in rs], "o-", color=color, lw=1.6, label=label)
    cor = [r for r in rows if r["setting"] == "corollary"]
    c0 = cor[0]["Z_norm"] / cor[0]["T"] ** 0.75
    axs[2].plot([r["T"] for r in cor], [c0 * r["T"] ** 0.75 for r in cor], ":", color="#888",
                label="∝ T^(3/4) (the paper's bound)")
    for ax, title in zip(axs, ("regret per slot  R_T / T  (benchmark: best allocation with the variation budget)",
                               "flow violation  max_i [Σ g_i]+ / T", "virtual queues  ||Z[T]||")):
        ax.set(xscale="log", xlabel="horizon T", title=title)
        ax.set_xticks(THEORY_T, labels=[str(T) for T in THEORY_T])
        ax.minorticks_off()
        ax.spines[["top", "right"]].set_visible(False)
        ax.title.set_fontsize(9)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("steady scenario: the paper's corollary", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "theory.png"), dpi=120)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep()
        print(f"\n-> {OUT}/sweep.csv, sweep.png")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "theory":
        theory()
        print(f"\n-> {OUT}/theory.csv, theory.png")
        return
    results = {"current": run("current")}
    steady()
    results["steady"] = run("steady")
    for name, r in results.items():
        save(name, r)
        report(name, r)
    plot(results)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
