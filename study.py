#!/usr/bin/env python3
"""
Step 5 -- which cost, and how the key parameters change the behaviour.

    python3 study.py costs      every cost option of env.py, same run, same metrics
    python3 study.py sweep      V, alpha, rho, B_max and the load, one at a time

Both use sim.run on the Python environment (no Open5GS) and write results/study/:
    costs.csv, costs.png
    sweep.csv, sweep_<parameter>.png, sweep_<parameter>_series.csv
"""
import csv
import os
import sys
from multiprocessing import Pool

import sim
from env import COST, DRAIN, EVENTS, GAMMA, LOAD, PHI, PSI

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "study")
SEEDS = (1, 2, 3)
BUSY, PARENT = "sat1.a0", "sat1"       # the node followed in the time-series panels (1st busy period)
ZOOM = (300, 1000)                     # slots shown around that period

SWEEPS = {                             # parameter: values (the default of sim.py / env.py is among them)
    "V": [0.5, 1, 2, 4, 8, 16, 32],
    "alpha": [10, 30, 100, 300, 1000],
    "rho": [0.005, 0.02, 0.05, 0.1, 0.2, 0.25],        # 0.25 = 1/4: the root cannot move at all
    "b_max": [16, 24, 32, 48, 64, 96],
    "load": [0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.5],
}
DEFAULT = {"V": sim.V, "alpha": sim.ALPHA, "rho": sim.RHO, "b_max": sim.B_MAX, "load": 1.0}
LABEL = {"V": "V  (weight of the cost)", "alpha": "α  (step size)", "rho": "ρ  (floor of a routing share)",
         "b_max": "B_max  (largest allocation, jobs per slot)", "load": "load  (1 = 16–48 jobs per slot)"}
METRICS = [("completion", "jobs completed / jobs injected"), ("sojourn", "mean time in system (slots)"),
           ("backlog", "mean backlog, all nodes (jobs)"), ("peak", "largest queue of a busy node (jobs)"),
           ("cost", "mean cost L_t per slot"), ("Z", "sum of the virtual queues Z (mean)"),
           ("viol", "flow violation  max_i [Σg_i]+ / T"), ("rough", "routing change per slot at the root")]


def one(job: dict) -> dict:
    """One run. job = sim.run arguments, plus 'load' (scales the arrivals) and 'series'."""
    args = {k: v for k, v in job.items() if k not in ("load", "series", "tag")}
    load = job.get("load", 1.0)
    slots, log = sim.run(gamma=(round(GAMMA[0] * load), round(GAMMA[1] * load)), **args)
    m = sim.metrics(slots, log)
    out = {k: v for k, v in m.items() if k != "events"}
    out["peak"] = max(e["peak"] for e in m["events"])
    out["shares"] = [e["during"] for e in m["events"]]
    out["react"] = [e["react"] for e in m["events"]]
    out["tag"] = job.get("tag")
    if job.get("series"):
        k = sim.NODES[PARENT].children.index(BUSY)
        out["series"] = {"x": [x[k] for x in log["x"][PARENT]], "B": log["B"][BUSY], "Q": log["Q"][BUSY],
                         "backlog": [sum(log["Q"][i][t] for i in sim.NODES) for t in range(len(slots))]}
    return out


def pool_map(jobs: list) -> list:
    with Pool(os.cpu_count()) as pool:
        return pool.map(one, jobs, chunksize=1)


# ── cost options ─────────────────────────────────────────────────────────────
def costs() -> None:
    options = [{"phi": f, "psi": q, "load": l, "drain": d}
               for l in LOAD for f in PHI for q in PSI for d in DRAIN
               if not (q == "none" and d == "capped")]          # no psi: nothing to drain
    res = pool_map([{"cost": c, "tag": c} for c in options])
    res.sort(key=lambda r: r["sojourn"])
    os.makedirs(OUT, exist_ok=True)
    cols = ["load", "phi", "psi", "drain", "completion", "sojourn", "backlog", "queue_max", "peak",
            "shares", "react", "rough", "over", "viol"]
    with open(os.path.join(OUT, "costs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in res:
            w.writerow([r["tag"][c] if c in r["tag"] else
                        " ".join(f"{v:.2f}" for v in r[c]) if isinstance(r[c], list) else f"{r[c]:.4f}"
                        for c in cols])
    print(f"{'load':5} {'phi':8} {'psi':6} {'drain':6} {'done':>6} {'time':>6} {'backlog':>7} {'Qmax':>5} "
          f"{'busy peak':>9}  {'share in the busy periods':26} {'react':15} {'rough':>7} {'B>cap':>6} {'viol':>6}")
    for r in res:
        c = r["tag"]
        mark = "  <- in use" if c == COST else ""
        print(f"{c['load']:5} {c['phi']:8} {c['psi']:6} {c['drain']:6} {r['completion']:6.3f} "
              f"{r['sojourn']:6.2f} {r['backlog']:7.0f} {r['queue_max']:5.0f} {r['peak']:9.0f}  "
              + " ".join(f"{v:.2f}" for v in r["shares"]) + "         "
              + " ".join(f"{v:3d}" for v in r["react"])
              + f" {r['rough']:7.4f} {r['over']:6.3f} {r['viol']:6.3f}{mark}")

    plot_costs()
    print(f"\n-> {OUT}/costs.csv, costs.png")


def plot_costs() -> None:
    """costs.png from costs.csv: one dot per cost option."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(os.path.join(OUT, "costs.csv"))))
    colors = {"none": "#888888", "linear": "#e0812b", "quad": "#2a5fd0", "cubic": "#2e9e6b"}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for r in rows:
        x, y = float(r["sojourn"]), float(r["peak"])
        ax.plot(x, y, "o" if r["load"] == "own" else "s", ms=7, mew=1.2, mec=colors[r["psi"]],
                mfc=colors[r["psi"]] if r["drain"] == "plain" else "none")
        if all(r[k] == v for k, v in COST.items()):
            ax.annotate("in use", (x, y), xytext=(-40, -28), textcoords="offset points", fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": "#444", "lw": 0.8})
    for q, col in colors.items():
        ax.plot([], [], "o", color=col, label=f"ψ {q}")
    ax.plot([], [], "o", color="#444", label="load: own")
    ax.plot([], [], "s", color="#444", label="load: child")
    ax.plot([], [], "o", mfc="none", mec="#444", label="drain: capped (hollow)")
    ax.set(xlabel="mean time in system (slots)", ylabel="largest queue of a busy node (jobs)",
           xscale="log", yscale="log", title="every cost option: lower left is better")
    ax.set_xticks([4, 5, 6, 8, 10, 20, 40], labels=["4", "5", "6", "8", "10", "20", "40"])
    ax.set_yticks([10, 20, 50, 100, 200, 500], labels=["10", "20", "50", "100", "200", "500"])
    ax.minorticks_off()
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "costs.png"), dpi=140)


# ── parameter sweeps ─────────────────────────────────────────────────────────
def sweep() -> None:
    jobs = [{name: v, "seed": s, "series": s == SEEDS[0], "tag": (name, v)}
            for name, values in SWEEPS.items() for v in values for s in SEEDS]
    res = pool_map(jobs)
    os.makedirs(OUT, exist_ok=True)
    keys = [k for k, _ in METRICS]
    with open(os.path.join(OUT, "sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "seed"] + keys + ["shares", "react"])
        for j, r in zip(jobs, res):
            w.writerow([*r["tag"], j["seed"]] + [f"{r[k]:.5f}" for k in keys]
                       + [" ".join(f"{v:.3f}" for v in r["shares"]), " ".join(str(v) for v in r["react"])])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, values in SWEEPS.items():
        runs = {v: [r for r in res if r["tag"] == (name, v)] for v in values}
        ramp = [plt.cm.Blues(0.3 + 0.7 * k / (len(values) - 1)) for k in range(len(values))]
        fig, axs = plt.subplots(3, 4, figsize=(19, 11.5))

        with open(os.path.join(OUT, f"sweep_{name}_series.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", name, "share_busy_amf", "B_busy_amf", "queue_busy_amf", "backlog"])
            for v in values:
                s = runs[v][0]["series"]
                w.writerows((t + 1, v, f"{s['x'][t]:.4f}", f"{s['B'][t]:.3f}", s["Q"][t], s["backlog"][t])
                            for t in range(len(s["x"])))

        panels = [("x", f"routing share of the busy AMF {BUSY}", ZOOM), ("B", f"allocation B of {BUSY} (jobs per slot)", ZOOM),
                  ("Q", f"physical queue of {BUSY} (jobs)", ZOOM), ("backlog", "backlog of all nodes (jobs), 25-slot mean", None)]
        for ax, (key, title, zoom) in zip(axs[0], panels):
            top = 0.0
            for v, col in zip(values, ramp):
                y = runs[v][0]["series"][key]
                if zoom is None:
                    y = smooth(y)
                a, b = zoom or (sim.WARM, len(y))
                top = max(top, max(y[a:b]))
                ax.plot(range(1, len(y) + 1), y, color=col, lw=1.4,
                        label=f"{v:g}" + (" (default)" if v == DEFAULT[name] else ""))
            ax.set(title=title, xlabel="slot", xlim=zoom or (0, len(y)), ylim=(0, 1.15 * top))
            for a, b, _, _ in EVENTS:
                ax.axvspan(a, b, color="#000", alpha=0.06, lw=0)
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", ncol=len(values) + 1,
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.968), title=f"{name}  (curves of the top row)")

        for ax, (key, title) in zip(axs[1:].flat, METRICS):
            lo = [min(r[key] for r in runs[v]) for v in values]
            hi = [max(r[key] for r in runs[v]) for v in values]
            ax.fill_between(values, lo, hi, color="#2a5fd0", alpha=0.18, lw=0)
            ax.plot(values, [sim.mean(r[key] for r in runs[v]) for v in values], "o-", color="#2a5fd0", lw=1.6, ms=5)
            ax.axvline(DEFAULT[name], color="#999", lw=0.8, ls=":")
            ax.set(title=title, xlabel=LABEL[name], ylim=(0, None))
            if name in ("V", "alpha", "rho"):
                ax.set_xscale("log")
        for ax in axs.flat:
            ax.spines[["top", "right"]].set_visible(False)
            ax.title.set_fontsize(10)
        fig.suptitle(f"effect of {LABEL[name]}   (top: seed {SEEDS[0]}, busy periods shaded;  bottom: mean and "
                     f"range of seeds {', '.join(map(str, SEEDS))};  dotted line = default)", fontsize=11, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.935))
        fig.savefig(os.path.join(OUT, f"sweep_{name}.png"), dpi=120)
        plt.close(fig)

        print(f"\n{name}")
        print(f"  {'value':>7} " + " ".join(f"{k:>10}" for k in keys) + "   share of the busy node in the busy periods")
        for v in values:
            print(f"  {v:7g} " + " ".join(f"{sim.mean(r[k] for r in runs[v]):10.3f}" for k in keys) + "   "
                  + " ".join(f"{sim.mean(r['shares'][e] for r in runs[v]):.2f}" for e in range(len(EVENTS))))
    print(f"\n-> {OUT}")


def smooth(v, w=25):
    out, acc = [], 0.0
    for k, a in enumerate(v):
        acc += a - (v[k - w] if k >= w else 0.0)
        out.append(acc / min(k + 1, w))
    return out


if __name__ == "__main__":
    {"costs": costs, "sweep": sweep}[sys.argv[1] if len(sys.argv) > 1 else "costs"]()
