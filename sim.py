#!/usr/bin/env python3
"""
Step 3 check -- run Algorithm 1 on the Python environment (no Open5GS yet).

    python3 sim.py

Writes results/sim/slots.csv, nodes.csv, overview.png, events.png, dynamics.png and
prints the checks.
events.png    each scripted busy period: capacity, routing shares x, allocation B,
              physical queue and cost of the busy node and its siblings.
dynamics.png  the whole run for two groups of siblings (the four platforms; the
              three SMFs of one satellite AMF): capacity, x, B, queue, queueing
              time, processing time and cost.
response.png  what happens, on average, around the moments a node's queue grows
              above its siblings' (outside the scripted busy periods): its queue,
              its routing share, its B, its queueing time and its cost.
"""
import csv
import os

from env import COST, EVENTS, GAMMA, Env, busy
from hold import SINK, Hold, children
from tree import NODES, PLATFORMS, ROOT, of_kind

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sim")

T = 3200
V = 4.0            # weight of the cost against the virtual queues
ALPHA = 100.0      # step size: larger = smaller moves of x and B per slot
RHO = 0.02         # floor on every routing share
B_MAX = 64.0       # jobs per slot
SEED = 1
WARM = 150         # B starts at 0, so the first slots are a cold start
BUSY = [e[2].rstrip("*") for e in EVENTS]      # the nodes of the scripted busy periods


def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else 0.0


def run(T=T, V=V, alpha=ALPHA, rho=RHO, b_max=B_MAX, seed=SEED, events=EVENTS, cost=COST,
        gamma=GAMMA, on_done=None, on_slot=None):
    """on_done(t, jobs): called with the jobs that left their leaf in slot t (testbed.py).
    on_slot(t, Y, fb): called with the allocation and the telemetry of slot t (regret.py)."""
    env, alg = Env(seed, events, cost, gamma), Hold(V, alpha, rho, b_max)
    log = {k: {i: [] for i in NODES} for k in ("B", "Z", "Q", "util", "used", "got", "cap", "cost", "budget", "n", "wait", "proc", "x", "viol", "arr")}
    slots, gsum = [], {i: 0.0 for i in NODES}
    for t in range(1, T + 1):
        Y = alg.Y()
        fb = env.slot(t, Y)
        if on_done:
            on_done(t, fb["done"])
        if on_slot:
            on_slot(t, Y, fb)
        for i, n in NODES.items():                     # flow constraint of the applied Y[t]
            g = (fb["gamma"] if i == ROOT else fb["lam"][n.parent, i] * Y[n.parent, i]) - alg.B[i]
            gsum[i] += g
            log["viol"][i].append(max(gsum[i], 0.0))
            log["arr"][i].append(g + alg.B[i])
            log["B"][i].append(alg.B[i])
            log["Z"][i].append(alg.Z[i])
            log["Q"][i].append(fb["Q"][i])
            log["util"][i].append(fb["util"][i])       # allocation B_i / cap_i
            log["used"][i].append(fb["used"][i])       # share of its CPU capacity really spent
            log["got"][i].append(fb["got"][i])         # tasks that really arrived (0 at the root)
            log["cap"][i].append(fb["cap"][i])         # jobs the node can process in this slot
            log["cost"][i].append(fb["cost"][i])       # its part of L_t
            log["budget"][i].append(fb["budget"][i])   # CPU cycles (Mc) it has in this slot
            log["n"][i].append(fb["timing"][i][0])     # jobs it finished in this slot,
            log["wait"][i].append(fb["timing"][i][1])  # their queueing time (slots, summed)
            log["proc"][i].append(fb["timing"][i][2])  # and processing time (slots, summed)
            log["x"][i].append(list(alg.x[i]))
        slots.append({"t": t, "gamma": fb["gamma"], "done": len(fb["done"]),
                      "cost": sum(fb["cost"].values()), "waiting": fb["waiting"],
                      "sojourn": mean(t - j.born for j in fb["done"]),          # means over the jobs
                      "queueing": mean(j.wait for j in fb["done"]),            # that left in slot t
                      "processing": mean(j.proc for j in fb["done"])})
        alg.step(fb["grad"], fb["lam"], fb["gamma"])
    return slots, log


def share(log, parent, child, a, b):
    """Mean routing share parent -> child over slots a..b."""
    k = NODES[parent].children.index(child)
    return mean(x[k] for x in log["x"][parent][a - 1:b])


def metrics(slots, log, events=EVENTS) -> dict:
    """The numbers every run is judged by. They do not depend on which cost is used."""
    T, W = len(slots), min(WARM, len(slots) // 2)
    late = slots[W:]
    m = {"completion": sum(s["done"] for s in slots) / sum(s["gamma"] for s in slots),
         "sojourn": sum(s["sojourn"] * s["done"] for s in late) / max(sum(s["done"] for s in late), 1),
         "queueing": sum(s["queueing"] * s["done"] for s in late) / max(sum(s["done"] for s in late), 1),
         "processing": sum(s["processing"] * s["done"] for s in late) / max(sum(s["done"] for s in late), 1),
         "backlog": mean(sum(log["Q"][i][t] for i in NODES) for t in range(W, T)),
         "queue_max": max(max(log["Q"][i][W:]) for i in NODES),
         "cost": mean(s["cost"] for s in late),
         "viol": max(log["viol"][i][-1] for i in NODES) / T,
         "Z": mean(sum(log["Z"][i][t] for i in NODES) for t in range(W, T)),
         "rough": mean(sum(abs(a - b) for a, b in zip(log["x"][ROOT][t], log["x"][ROOT][t - 1]))
                       for t in range(W, T)),
         "over": mean(mean(u > 1 for u in log["util"][i][W:]) for i in NODES),
         "events": []}
    for a, b, node, frac in events:
        if b > T:
            continue
        child = node.rstrip("*")
        parent = NODES[child].parent
        k = NODES[parent].children.index(child)
        x = [v[k] for v in log["x"][parent]]
        before, during = mean(x[max(0, a - 201):a - 1]), mean(x[a + 99:b])
        after = mean(x[b + 199:min(T, b + 400)])
        half = (before + during) / 2
        m["events"].append({
            "node": child, "lost": frac, "before": before, "during": during, "after": after,
            "react": next((t - a for t in range(a, b) if x[t] <= half), b - a),   # slots to half the move
            "peak": max(log["Q"][child][a - 1:b]), "end": log["Q"][child][b - 1]})
    return m


def check(slots, log):
    T = len(slots)
    m = metrics(slots, log)
    jobs_in, jobs_done = sum(s["gamma"] for s in slots), sum(s["done"] for s in slots)
    print(f"slots {T}   jobs in {jobs_in}   jobs done {jobs_done} ({m['completion']:.1%})   "
          f"still waiting at the end {slots[-1]['waiting']}"
          f"   -> jobs conserved: {'yes' if jobs_in == jobs_done + slots[-1]['waiting'] else 'NO'}")
    print(f"mean cost per slot {m['cost']:.2f}   mean backlog {m['backlog']:.0f}   mean time in system "
          f"{m['sojourn']:.2f} slots = 4 (one slot per stage after the root) + {m['queueing']:.2f} of queueing")

    print("\ntime at each stage (per job; processing = share of a slot the node needs for the job)"
          "   and CPU capacity (Mc per slot) outside the scripted busy periods")
    print(f"  {'stage':5} {'queueing (slots)':>16} {'processing (slots)':>18}   {'satellites: min mean max':>26}   "
          f"{'TN: min mean max':>20}")
    for kind in ("root", "disp", "amf", "smf", "upf"):
        ids = [x.id for x in of_kind(kind)]
        n = sum(sum(log["n"][i][WARM:]) for i in ids)
        line = (f"  {kind:5} {sum(sum(log['wait'][i][WARM:]) for i in ids) / n:16.3f} "
                f"{sum(sum(log['proc'][i][WARM:]) for i in ids) / n:18.3f}   ")
        for tn in (False, True):
            b = [v for x in of_kind(kind) if (x.n == 4) == tn
                 for t, v in enumerate(log["budget"][x.id], 1) if t > WARM and not busy(EVENTS, x.id, t)]
            line += (f"{min(b):9.2f} {mean(b):6.2f} {max(b):6.2f}      " if b else " " * 28)
        print(line)

    print("\ncapacity  (CPU really spent in a slot / CPU the node has in that slot; must never exceed 1)")
    print(f"  {'stage':5} {'mean':>6} {'max':>6}   {'slots at full capacity':>22}   {'slots with B > cap':>18}")
    for kind in ("root", "disp", "amf", "smf", "upf"):
        ids = [x.id for x in of_kind(kind)]
        u = [v for i in ids for v in log["used"][i][WARM:]]
        print(f"  {kind:5} {mean(u):6.2f} {max(u):6.2f}   {mean(v > 0.999 for v in u):22.1%}   "
              f"{mean(mean(v > 1 for v in log['util'][i][WARM:]) for i in ids):18.1%}")

    worst = 0.0                                    # do the real jobs follow the routing vectors?
    for i, n in NODES.items():
        got = [sum(log["got"][j][WARM:]) for j in n.children]
        for k in range(len(got)):
            worst = max(worst, abs(got[k] / max(sum(got), 1) - mean(x[k] for x in log["x"][i][WARM:])))
    print(f"\nrouting  largest gap between a child's real share of the jobs and its share x: {worst:.3f}")

    print("\nflow constraint  [sum_t g_i]+ / T  (must go to 0)   and queues (mean over the 2nd half)")
    print(f"  {'stage':5} {'viol/T at T/4':>14} {'at T':>8} {'max Z':>8} {'mean Q':>8} {'max Q':>8}")
    for kind in ("root", "disp", "amf", "smf", "upf"):
        ids = [x.id for x in of_kind(kind)]
        q4 = max(log["viol"][i][T // 4 - 1] for i in ids) / (T // 4)
        qT = max(log["viol"][i][-1] for i in ids) / T
        print(f"  {kind:5} {q4:14.3f} {qT:8.3f} {max(max(log['Z'][i]) for i in ids):8.1f} "
              f"{mean(mean(log['Q'][i][T // 2:]) for i in ids):8.2f} "
              f"{max(max(log['Q'][i][WARM:]) for i in ids):8.0f}")

    print("\nresponse to congestion  (a child's queue rises into the top 5% above its siblings', outside "
          "the scripted busy periods)")
    print(f"  {'parent':6} {'episodes':>8}   {'queue above siblings: at onset, +5, +10, +40 slots':>52}   "
          f"{'its share: 20 slots before, at onset, lowest after (at slot), +40':>66}")
    for kind in ("root", "disp", "amf", "smf"):
        e = episodes(log, kind)
        ex = [a - b for a, b in zip(e["Q"], e["Q_sib"])]
        at = lambda k: k - LAGS[0]
        low = min(range(0, 41), key=lambda k: e["x"][at(k)])
        print(f"  {kind:6} {e['n']:8d}   {ex[at(0)]:+18.2f} {ex[at(5)]:+7.2f} {ex[at(10)]:+7.2f} {ex[at(40)]:+7.2f}"
              f"{'':17}{e['x'][0]:14.3f} {e['x'][at(0)]:8.3f} {e['x'][at(low)]:8.3f} ({low:+d}) {e['x'][at(40)]:8.3f}")

    print("\ntraffic shift during each busy period (routing share of the busy child)")
    for (a, b, node, frac), e in zip(EVENTS, m["events"]):
        parent = NODES[e["node"]].parent
        spent = mean(log["used"][e["node"]][a + 99:b])
        real = sum(log["got"][e["node"]][a + 99:b]) / sum(sum(log["got"][j][a + 99:b]) for j in NODES[parent].children)
        print(f"  {e['node']:10} loses {frac:.0%} of its capacity, slots {a}-{b}: share at {parent} "
              f"{e['before']:.2f} -> {e['during']:.2f} (real jobs {real:.2f}) -> {e['after']:.2f} after;  half of the shift after "
              f"{e['react']} slots;  its queue peaks at {e['peak']}, ends at {e['end']};  "
              f"it uses {spent:.0%} of the capacity it has left")
    return m


LAGS = (-20, 60)        # slots shown before and after the start of a congestion episode


def episodes(log, kind: str) -> dict:
    """Average behaviour around the slots where a child of a `kind` node becomes congested.

    A child is congested when its queue exceeds the mean queue of its siblings by more than
    95% of all such differences do. An episode starts when that happens for the first time in
    30 slots. Scripted busy periods on the same platform are left out. Returns, for every lag
    in LAGS, the mean over the episodes of the child's queue, share, B, cost and queueing time,
    and the same for the mean of its siblings.
    """
    T = len(log["Q"][ROOT])

    def scripted(c, t):
        return any(a - 30 <= t <= b + 100 and (kind == "root" or node.split(".")[0].rstrip("*") == c.split(".")[0])
                   for a, b, node, _ in EVENTS)

    groups = [(x.id, x.children) for x in of_kind(kind)]
    excess = {c: [log["Q"][c][t] - sum(log["Q"][j][t] for j in kids) / len(kids) for t in range(T)]
              for _, kids in groups for c in kids}
    pool = sorted(v for c, e in excess.items() for t, v in enumerate(e, 1) if t > WARM and not scripted(c, t))
    limit = pool[int(0.95 * len(pool))]
    lags = range(LAGS[0], LAGS[1] + 1)
    keys = ("Q", "B", "cost", "wait", "n")
    acc = {k: [0.0] * len(lags) for k in keys + tuple(k + "_sib" for k in keys) + ("x",)}
    count = 0
    for parent, kids in groups:
        for k, c in enumerate(kids):
            last = -10 ** 9
            for t in range(WARM - LAGS[0], T - LAGS[1]):                   # t = slot number - 1
                if excess[c][t] > limit >= excess[c][t - 1] and t - last > 30 and not scripted(c, t + 1):
                    last = t
                    count += 1
                    for m, lag in enumerate(lags):
                        acc["x"][m] += log["x"][parent][t + lag][k]
                        for key in keys:
                            acc[key][m] += log[key][c][t + lag]
                            acc[key + "_sib"][m] += (sum(log[key][j][t + lag] for j in kids if j != c)
                                                     / (len(kids) - 1))
    out = {k: [v / max(count, 1) for v in vals] for k, vals in acc.items()}
    for side in ("", "_sib"):                                              # queueing time per finished job
        out["delay" + side] = [w / n if n else 0.0 for w, n in zip(out["wait" + side], out["n" + side])]
    out["n"] = count
    return out


def save(slots, log, out=OUT):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "slots.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(slots[0]))
        w.writeheader()
        w.writerows(slots)
    with open(os.path.join(out, "nodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "node", "B", "arrivals", "Z", "Q", "util", "cap", "cost", "x"])
        for i in NODES:
            for t in range(len(slots)):
                w.writerow([t + 1, i, f"{log['B'][i][t]:.3f}", f"{log['arr'][i][t]:.3f}",
                            f"{log['Z'][i][t]:.3f}", log["Q"][i][t], f"{log['util'][i][t]:.3f}",
                            f"{log['cap'][i][t]:.3f}", f"{log['cost'][i][t]:.3f}",
                            " ".join(f"{v:.3f}" for v in log["x"][i][t])])


def plot(slots, log, out=OUT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = len(slots)
    ts = range(1, T + 1)
    colors = ["#2a5fd0", "#e0812b", "#2e9e6b", "#b0438a"]

    def smooth(v, w=25):
        out, acc = [], 0.0
        for k, a in enumerate(v):
            acc += a - (v[k - w] if k >= w else 0.0)
            out.append(acc / min(k + 1, w))
        return out

    def shade(ax):
        for a, b, node, _ in EVENTS:
            ax.axvspan(a, b, color="#000", alpha=0.06, lw=0)

    def routing(ax, parent, title):
        for k, c in enumerate(NODES[parent].children):
            ax.plot(ts, [x[k] for x in log["x"][parent]], color=colors[k], lw=1.5,
                    label=c.rsplit(".", 1)[-1] if "." in c else c)
        ax.set(title=title, ylim=(0, 1), ylabel="routing share x")
        ax.legend(frameon=False, ncol=4, fontsize=7, loc="upper right")

    fig, axs = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    routing(axs[0, 0], ROOT, "root -> platforms")
    routing(axs[0, 1], "sat1", "dispatcher sat1 -> its AMFs  (a0 busy in the 1st period)")
    routing(axs[1, 0], "sat2.a0", "AMF sat2.a0 -> its SMFs  (s1 busy in the 2nd period)")
    ax = axs[1, 1]
    for k, node in enumerate(BUSY):
        ax.plot(ts, log["B"][node], color=colors[k], lw=1.2, label=node)
    ax.set(title="allocation B of the four nodes that get busy  (one per period)", ylabel="jobs per slot",
           ylim=(0, 1.3 * max(max(log["B"][n][WARM:]) for n in BUSY)))
    ax.legend(frameon=False, ncol=4, fontsize=7)

    ax = axs[2, 0]
    for k, node in enumerate(BUSY):
        ax.plot(ts, log["Q"][node], color=colors[k], lw=1.2, label=node)
    ax.set(title="physical queue of the four busy nodes  (cold start clipped)",
           ylabel="jobs waiting",
           ylim=(0, 1.4 * max(max(log["Q"][n][WARM:]) for n in BUSY)))
    ax.legend(frameon=False, ncol=4, fontsize=7)

    ax = axs[2, 1]
    for k, kind in enumerate(("disp", "amf", "smf", "upf")):
        ids = [x.id for x in of_kind(kind)]
        ax.plot(ts, [sum(log["Q"][i][t] for i in ids) for t in range(T)], color=colors[k], lw=1.2,
                label=kind)
    ax.plot(ts, log["Q"][ROOT], color="#555", lw=1.2, label="root")
    ax.set(title="physical queues, total per stage  (cold start clipped)", ylabel="jobs waiting",
           ylim=(0, 1.4 * max(sum(log["Q"][x.id][t] for x in of_kind("upf")) for t in range(WARM, T))))
    ax.legend(frameon=False, ncol=5, fontsize=7)

    ax = axs[3, 0]
    for k, kind in enumerate(("disp", "amf", "smf", "upf")):
        ids = [x.id for x in of_kind(kind)]
        ax.plot(ts, [max(log["viol"][i][t] for i in ids) / (t + 1) for t in range(T)],
                color=colors[k], lw=1.2, label=kind)
    ax.plot(ts, [log["viol"][ROOT][t] / (t + 1) for t in range(T)], color="#555", lw=1.2, label="root")
    ax.set(title="flow-constraint violation  max_i [sum g_i]+ / t   (must go to 0)",
           ylabel="jobs per slot", xlabel="slot", ylim=(0, 2))
    ax.legend(frameon=False, ncol=5, fontsize=7)

    ax = axs[3, 1]
    ax.plot(ts, smooth([s["cost"] for s in slots]), color="#2a5fd0", lw=1.5)
    ax.set(title="cost L_t per slot, 25-slot mean  (cold start clipped)", ylabel="cost",
           xlabel="slot", ylim=(0, 1.4 * max(smooth([s["cost"] for s in slots])[WARM:])))

    for ax in axs.flat:
        shade(ax)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, T)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "overview.png"), dpi=130)


def plot_events(slots, log, out=OUT):
    """One row per busy period: what the busy node and its siblings do before, during and after."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = len(slots)
    events = [e for e in EVENTS if e[1] <= T]
    colors = ["#2a5fd0", "#e0812b", "#2e9e6b", "#b0438a"]

    def smooth(v, w):
        out_, acc = [], 0.0
        for k, a in enumerate(v):
            acc += a - (v[k - w] if k >= w else 0.0)
            out_.append(acc / min(k + 1, w))
        return out_

    fig, axs = plt.subplots(len(events), 5, figsize=(24, 4.1 * len(events)), squeeze=False)
    for row, (a, b, node, frac) in zip(axs, events):
        busy = node.rstrip("*")
        parent = NODES[busy].parent
        kids = NODES[parent].children
        lo, hi = max(1, a - 150), min(T, b + 250)
        ts = range(lo, hi + 1)
        cut = slice(lo - 1, hi)

        def lines(ax, series, w=1):
            for k, c in enumerate(kids):
                ax.plot(ts, smooth(series(c, k), w)[cut], color=colors[k], lw=2.2 if c == busy else 1.1,
                        alpha=1.0 if c == busy else 0.75, label=c + ("  (busy)" if c == busy else ""))

        lines(row[0], lambda c, k: log["cap"][c], 5)
        row[0].plot(ts, smooth(log["got"][busy], 10)[cut], color="#222", lw=1.0, ls="--",
                    label=f"jobs arriving at {busy}")
        row[0].set(title=f"capacity (jobs per slot): {busy} loses {frac:.0%}", ylabel="jobs per slot")
        lines(row[1], lambda c, k: [x[k] for x in log["x"][parent]])
        row[1].set(title=f"routing shares x of {parent}", ylabel="share", ylim=(0, None))
        lines(row[2], lambda c, k: log["B"][c])
        row[2].plot(ts, smooth(log["cap"][busy], 5)[cut], color="#222", lw=1.0, ls=":",
                    label=f"capacity of {busy}")
        row[2].set(title="allocation B", ylabel="jobs per slot")
        row[2].legend(frameon=False, fontsize=7, loc="lower left")
        lines(row[3], lambda c, k: log["Q"][c], 10)
        row[3].set(title="physical queue, 10-slot mean", ylabel="jobs waiting")
        lines(row[4], lambda c, k: log["cost"][c], 10)
        row[4].plot(ts, [v / 10 for v in smooth([s["cost"] for s in slots], 10)[cut]], color="#222", lw=1.0,
                    ls="--", label="total L_t / 10")
        row[4].set(title="cost of each node, 10-slot mean", ylabel="cost")
        for ax in row:
            ax.axvspan(a, b, color="#000", alpha=0.06, lw=0)
            ax.set(xlim=(lo, hi), xlabel="slot")
            ax.set_ylim(0, None)
            ax.spines[["top", "right"]].set_visible(False)
            ax.title.set_fontsize(10)
        row[0].legend(frameon=False, fontsize=7, loc="lower left")
        row[4].legend(handles=row[4].lines[-1:], frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "events.png"), dpi=110)


def plot_dynamics(slots, log, out=OUT, groups=(ROOT, "sat1", "sat3.a0"), w=25):
    """The whole run for groups of siblings: does the traffic follow the congestion?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = len(slots)
    ts = range(1, T + 1)
    colors = ["#2a5fd0", "#e0812b", "#2e9e6b", "#b0438a"]

    def smooth(v):
        out_, acc = [], 0.0
        for k, a in enumerate(v):
            acc += a - (v[k - w] if k >= w else 0.0)
            out_.append(acc / min(k + 1, w))
        return out_

    def ratio(a, b):                                 # running mean of a per unit of b
        return [x / y if y else 0.0 for x, y in zip(smooth(a), smooth(b))]

    rows = [("capacity (jobs per slot)", lambda c, k, p: smooth(log["cap"][c])),
            ("routing share x", lambda c, k, p: smooth([x[k] for x in log["x"][p]])),
            ("allocation B (jobs per slot)", lambda c, k, p: smooth(log["B"][c])),
            ("physical queue (jobs waiting)", lambda c, k, p: smooth(log["Q"][c])),
            ("queueing time at the node (slots per job)", lambda c, k, p: ratio(log["wait"][c], log["n"][c])),
            ("processing time at the node (slots per job)", lambda c, k, p: ratio(log["proc"][c], log["n"][c])),
            ("cost of the node", lambda c, k, p: smooth(log["cost"][c]))]
    fig, axs = plt.subplots(len(rows), len(groups), figsize=(9.5 * len(groups), 2.9 * len(rows)),
                            sharex=True, squeeze=False)
    for col, parent in enumerate(groups):
        kids = NODES[parent].children
        for r, (title, series) in enumerate(rows):
            ax = axs[r, col]
            top = 0.0
            for k, c in enumerate(kids):
                y = series(c, k, parent)
                top = max(top, max(y[WARM:]))
                ax.plot(ts, y, color=colors[k], lw=1.3, label=c)
            ax.set(title=f"{title}   ({w}-slot mean)", xlim=(0, T), ylim=(0, 1.12 * top))
            for a, b, node, _ in EVENTS:             # busy periods that touch this group
                if any(node.rstrip("*") == c or node.startswith(c + ".") for c in kids):
                    ax.axvspan(a, b, color="#000", alpha=0.06, lw=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.title.set_fontsize(10)
        axs[0, col].legend(frameon=False, ncol=4, fontsize=8, loc="lower right",
                           title=f"children of {parent}")
        axs[-1, col].set_xlabel("slot")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "dynamics.png"), dpi=100)


def plot_response(log, out=OUT):
    """The average congestion episode at each level of the tree."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = [("root", "root -> platforms"), ("disp", "dispatcher -> its AMFs"),
              ("amf", "AMF -> its SMFs"), ("smf", "SMF -> its UPFs")]
    rows = [("Q", "physical queue (jobs waiting)"), ("x", "routing share x of the congested child"),
            ("B", "allocation B (jobs per slot)"), ("delay", "queueing time at the node (slots per job)"),
            ("cost", "cost of the node")]
    lags = range(LAGS[0], LAGS[1] + 1)
    fig, axs = plt.subplots(len(rows), len(levels), figsize=(5.2 * len(levels), 2.7 * len(rows)), sharex=True)
    for col, (kind, name) in enumerate(levels):
        e = episodes(log, kind)
        for r, (key, title) in enumerate(rows):
            ax = axs[r, col]
            ax.plot(lags, e[key], color="#d1495b", lw=2.0, label="the congested child")
            if key != "x":
                ax.plot(lags, e[key + "_sib"], color="#2a5fd0", lw=1.4, label="mean of its siblings")
            else:
                ax.axhline(mean(e["x"][:15]), color="#888", lw=0.8, ls=":", label="its share before")
            ax.axvline(0, color="#000", lw=0.6, alpha=0.4)
            lo, hi = min(e[key] + e.get(key + "_sib", e[key])), max(e[key] + e.get(key + "_sib", e[key]))
            pad = 0.15 * (hi - lo) or 0.01
            ax.set(title=f"{name}: {e['n']} episodes" if r == 0 else title, ylim=(max(0, lo - pad), hi + pad))
            if r == 0:
                ax.set_ylabel(title)
            ax.spines[["top", "right"]].set_visible(False)
            ax.title.set_fontsize(10)
            if col == 0:
                ax.legend(frameon=False, fontsize=8)
        axs[-1, col].set_xlabel("slots after the queue rose above the siblings'")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "response.png"), dpi=110)


if __name__ == "__main__":
    slots, log = run()
    check(slots, log)
    save(slots, log)
    plot(slots, log)
    plot_events(slots, log)
    plot_dynamics(slots, log)
    plot_response(log)
    print(f"\n-> {OUT}")
