#!/usr/bin/env python3
"""
Step 4 -- the run of sim.py, with every finished job executed as a real handover.

    sudo ./setup.sh           (once after a reboot)
    python3 core.py start
    python3 testbed.py

One job = one UE with s PDU sessions. For every leaf (48) and every s (1, 2, 3)
one UE is attached on exactly that AMF -> SMF -> UPF chain: 144 UEs. When a job
leaves its UPF leaf in the Python environment, that UE gets a real Xn handover:
the Path Switch goes through its AMF, its SMF (once per session) and its UPF
(once per session). The UE moves between the two gNBs of its AMF, so it can be
used again for the next job on the same leaf.

The algorithm reads nothing back from Open5GS, so its curves are those of
sim.py. What Open5GS adds is measured here:
  * every job the algorithm finished is one completed handover with s sessions;
  * the CPU time each real AMF / SMF / UPF spent on it (the real p per stage);
  * the real load leaving a busy node when the algorithm shifts the traffic.

Writes results/testbed/: handovers.csv, cpu.csv, testbed.png, and the same
slots.csv, nodes.csv, overview.png as sim.py.
"""
import csv
import os
import time

import core
import ran
import sim
from env import EVENTS, SESSIONS
from tree import NODES, N_SMF, N_UPF, of_kind, path

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "testbed")
PER_AMF = N_SMF * N_UPF          # leaves under one AMF
IDLE_S, IDLE_N = 2, 15           # idle CPU of the NFs before the run: median of 15 windows of 2 s
TRIM = 200                       # every so many slots the log files are emptied (the disk is small)
STAGES = ("amf", "smf", "upf")


def cpu_us(pid: int) -> float:
    """CPU time a process has used so far (all its threads), in microseconds."""
    return sum(int(open(f"/proc/{pid}/task/{t}/schedstat").read().split()[0])
               for t in os.listdir(f"/proc/{pid}/task")) / 1e3


def all_cpu(pid: dict) -> dict:
    return {i: cpu_us(p) for i, p in pid.items()}


def expected() -> dict[str, float]:
    """What every NF must hold while the 144 UEs are attached."""
    per_leaf = float(sum(SESSIONS))
    return {x.id: {"amf": float(PER_AMF * len(SESSIONS)), "smf": N_UPF * per_leaf, "upf": per_leaf}[x.kind]
            for x in core.NFS}


class Testbed:
    def __init__(self):
        pids = core.pids()
        if len(pids) != len(core.NFS):
            raise SystemExit("the core is not fully up -- python3 core.py start")
        self.pid = {x.id: pids[core.name(x)] for x in core.NFS}
        self.rans = {a.id: ran.Ran(a) for a in of_kind("amf")}
        self.ue = {}                         # (leaf, s) -> (its Ran, subscriber)
        self.handovers, self.cpu = [], []    # one row per job; one row per (slot, NF)

    def start(self) -> None:
        ran.kill_stray()
        if any(core.occupancy().values()):
            raise SystemExit("the core still holds sessions -- python3 core.py stop, then start")
        for amf in of_kind("amf"):
            r, g = self.rans[amf.id], core.amf_index(amf)
            r.start_gnbs()
            for s in SESSIONS:
                for smf in amf.children:
                    for leaf in NODES[smf].children:
                        x = NODES[leaf]
                        sub = PER_AMF * (len(SESSIONS) * g + s - 1) + N_UPF * x.k + x.u
                        r.add_ue(sub, s)
                        self.ue[leaf, s] = (r, sub)
            r.wait_attached()
            print(f"  {amf.id:8} {len(r.ues)} UEs attached")
        self.verify("after the attach")
        rates = {i: [] for i in self.pid}        # idle CPU of every NF, us per second
        self.clock, self.used = time.time(), all_cpu(self.pid)
        for _ in range(IDLE_N):
            time.sleep(IDLE_S)
            now, used = time.time(), all_cpu(self.pid)
            for i in self.pid:
                rates[i].append((used[i] - self.used[i]) / (now - self.clock))
            self.clock, self.used = now, used
        self.idle = {i: sorted(v)[IDLE_N // 2] for i, v in rates.items()}

    def verify(self, when: str) -> None:
        """Every NF holds exactly its UEs / sessions (from its own metrics page)."""
        want = expected()
        end = time.time() + 20
        while True:
            occ = core.occupancy()
            bad = {k: (occ[k], v) for k, v in want.items() if occ[k] != v}
            if not bad or time.time() > end:
                break
            time.sleep(0.5)
        if bad:
            raise RuntimeError(f"{when}: wrong occupancy (got, expected) {dict(list(bad.items())[:6])}")
        print(f"  {when}: every AMF holds {PER_AMF * len(SESSIONS)} UEs, every SMF "
              f"{N_UPF * sum(SESSIONS)} sessions, every UPF {sum(SESSIONS)} sessions")

    def slot(self, t: int, jobs: list) -> None:
        """Execute the jobs that left their leaf in slot t as handovers; measure the NFs."""
        if t % TRIM == 0:                    # nothing is in flight between two slots
            core.trim_logs()
            for r in self.rans.values():
                for p in list(r.gnbs.values()) + list(r.ues.values()):
                    p.trim()
        todo = list(jobs)
        while todo:                          # a UE needed twice in one slot goes in a 2nd round
            batch, later = {}, []
            for job in todo:
                r, sub = self.ue[job.leaf, job.s]
                if sub in batch:
                    later.append(job)
                else:
                    batch[sub] = job
                    r.trigger(sub)
            for sub, res in ran.finish(list(self.rans.values())).items():
                job = batch[sub]
                self.handovers.append({"t": t, "leaf": job.leaf, "s": job.s,
                                       "switched": res["sessions"], "ms": res["total_ms"]})
            todo = later
        n = {i: [0, 0] for i in self.pid}    # handovers and sessions each NF executed
        for job in jobs:
            for i in path(job.leaf)[2:]:
                n[i][0] += 1
                n[i][1] += job.s
        now, used = time.time(), all_cpu(self.pid)
        for i in self.pid:                   # CPU of this slot, idle part removed
            self.cpu.append((t, i, n[i][0], n[i][1],
                             used[i] - self.used[i] - self.idle[i] * (now - self.clock)))
        self.clock, self.used = now, used

    def close(self) -> None:
        for r in self.rans.values():
            r.close()


# ── checks, files, figure ────────────────────────────────────────────────────
def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else 0.0


def fit(rows) -> tuple[float, float, float, float]:
    """Least squares  cpu = a * handovers + b * sessions  over (slot, NF) rows.

    Some slots carry CPU that no handover caused (a timer of the NF, the whole
    machine being slow for a moment). Rows further than 3 robust standard
    deviations from the fit are left out and the fit is repeated.
    Returns a, b, R^2 of the rows kept, and that 3-deviation limit.
    """
    import numpy as np
    A = np.array([[r[2], r[3]] for r in rows], dtype=float)
    y = np.array([r[4] for r in rows])
    keep = np.ones(len(y), dtype=bool)
    for _ in range(6):
        (a, b), *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        res = y - A @ [a, b]
        limit = 3 * 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        keep = np.abs(res) < limit
    r2 = 1 - (res[keep] ** 2).sum() / ((y[keep] - y[keep].mean()) ** 2).sum()
    return a, b, r2, limit


def group(node: str) -> list[str]:
    """The NFs whose CPU stands for a tree node: itself, or the AMFs of a dispatcher."""
    return [node] if NODES[node].kind in STAGES else NODES[node].children


def check(tb: Testbed, slots: list) -> dict:
    T = len(slots)
    done = sum(s["done"] for s in slots)
    wrong = sum(h["switched"] != h["s"] for h in tb.handovers)
    ms = sorted(h["ms"] for h in tb.handovers)
    print(f"\njobs finished by the algorithm {done}   real handovers completed {len(tb.handovers)}   "
          f"with a wrong number of sessions {wrong}")
    print(f"handover time: median {ms[len(ms) // 2]:.1f} ms   95% {ms[int(0.95 * len(ms))]:.1f} ms   "
          f"max {ms[-1]:.1f} ms")

    print("\nmeasured CPU per handover   cpu = a + b * s   (microseconds, idle removed, outlier slots left out)")
    print(f"  {'stage':5} {'a':>7} {'b':>7} {'R^2':>6}   " + "".join(f"{'s=' + str(s):>8}" for s in SESSIONS)
          + "   median CPU in slots with no handover")
    p, used = {}, {}                                        # used: (node, slot) -> CPU
    for kind in STAGES:
        ids = {x.id for x in of_kind(kind)}
        rows = [r for r in tb.cpu if r[1] in ids]
        a, b, r2, limit = fit([r for r in rows if r[2]])
        p[kind] = (a, b)
        zero = sorted(r[4] for r in rows if not r[2])
        print(f"  {kind:5} {a:7.0f} {b:7.0f} {r2:6.2f}   " + "".join(f"{a + b * s:8.0f}" for s in SESSIONS)
              + f"   {zero[len(zero) // 2]:6.1f}")
        for t, i, j, n, c in rows:                          # CPU no handover caused is cut off
            used[i, t] = min(c, a * j + b * n + limit)
    tb.used_clean = used

    def share(child, a, b):
        sibs = NODES[NODES[child].parent].children
        tot = sum(used[i, t] for s in sibs for i in group(s) for t in range(a, b + 1))
        return sum(used[i, t] for i in group(child) for t in range(a, b + 1)) / tot

    print("\nreal CPU of the busy node among its siblings (share of the Open5GS CPU time)")
    for a, b, node, frac in EVENTS:
        if b > T:
            continue
        child = node.rstrip("*")
        print(f"  {child:10} busy in slots {a}-{b}: {share(child, max(1, a - 200), a - 1):.2f} -> "
              f"{share(child, a + 100, b):.2f} -> {share(child, b + 200, min(T, b + 400)):.2f} after")
    return p


def save(tb: Testbed) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "handovers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tb.handovers[0]))
        w.writeheader()
        w.writerows(tb.handovers)
    with open(os.path.join(OUT, "cpu.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "node", "handovers", "sessions", "cpu_us"])
        w.writerows((t, i, j, s, f"{c:.1f}") for t, i, j, s, c in tb.cpu)


def plot(tb: Testbed, slots: list, p: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = len(slots)
    ts = range(1, T + 1)
    colors = ["#2a5fd0", "#e0812b", "#2e9e6b", "#b0438a"]
    used = {x.id: [0.0] * T for x in core.NFS}
    for (i, t), c in tb.used_clean.items():
        used[i][t - 1] = c / 1e3                            # ms of CPU per slot

    def smooth(v, w=25):
        out, acc = [], 0.0
        for k, a in enumerate(v):
            acc += a - (v[k - w] if k >= w else 0.0)
            out.append(acc / min(k + 1, w))
        return out

    def cpu_panel(ax, groups, title, busy):
        top = 0.0
        for k, (label, ids) in enumerate(groups):
            y = smooth([sum(used[i][t] for i in ids) for t in range(T)])
            top = max(top, max(y[min(sim.WARM, T - 1):]))
            ax.plot(ts, y, color=colors[k], lw=2.2 if label in busy else 1.2, label=label)
        ax.set(title=title, ylabel="real CPU, ms per slot", xlabel="slot", xlim=(0, T), ylim=(0, 1.35 * top))
        for a, b, node, _ in EVENTS:                     # shade the periods of the nodes in this panel
            if node.rstrip("*") in busy:
                ax.axvspan(a, b, color="#000", alpha=0.06, lw=0)
        ax.legend(frameon=False, ncol=4, fontsize=7, loc="upper right")

    def nfs_under(node):                                 # the Open5GS NFs that stand for a tree node
        return [x.id for x in core.NFS if x.id == node or x.id.startswith(node + ".")]

    fig, axs = plt.subplots(2, 3, figsize=(16, 7.5))
    period = {e[2].rstrip("*"): k + 1 for k, e in enumerate(EVENTS)}
    for ax, kind in zip(axs[0], ("amf", "smf", "disp")):
        busy = [n for n in period if NODES[n].kind == kind]
        parent = NODES[busy[0]].parent if busy else None
        if parent is None:
            ax.set_visible(False)
            continue
        kids = NODES[parent].children
        names = {"amf": "AMFs", "smf": "SMFs", "disp": "platforms"}[kind]
        cpu_panel(ax, [(c, nfs_under(c) if kind == "disp" else [c]) for c in kids],
                  (f"real CPU of the {names} of {parent}" if kind != "disp" else "real CPU per platform")
                  + (" (20 NFs each)" if kind == "disp" else "") + ";  busy: "
                  + ", ".join(f"{n} (#{period[n]})" for n in busy if n in kids),
                  [n for n in busy if n in kids])

    ax = axs[1, 0]
    per_slot = [0] * T
    for h in tb.handovers:
        per_slot[h["t"] - 1] += 1
    ax.plot(ts, smooth(per_slot), color=colors[0], lw=1.5, label="real handovers completed")
    ax.plot(ts, smooth([s["gamma"] for s in slots]), color="#888", lw=1.0, label="jobs injected at the root")
    ax.set(title="handovers executed per slot, 25-slot mean", ylabel="per slot", xlabel="slot",
           xlim=(0, T), ylim=(0, 1.5 * mean(per_slot)))
    ax.legend(frameon=False, fontsize=7)

    ax = axs[1, 1]
    for k, kind in enumerate(STAGES):
        a, b = p[kind]
        ax.plot(SESSIONS, [a + b * s for s in SESSIONS], "o-", color=colors[k], lw=1.5, ms=5,
                label=kind.upper())
    ax.set(title="measured CPU of one handover, by sessions s", xlabel="sessions s of the job",
           ylabel="CPU time, microseconds", xticks=SESSIONS, ylim=(0, None))
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axs[1, 2]
    ms = sorted(h["ms"] for h in tb.handovers)
    ax.hist(ms[:int(0.99 * len(ms))], bins=60, color=colors[0])
    ax.set(title="handover time  (slowest 1% cut)",
           xlabel="ms", ylabel="handovers")

    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.title.set_fontsize(10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "testbed.png"), dpi=130)


def main(T: int = sim.T) -> None:
    tb = Testbed()
    try:
        print("attaching the UEs")
        tb.start()
        print(f"running {T} slots")
        t0 = time.time()
        slots, log = sim.run(T=T, on_done=tb.slot)
        print(f"  {len(tb.handovers)} handovers in {time.time() - t0:.0f} s")
        tb.verify("after the run")
    finally:
        tb.close()
    sim.check(slots, log)
    p = check(tb, slots)
    sim.save(slots, log, OUT)
    sim.plot(slots, log, OUT)
    save(tb)
    plot(tb, slots, p)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
