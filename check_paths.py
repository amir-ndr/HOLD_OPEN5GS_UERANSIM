#!/usr/bin/env python3
"""
Step 2 check -- every leaf of the tree is reachable and can be handed over.

For each of the 8 AMFs: attach 6 UEs (one per UPF under that AMF, 3 PDU sessions
each), verify from the NFs' own counters that the sessions sit on exactly the
addressed AMF / SMF / UPF and nowhere else, hand every UE over to the AMF's
other gNB and back, and verify nothing moved in the core.

    python3 core.py start
    python3 check_paths.py

Writes results/paths/paths.csv, handovers.csv and handover.png.
"""
import csv
import os
import time

import core
import ran
from tree import NODES, N_SMF, N_UPF, of_kind

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "paths")
N_SESS = len(core.SESSIONS)
PER_AMF = N_SMF * N_UPF


def expected(amf) -> dict[str, float]:
    """Counters every NF must show while this AMF's 6 UEs are attached."""
    want = {x.id: 0.0 for x in core.NFS}
    want[amf.id] = PER_AMF
    for s in amf.children:
        want[s] = N_UPF * N_SESS
        for u in NODES[s].children:
            want[u] = N_SESS
    return want


def settle(want: dict, timeout: float = 25.0) -> dict:
    """Poll the counters until they match; return the mismatches left."""
    end = time.time() + timeout
    while True:
        occ = core.occupancy()
        bad = {k: (occ[k], v) for k, v in want.items() if occ[k] != v}
        if not bad or time.time() > end:
            return bad
        time.sleep(0.5)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    if len(core.pids()) != len(core.NFS):
        raise SystemExit("the core is not fully up -- python3 core.py start")
    ran.kill_stray()
    idle = {x.id: 0.0 for x in core.NFS}
    if settle(idle, 15):
        raise SystemExit("the core still holds sessions -- python3 core.py stop, then start")

    paths, hos = [], []
    for amf in of_kind("amf"):
        g = core.amf_index(amf)
        r = ran.Ran(amf)
        row = {"amf": amf.id, "attached": 0, "landed": 0, "ho_out": 0, "ho_back": 0, "kept": 0}
        try:
            r.start_gnbs()
            for j in range(PER_AMF):
                r.add_ue(PER_AMF * g + j, N_SESS)
            r.wait_attached()
            row["attached"] = 1
            bad = settle(expected(amf))
            row["landed"] = int(not bad)
            for direction in ("ho_out", "ho_back"):
                for sub in r.ues:
                    r.trigger(sub)
                res = list(ran.finish([r]).values())
                row[direction] = int(len(res) == PER_AMF and all(x["sessions"] == N_SESS for x in res))
                hos += [dict(amf=amf.id, direction=direction, **x) for x in res]
                time.sleep(0.5)
            bad = bad or settle(expected(amf), 10)
            row["kept"] = int(not bad)
            if bad:
                print(f"    mismatch (got, expected): {dict(list(bad.items())[:6])}")
        except Exception as e:
            print(f"    {type(e).__name__}: {str(e)[:600]}")
        finally:
            r.close()
        left = settle(idle, 20)
        row["released"] = int(not left)
        paths.append(row)
        ms = [x["total_ms"] for x in hos if x["amf"] == amf.id]
        print(f"  {amf.id:8} attach {row['attached']}  landed {row['landed']}  out {row['ho_out']}"
              f"  back {row['ho_back']}  kept {row['kept']}  released {row['released']}"
              + (f"   handover {min(ms):.1f}-{max(ms):.1f} ms" if ms else ""))
        if left:
            print(f"    not released: {dict(list(left.items())[:6])}")
            break

    for name, rows in (("paths.csv", paths), ("handovers.csv", hos)):
        if rows:
            with open(os.path.join(OUT, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    if hos:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.2))
        amfs = [a.id for a in of_kind("amf")]
        for d, color, dx in (("ho_out", "#2a5fd0", -0.12), ("ho_back", "#e0812b", 0.12)):
            pts = [(amfs.index(h["amf"]) + dx, h["total_ms"]) for h in hos if h["direction"] == d]
            ax.plot(*zip(*pts), "o", ms=4, color=color, alpha=0.8,
                    label="to the other gNB" if d == "ho_out" else "back")
        ax.set_xticks(range(len(amfs)), amfs, rotation=30, ha="right", fontsize=8)
        ax.set(ylabel="handover time (ms)", ylim=(0, None))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "handover.png"), dpi=150)

    ok = len(paths) == len(of_kind("amf")) and all(all(v for k, v in r.items() if k != "amf") for r in paths)
    leaves = sum(r["landed"] for r in paths) * PER_AMF
    print(f"\n{'OK  ' if ok else 'FAIL'} {leaves}/{len(of_kind('upf'))} leaves verified, "
          f"{len(hos)} handovers completed  -> {OUT}")


if __name__ == "__main__":
    main()
