#!/usr/bin/env python3
"""
Step 1 -- UE, TN and satellite topology.

Propagates the Starlink constellation over the UE, follows the serving
satellite, and at every handover event lists the four target platforms of the
paper's tree: the three best candidate satellites and the TN node.

    python3 topology.py

Writes results/topology/events.csv, timeline.csv and serving.png, and prints
the checks. There are no options: change the constants below.
"""
import csv
import os

import numpy as np
from sgp4.api import Satrec, SatrecArray

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "topology")
TLE = "/home/amirndr/5g-lab/starlink.tle"

UE_LAT, UE_LON, UE_ALT_M = 45.5017, -73.5673, 50.0      # Montreal
TN_LAT, TN_LON, TN_ALT_M = 45.5300, -73.6200, 80.0      # terrestrial BS, ~5 km away

SHELL_KM = (500.0, 600.0)     # Starlink shell used
WINDOW_S = 90 * 60            # simulated time
STEP_S = 5                    # sample period
ELEV_TRIGGER = 25.0           # serving below this -> handover event
ELEV_MIN = 10.0               # a satellite is unusable below this
N_CAND = 3                    # candidate target satellites per event
ISL_GRAZE_KM = 80.0           # an ISL must stay above this altitude

MU = 398600.4418              # km^3/s^2
RE = 6378.137                 # km
F = 1 / 298.257223563


def load_tle():
    """Satellites of the chosen shell with a fresh element set."""
    lines = [l.rstrip() for l in open(TLE) if l.strip()]
    names, sats = [], []
    for i in range(0, len(lines) - 2, 3):
        s = Satrec.twoline2rv(lines[i + 1], lines[i + 2])
        alt = (MU / (s.no_kozai / 60.0) ** 2) ** (1 / 3) - RE
        if SHELL_KM[0] <= alt <= SHELL_KM[1]:
            names.append(lines[i].strip())
            sats.append(s)
    epoch = np.array([s.jdsatepoch + s.jdsatepochF for s in sats])
    t0 = float(np.median(epoch))
    keep = np.abs(epoch - t0) < 7.0           # stale elements propagate badly
    return [n for n, k in zip(names, keep) if k], [s for s, k in zip(sats, keep) if k], t0


def ecef_of(lat, lon, alt_m):
    """Geodetic (WGS84) -> ECEF position [km] and local 'up' unit vector."""
    la, lo, h = np.radians(lat), np.radians(lon), alt_m / 1000.0
    e2 = F * (2 - F)
    n = RE / np.sqrt(1 - e2 * np.sin(la) ** 2)
    pos = np.array([(n + h) * np.cos(la) * np.cos(lo),
                    (n + h) * np.cos(la) * np.sin(lo),
                    (n * (1 - e2) + h) * np.sin(la)])
    up = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return pos, up


def gmst(jd):
    """Greenwich mean sidereal time [rad] (IAU 1982)."""
    t = (jd - 2451545.0) / 36525.0
    sec = (67310.54841 + (876600 * 3600 + 8640184.812866) * t
           + 0.093104 * t ** 2 - 6.2e-6 * t ** 3)
    return np.radians((sec % 86400.0) / 240.0)


def propagate(sats, jd):
    """ECEF positions [km], shape (n_sat, n_time, 3), and the SGP4 error flags."""
    whole = np.floor(jd)
    err, r, _ = SatrecArray(sats).sgp4(whole, jd - whole)
    g = gmst(jd)
    c, s = np.cos(g), np.sin(g)
    x = c * r[:, :, 0] + s * r[:, :, 1]         # TEME -> ECEF: rotate by -GMST
    y = -s * r[:, :, 0] + c * r[:, :, 1]
    return np.stack([x, y, r[:, :, 2]], axis=2), err


def elevation(pos, site, up):
    d = pos - site
    return np.degrees(np.arcsin((d @ up) / np.linalg.norm(d, axis=-1)))


def isl(a, b):
    """Range [km] between two satellites and whether the link clears the atmosphere."""
    d = b - a
    t = np.clip(-(a @ d) / (d @ d), 0.0, 1.0)
    return float(np.linalg.norm(d)), bool(np.linalg.norm(a + t * d) >= RE + ISL_GRAZE_KM)


def build():
    names, sats, t0 = load_tle()
    ue, ue_up = ecef_of(UE_LAT, UE_LON, UE_ALT_M)
    tn, _ = ecef_of(TN_LAT, TN_LON, TN_ALT_M)

    # coarse pass over the whole shell, then a fine pass over what the UE can see
    coarse = t0 + np.arange(0, WINDOW_S + 1, 30) / 86400.0
    pos, err = propagate(sats, coarse)
    seen = (elevation(pos, ue, ue_up).max(axis=1) > ELEV_MIN - 5) & (err == 0).all(axis=1)
    names = [n for n, k in zip(names, seen) if k]
    sats = [s for s, k in zip(sats, seen) if k]

    secs = np.arange(0, WINDOW_S + 1, STEP_S)
    jd = t0 + secs / 86400.0
    pos, _ = propagate(sats, jd)
    elev = elevation(pos, ue, ue_up)

    serving = int(np.argmax(elev[:, 0]))
    timeline, events = [], []
    for k, t in enumerate(secs):
        if elev[serving, k] < ELEV_TRIGGER:
            order = [int(i) for i in np.argsort(-elev[:, k]) if i != serving][:N_CAND]
            ev = {"t_s": int(t), "serving": names[serving],
                  "serving_elev": round(float(elev[serving, k]), 2)}
            for n, c in enumerate(order, 1):
                rng, los = isl(pos[serving, k], pos[c, k])
                ev[f"sat{n}"] = names[c]
                ev[f"sat{n}_elev"] = round(float(elev[c, k]), 2)
                ev[f"sat{n}_isl_km"] = round(rng, 1)
                ev[f"sat{n}_isl_los"] = int(los)
            ev["tn_km"] = round(float(np.linalg.norm(tn - ue)), 2)
            events.append(ev)
            serving = order[0]                 # the UE moves to the best candidate
        timeline.append({"t_s": int(t), "serving": names[serving],
                         "elev": round(float(elev[serving, k]), 2),
                         "visible": int((elev[:, k] >= ELEV_TRIGGER).sum())})
    return dict(names=names, sats=sats, jd=jd, secs=secs, pos=pos, elev=elev,
                ue=ue, timeline=timeline, events=events, t0=t0)


def check(tp):
    names, pos, secs, elev = tp["names"], tp["pos"], tp["secs"], tp["elev"]
    tl, ev = tp["timeline"], tp["events"]
    idx = {n: i for i, n in enumerate(names)}
    chain = sorted({r["serving"] for r in tl}, key=lambda n: idx[n])
    ci = [idx[n] for n in chain]

    alt = np.linalg.norm(pos[ci], axis=2) - RE
    # speed in the inertial frame: add back the Earth's rotation
    w = np.array([0, 0, 7.2921159e-5])
    v = np.diff(pos[ci], axis=1) / STEP_S + np.cross(w, pos[ci][:, :-1])
    speed = np.linalg.norm(v, axis=2)
    period = np.array([2 * np.pi / tp["sats"][i].no_kozai for i in ci])

    print(f"shell satellites the UE can see in the window : {len(names)}")
    print(f"serving satellites in {WINDOW_S // 60} min                : {len(chain)}")
    print(f"  altitude  {alt.min():.0f}-{alt.max():.0f} km   speed {speed.min():.2f}-{speed.max():.2f} km/s"
          f"   period {period.min():.1f}-{period.max():.1f} min")

    dwell = np.diff([0] + [e["t_s"] for e in ev])
    s_el = np.array([r["elev"] for r in tl])
    print(f"handover events                               : {len(ev)}")
    print(f"  time between events  mean {dwell.mean():.0f} s  min {dwell.min()} s  max {dwell.max()} s")
    print(f"  serving elevation    min {s_el.min():.1f} deg  mean {s_el.mean():.1f} deg")
    print(f"  satellites above {ELEV_TRIGGER:.0f} deg  min {min(r['visible'] for r in tl)}"
          f"  mean {np.mean([r['visible'] for r in tl]):.1f}")

    # cross-check our own TEME->ECEF and elevation against skyfield
    from skyfield.api import EarthSatellite, load, wgs84
    ts = load.timescale(builtin=True)
    site = wgs84.latlon(UE_LAT, UE_LON, UE_ALT_M)
    worst = 0.0
    for e in ev[:: max(1, len(ev) // 8)]:
        k = int(np.searchsorted(secs, e["t_s"]))
        s = tp["sats"][idx[e["sat1"]]]
        sky = EarthSatellite.from_satrec(s, ts)
        el = (sky - site).at(ts.ut1_jd(tp["jd"][k])).altaz()[0].degrees
        worst = max(worst, abs(el - elev[idx[e["sat1"]], k]))

    ok = lambda b: "OK  " if b else "FAIL"
    full = [e for e in ev if f"sat{N_CAND}" in e]
    print("\nchecks")
    print(f"  {ok(worst < 0.1)} elevation agrees with skyfield (worst difference {worst:.3f} deg)")
    print(f"  {ok(s_el.min() >= ELEV_MIN)} the UE is never served below {ELEV_MIN:.0f} deg")
    print(f"  {ok(len(full) == len(ev))} every event has {N_CAND} candidate satellites")
    low = [e for e in full if e[f"sat{N_CAND}_elev"] < ELEV_TRIGGER]
    print(f"  {ok(not low)} every candidate is above {ELEV_TRIGGER:.0f} deg ({len(low)} event(s) are not)")
    nolos = sum(1 - e[f"sat{n}_isl_los"] for e in full for n in range(1, N_CAND + 1))
    print(f"  {ok(nolos == 0)} serving-to-candidate ISL has line of sight ({nolos} link(s) do not)")
    same = sum(a["serving"] == b["serving"] for a, b in zip(ev, ev[1:]))
    print(f"  {ok(same == 0)} every event changes the serving satellite")


def save(tp):
    os.makedirs(OUT, exist_ok=True)
    for name, rows in (("events.csv", tp["events"]), ("timeline.csv", tp["timeline"])):
        with open(os.path.join(OUT, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = {n: i for i, n in enumerate(tp["names"])}
    minutes = tp["secs"] / 60.0
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for n in {r["serving"] for r in tp["timeline"]}:      # each serving satellite's full pass
        e = tp["elev"][idx[n]]
        ax.plot(minutes, np.where(e >= 0, e, np.nan), color="#b8bcc4", lw=0.8)
    ax.plot(minutes, [r["elev"] for r in tp["timeline"]], color="#2a5fd0", lw=2,
            label="serving satellite")
    ax.axhline(ELEV_TRIGGER, color="#555", lw=0.8, ls="--")
    ax.text(minutes[-1], ELEV_TRIGGER + 1, "handover trigger", ha="right", fontsize=8, color="#555")
    ax.plot([e["t_s"] / 60 for e in tp["events"]], [e["serving_elev"] for e in tp["events"]],
            "o", ms=4, color="#d1495b", label="handover event")
    ax.set(xlabel="time (min)", ylabel="elevation at the UE (deg)", xlim=(0, minutes[-1]), ylim=(0, 90))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "serving.png"), dpi=150)


if __name__ == "__main__":
    tp = build()
    check(tp)
    save(tp)
    print(f"\nfirst events (targets = sat1, sat2, sat3, TN)")
    for e in tp["events"][:5]:
        print(f"  t={e['t_s']:5d}s  {e['serving']} ({e['serving_elev']} deg) -> "
              + ", ".join(f"{e[f'sat{n}']} {e[f'sat{n}_elev']:.0f} deg" for n in range(1, N_CAND + 1)))
    print(f"\n-> {OUT}")
