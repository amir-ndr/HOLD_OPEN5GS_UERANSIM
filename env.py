#!/usr/bin/env python3
"""
The environment the algorithm acts on: jobs, physical FIFO queues, node
capacities, background load, and the cost feedback sent to the root.

JOB       <p_disp, p_amf, p_smf, p_upf, s>: CPU cycles needed at each stage
          (in Mc = 10^6 cycles) and the number of PDU sessions. Drawn at random
          for every job. At AMF -> SMF the job unpacks into s session tasks,
          each needing p_smf and then p_upf; the s tasks stay together.
          p_amf, p_smf, p_upf come from the CPU measured on Open5GS (testbed.py):
          the measured value for the job's s, times a random factor.
QUEUES    counted in jobs (root, dispatcher, AMF) or session tasks (SMF, UPF).
CAPACITY  C_i Mc per slot (TN a little larger). Two things take part of it away:
          a random background load on every node, a share of C_i that drifts
          slowly inside the range BACKGROUND (so the capacities of TN and of the
          satellites vary over ranges that overlap), and the scripted busy
          periods EVENTS.
TIME      for every job at every node: the slots it waited before the slot in
          which the node finished it (queueing), and the share of a slot the node
          needed for it, cycles / capacity (processing). A job moves one stage
          per slot, so time in system = 4 slots + its queueing at the 5 nodes.
SERVICE   a node is a FIFO server. In slot t it may process and forward up to
          B_i[t] = sum_j Y_ij[t] (the allocation), and it can spend at most its
          capacity of that slot in CPU cycles. A job too large for what is left
          of the slot is continued in the next slot; unused capacity is lost,
          never saved. The jobs it forwards are split among its children in the
          proportions x_ij = Y_ij / B_i, whatever the number of jobs is.
          B and Y are allocations; this is their job-level realisation, which
          the paper leaves to the local scheduler.

COST (what is fed back) -- convex in Y, built from two functions:

    cap_i  = C_i (1 - background_i[t]) / p_i[t]   jobs node i can process per slot
    phi(u)  processing cost of a load u = (jobs per slot) / cap
    psi(r)  queueing cost of a backlog r = (jobs left waiting) / cap, i.e. in slots

    L_t(Y) = sum of  cap * phi(load)   +   sum_i  cap_i * psi(R_i / cap_i)

    R_i = [ Q_i[t] + arrivals_i(Y) - served_i ]+   backlog node i is left with

Four choices, all kept as options (COST below picks one of each):

    phi    "quad" u^2/2 | "quartic" u^2/2 + u^4/4 | "steep" u^2/2 + u^8/8 | "mm1" u/(1-u) scaled
           "power4" u^4/4 | "power8" u^8/8   (almost free below the capacity, sharp near it)
    psi    "none" | "linear" r | "quad" r^2/2 | "cubic" r^3/3
    load   "own"    every node pays phi for what it processes itself:  B_i / cap_i
           "child"  every edge pays phi for what it offers to the child:  lambda_ij Y_ij / cap_j
                    (the paper's wording: the penalty of node i's allocation and of
                    its distribution among downstream functions); a leaf pays for B_i
    drain  "plain"  served_i = B_i
           "capped" served_i = min(B_i, cap_i): allocating beyond the capacity drains nothing

Gradient reported for edge (i, j):

    dL/dY_ij =   phi'(B_i / cap_i)                            load "own"
               | lambda_ij * phi'(lambda_ij Y_ij / cap_j)     load "child"
               - psi'(R_i / cap_i)                node i drains its own backlog
               + lambda_ij * psi'(R_j / cap_j)    child j's backlog grows

    The queueing part is a backpressure on the PHYSICAL queues, with the same
    shape as the paper's virtual-queue weight  lambda_ij Z_j - Z_i.
"""
import random
from collections import deque

from hold import SINK, children
from tree import NODES, ROOT

# ── numbers ──────────────────────────────────────────────────────────────────
GAMMA = (16, 48)                    # jobs injected at the root per slot, uniform
SESSIONS = (1, 2, 3)                # s of a job, uniform
# CPU time of ONE handover with s sessions, measured on Open5GS: microseconds = a + b * s
MEASURED = {"amf": (174, 62), "smf": (118, 80), "upf": (74, 45)}
CPU_GHZ = 3.2                       # assumed clock (the VM does not show it): 1 us = 3200 cycles
SPREAD = 0.25                       # a job's p = measured value * uniform(1 - SPREAD, 1 + SPREAD)
P_DISP = (0.15, 0.25)               # Mc per job at a dispatcher (Python only, nothing to measure)
CAPACITY = {"root": 12.8, "disp": 3.2, "amf": 7.6, "smf": 2.4, "upf": 0.7}   # Mc per slot
TN_FACTOR = 1.10                    # TN nodes are a little larger
BACKGROUND = (0.0, 0.5)             # random background load of a node: range of the share it takes
DRIFT = 0.02                        # how far that share moves per slot (random step, both ways)

# background load: (first slot, last slot, node, share of its capacity taken away)
# "sat3*" means sat3 and every node below it
EVENTS = [(400, 800, "sat1.a0", 0.8),          # one AMF gets busy
          (1100, 1500, "sat2.a0.s1", 0.7),     # one SMF of a satellite gets busy
          (1800, 2200, "sat3*", 0.5),          # a whole satellite gets busy
          (2500, 2900, "tn*", 0.5)]            # the whole TN gets busy

# ── the cost options ─────────────────────────────────────────────────────────
COST = {"phi": "power4", "psi": "quad", "load": "own", "drain": "plain"}   # the cost in use

U0 = 0.9                            # "mm1" follows u/(1-u) up to here, then its 2nd-order extension


def _mm1(u: float) -> float:        # scaled by 1/8: same slope as "quad" at half load
    if u <= U0:
        return u / (8 * (1 - u))
    d = u - U0
    return U0 / (8 * (1 - U0)) + d / (8 * (1 - U0) ** 2) + d * d / (8 * (1 - U0) ** 3)


def _mm1_prime(u: float) -> float:
    if u <= U0:
        return 1 / (8 * (1 - u) ** 2)
    return 1 / (8 * (1 - U0) ** 2) + (u - U0) / (4 * (1 - U0) ** 3)


PHI = {"quad": (lambda u: u * u / 2, lambda u: u),               # name: (phi, phi')
       "quartic": (lambda u: u * u / 2 + u ** 4 / 4, lambda u: u + u ** 3),
       "steep": (lambda u: u * u / 2 + u ** 8 / 8, lambda u: u + u ** 7),
       "mm1": (_mm1, _mm1_prime),
       "power4": (lambda u: u ** 4 / 4, lambda u: u ** 3),
       "power8": (lambda u: u ** 8 / 8, lambda u: u ** 7)}
PSI = {"none": (lambda r: 0.0, lambda r: 0.0),                   # name: (psi, psi')
       "linear": (lambda r: r, lambda r: 1.0 if r > 0 else 0.0),
       "quad": (lambda r: r * r / 2, lambda r: r),
       "cubic": (lambda r: r ** 3 / 3, lambda r: r * r)}
LOAD = ("own", "child")
DRAIN = ("plain", "capped")

STAGE = {"root": "disp", "disp": "disp", "amf": "amf", "smf": "smf", "upf": "upf"}


def units(kind: str, s: int) -> int:
    """Tasks one job is at this stage: 1 job, or s session tasks after the AMF."""
    return s if kind in ("smf", "upf") else 1


def p_measured(kind: str, s: int) -> float:
    """Mc per task at this stage for a job with s sessions (the handover's CPU, shared by its tasks)."""
    a, b = MEASURED[kind]
    return (a + b * s) * CPU_GHZ / 1e3 / units(kind, s)


P_MEAN = {"disp": sum(P_DISP) / 2,                  # mean Mc per task, used when a queue is empty
          **{k: sum(p_measured(k, s) * units(k, s) for s in SESSIONS)
             / sum(units(k, s) for s in SESSIONS) for k in MEASURED}}


def busy(events, i: str, t: int) -> float:
    """Share of its capacity node i loses to a scripted busy period in slot t."""
    return max([share for a, b, node, share in events if a <= t <= b
                and (i == node or (node.endswith("*") and i.startswith(node[:-1])))], default=0.0)


class Job:
    __slots__ = ("p", "s", "born", "leaf", "at", "wait", "proc")

    def __init__(self, rng: random.Random, born: int):
        self.s = rng.choice(SESSIONS)
        self.p = {"disp": rng.uniform(*P_DISP)}
        for kind in MEASURED:
            self.p[kind] = p_measured(kind, self.s) * rng.uniform(1 - SPREAD, 1 + SPREAD)
        self.born = born
        self.leaf = None
        self.at = born                  # first slot it can be served at the node it is at
        self.wait = self.proc = 0.0     # its queueing and processing time so far, in slots


class Env:
    def __init__(self, seed: int = 1, events=EVENTS, cost=COST, gamma=GAMMA):
        self.rng = random.Random(seed)
        self.noise = random.Random(seed + 7919)      # separate, so the jobs do not depend on it
        self.events, self.cost, self.gamma = events, cost, gamma
        self.bg = {i: self.noise.uniform(*BACKGROUND) for i in NODES}   # random background load now
        self.queue = {i: deque() for i in NODES}
        self.allow = {i: 0.0 for i in NODES}        # allocation B_i not used yet
        self.x = {i: [1 / len(children(i))] * len(children(i)) for i in NODES}   # routing shares in force
        self.owed = {(i, j): 0.0 for i in NODES for j in children(i)}    # child j's share not yet sent
        self.started = {i: 0.0 for i in NODES}      # cycles already spent on the job at the head
        self.lam = {(i, j): 1.0 for i in NODES for j in children(i)}
        for i, n in NODES.items():
            if n.kind == "amf":
                for j in n.children:
                    self.lam[i, j] = sum(SESSIONS) / len(SESSIONS)

    # how much one job weighs in node i's queue, and what it costs node i
    @staticmethod
    def size(i: str, job: Job) -> int:
        return units(NODES[i].kind, job.s)

    @staticmethod
    def cycles(i: str, job: Job) -> float:
        return job.p[STAGE[NODES[i].kind]] * Env.size(i, job)

    def backlog(self, i: str) -> int:
        return sum(self.size(i, j) for j in self.queue[i])

    def busy(self, i: str, t: int) -> float:
        return busy(self.events, i, t)

    def drift(self) -> None:
        """Move every node's random background load one small step, staying inside BACKGROUND."""
        lo, hi = BACKGROUND
        for i in NODES:
            b = self.bg[i] + self.noise.gauss(0.0, DRIFT)
            self.bg[i] = min(max(2 * lo - b if b < lo else 2 * hi - b if b > hi else b, lo), hi)

    def budget(self, i: str, t: int) -> float:
        """CPU cycles (Mc) node i has in slot t."""
        n = NODES[i]
        return (CAPACITY[n.kind] * (TN_FACTOR if n.n == 4 else 1.0)
                * (1 - self.bg[i]) * (1 - self.busy(i, t)))

    def slot(self, t: int, Y: dict) -> dict:
        """Apply the allocation Y[t]; return the telemetry of slot t."""
        gamma = self.rng.randint(*self.gamma)
        Q0 = {i: self.backlog(i) for i in NODES}                 # backlog at the start
        self.queue[ROOT].extend(Job(self.rng, t) for _ in range(gamma))

        # jobs a node can process per slot now: its cycles / mean cycles per task waiting
        self.drift()
        budget = {i: self.budget(i, t) for i in NODES}
        cap = {}
        for i, n in NODES.items():
            tasks = sum(self.size(i, j) for j in self.queue[i])
            p = (sum(self.cycles(i, j) for j in self.queue[i]) / tasks if tasks
                 else P_MEAN[STAGE[n.kind]])
            cap[i] = budget[i] / p

        # service; what a node forwards reaches its child in the NEXT slot
        arrivals = {i: [] for i in NODES}
        done, used = [], {}
        timing = {i: [0, 0.0, 0.0] for i in NODES}               # jobs finished, their queueing, processing
        sent = {e: [0, 0] for e in self.owed}                    # [jobs, tasks delivered]
        for i in NODES:
            kids = children(i)
            B = sum(Y[i, j] for j in kids)                       # this slot's allocation
            if B > 0:
                self.x[i] = [Y[i, j] / B for j in kids]          # kept as it is while B = 0
            self.allow[i] = min(self.allow[i] + B, B + max(SESSIONS))
            work = budget[i]                                     # cycles left in this slot
            q = self.queue[i]
            while q:
                job = q[0]
                size = self.size(i, job)
                if self.allow[i] < size:
                    break                                        # the allocation does not cover it
                need = self.cycles(i, job) - self.started[i]
                if work < need:                                  # out of capacity: continue next slot
                    self.started[i] += work
                    work = 0.0
                    break
                work -= need
                self.started[i] = 0.0
                q.popleft()
                self.allow[i] -= size
                wait, proc = t - job.at, self.cycles(i, job) / budget[i]
                job.wait, job.proc, job.at = job.wait + wait, job.proc + proc, t + 1
                timing[i][0] += 1
                timing[i][1] += wait
                timing[i][2] += proc
                for k, x in zip(kids, self.x[i]):                # every child is owed its share x
                    self.owed[i, k] += x * size
                j = max(kids, key=lambda k: self.owed[i, k])     # the child most owed gets the job
                self.owed[i, j] -= size
                if j == SINK:
                    job.leaf = i
                    done.append(job)
                else:
                    arrivals[j].append(job)
                    sent[i, j][0] += 1
                    sent[i, j][1] += self.size(j, job)
            used[i] = 1 - work / budget[i]                       # share of the capacity spent
        for j, jobs in arrivals.items():
            self.queue[j].extend(jobs)

        # lambda_ij[t]: session tasks created per job the AMF forwarded in this slot.
        # One value per AMF (s belongs to the job, not to the SMF it is sent to).
        for i, n in NODES.items():
            jobs = sum(sent[i, j][0] for j in n.children)
            if jobs and n.kind == "amf":
                for j in n.children:
                    self.lam[i, j] = sum(sent[i, k][1] for k in n.children) / jobs

        cost, grad, util = self.feedback(Y, Q0, cap, gamma)
        return dict(gamma=gamma, lam=dict(self.lam), grad=grad, cost=cost, util=util, cap=cap,
                    budget=budget, timing=timing, used=used, Q={i: self.backlog(i) for i in NODES}, done=done,
                    got={j: sum(self.size(j, x) for x in jobs) for j, jobs in arrivals.items()},
                    waiting=sum(len(q) for q in self.queue.values()))

    def feedback(self, Y: dict, Q0: dict, cap: dict, gamma: int):
        """Cost of Y[t] and its gradient, as each node reports them."""
        phi, phi_p = PHI[self.cost["phi"]]
        psi, psi_p = PSI[self.cost["psi"]]
        own, capped = self.cost["load"] == "own", self.cost["drain"] == "capped"
        lam = self.lam
        B = {i: sum(Y[i, j] for j in children(i)) for i in NODES}
        util, left, drains, cost, grad = {}, {}, {}, {}, {}
        for i, n in NODES.items():
            arriving = gamma if i == ROOT else lam[n.parent, i] * Y[n.parent, i]
            util[i] = B[i] / cap[i]
            served = min(B[i], cap[i]) if capped else B[i]
            left[i] = max(Q0[i] + arriving - served, 0.0) / cap[i]
            drains[i] = 0.0 if capped and B[i] >= cap[i] else 1.0    # would more B_i drain more?
            cost[i] = cap[i] * psi(left[i])
            if own or not n.children:
                cost[i] += cap[i] * phi(util[i])
        for i in NODES:
            for j in children(i):
                if own or j == SINK:
                    grad[i, j] = phi_p(util[i])
                else:
                    offered = lam[i, j] * Y[i, j] / cap[j]
                    cost[i] += cap[j] * phi(offered)
                    grad[i, j] = lam[i, j] * phi_p(offered)
                grad[i, j] -= drains[i] * psi_p(left[i])
                if j != SINK:
                    grad[i, j] += lam[i, j] * psi_p(left[j])
        return cost, grad, util
