# HOLD_FINAL

HOLD implemented from scratch on the updated paper (System Model + Algorithm 1,
"Multi-Stage Handover"). Only HOLD: no baselines, no latency model, no Kingman
formula, no CPU quotas.

## Status now (15 Nov 2026)

* Steps 1–4 are done; step 5 (cost, tuning, results) is done except choosing the final paper
  figures.
* All experiments were run on the current environment (random capacity, TN +10 %, four busy
  periods, cost `own / power4 / quad / plain`): `sim.py`, `study.py sweep`, `study.py costs`
  and `testbed.py`. Results are in `results/sim/`, `results/study/`, `results/testbed/`.
* Last testbed run: 3200 slots, 101 627 jobs finished by the algorithm = 101 627 real Xn
  handovers on Open5GS, none failed, every session kept on its leaf.
* The Open5GS core (80 NFs) may still be running: `python3 core.py stop` when finished.

## What the code does, in simple words

**The idea.** A serving satellite must hand over a stream of UEs to four target platforms
(3 satellites and 1 TN). Each handover is a **job**, and every job must be processed by a
dispatcher, an AMF, an SMF and a UPF of the platform it is sent to. HOLD (Algorithm 1 of the
paper) decides, in every slot, how much each node processes (`B`) and how it splits that among
its children (`x`). We want to see that HOLD keeps the queues short and moves traffic away from
a node that gets congested.

**Two layers.**

| Layer | What it is | Files |
|---|---|---|
| **The model** (Python) | The tree of nodes, the jobs, the queue of every node, the CPU capacity of every node, the cost, and Algorithm 1. All decisions and all congestion happen here | `tree.py`, `env.py`, `hold.py`, `sim.py`, `study.py` |
| **The real 5G network** (Open5GS + UERANSIM_XN) | 80 real AMF, SMF and UPF processes, real gNBs and real UEs. It **executes** the decisions: every job that HOLD finishes becomes one real handover on the AMF, SMF and UPF that HOLD chose for it | `setup.sh`, `core.py`, `ran.py`, `check_paths.py`, `testbed.py` |

`topology.py` (step 1) computes the satellite geometry: which satellites can be the targets of a
handover. It describes the setting; the slot loop does not read it.

```
   THE MODEL (Python)                                   THE REAL 5G NETWORK
   ─────────────────────────────────────                ────────────────────────────────────
   env.py   jobs arrive, nodes process them,            for every job that left its UPF:
            queues grow or shrink,           ── job ──>   its UE does a real Xn handover;
            nodes report their cost                       the real AMF, SMF and UPF of that
   hold.py  Algorithm 1: new x and B                      job do the Path Switch
   sim.py   repeats this every slot          <── CPU ──  CPU time of all 80 NFs is read
                                                 (logged only, the algorithm never reads it)
```

**One slot in the model** (`sim.py` → `env.py` → `hold.py`):

1. 16–48 new jobs arrive at the root. Each job has a random number of PDU sessions `s` (1–3)
   and a random number of CPU cycles `p` it needs at each stage.
2. Every node processes jobs from its queue, limited by its allocation `B` and by its CPU
   capacity in that slot, and sends them to its children in the shares `x`. The capacity changes
   over time (random background load, scripted busy periods), so nodes get congested.
3. Every node reports its cost gradient: how close it is to its capacity, how long its own queue
   is, and how long its children's queues are.
4. Algorithm 1 uses the gradient and the virtual queues `Z` to compute the next `x` and `B`.

`sim.py` runs only this. `study.py` runs it many times with other parameters or other costs.

**The role of Open5GS** (what `testbed.py` adds to the same loop):

1. **Before the run.** `core.py` starts 8 AMFs, 24 SMFs and 48 UPFs as real Open5GS processes.
   `testbed.py` attaches 144 real UEs (UERANSIM_XN): one for every UPF leaf and every `s` = 1, 2,
   3. So for every (leaf, `s`) there is a real UE whose `s` PDU sessions sit on exactly that
   AMF → SMF → UPF.
2. **Every slot.** The model runs exactly as in `sim.py`. The jobs that leave their UPF in this
   slot are sent to Open5GS: for a job with leaf `L` and `s` sessions, the UE of (`L`, `s`) does
   an Xn handover between the two gNBs of its AMF. This makes the real AMF process a Path Switch,
   the real SMF update each of the `s` sessions, and the real UPF switch each session's tunnel:
   the AMF → SMF → UPF processing that the job stands for.
3. `testbed.py` waits until all handovers of the slot are complete, reads the CPU time each of
   the 80 NF processes used, and goes to the next slot.
4. **After the run.** It checks that every NF still holds exactly its UEs and sessions, fits the
   CPU of one handover per stage, and plots the real CPU of the NFs (`testbed.png`).

**What Open5GS gives back.**

* **Proof that every decision is executable:** each job HOLD finishes is one completed real
  handover, with the right number of sessions, on the AMF, SMF and UPF HOLD chose.
* **The real load of the NFs:** when HOLD moves traffic away from a busy node, the real CPU of
  that AMF, SMF or platform drops, and comes back afterwards.
* **The CPU cycles of a handover at each stage:** measured once on Open5GS and used as the job's
  `p` in `env.py`. This is the only number that goes from Open5GS into the model.

**What Open5GS does not do.**

* It does not decide anything, and the algorithm reads nothing from it during a run. So
  `testbed.py` and `sim.py` give the same `x`, `B`, queues and cost.
* It does not create the congestion. The real NFs have no CPU limit and a handover takes about
  0.3 ms of CPU, so they never queue. Capacity, background load, busy periods and queues are in
  `env.py`.
* Its handover time (about 18 ms) is not the model's time in system, which is counted in slots.

## The model we implement (from the paper)

| Symbol | Meaning |
|---|---|
| tree | root `i = 0` (source dispatcher on the serving satellite) → 4 target dispatchers → AMF → SMF → UPF (leaves). One parent per node. |
| `γ_0[t]` | jobs injected at the root in slot `t`, `0 ≤ γ_0[t] ≤ γ_max` |
| job | `⟨p_disp, p_amf, p_smf, p_upf, s⟩`: CPU cycles needed at each stage, and `s` = number of PDU sessions the job unpacks into at AMF → SMF |
| `B_i[t]` | processing and forwarding allocation of node `i`, `0 ≤ B_i ≤ B_max` (chosen number, no quota) |
| `x_i[t]` | routing vector of node `i` over its children, `Σ_j x_ij = 1`, `x_ij ≥ ρ` |
| `Y_ij[t]` | `B_i[t]·x_ij[t]`, work forwarded from `i` to child `j` |
| `λ_ij[t]` | stage-transition factor, 1 except AMF → SMF |
| `g_i^t(Y)` | `γ_i[t] + Σ_parents λ_ji·Y_ji − Σ_children Y_ij` |
| `Z_i[t]` | virtual queue: `Z_i[t+1] = max{ −g_i^t(Y[t+1]), Z_i[t] + g_i^t(Y[t+1]) }`, `Z_i[1] = 0` |
| `L_t` | convex penalty of `Y = B·x`; its gradient is reported straight to the root every slot |

Algorithm 1, every slot, every node, with `c_ij = V·∂L_t/∂Y_ij + λ_ij·Z_j − Z_i`:

    x_ij[t+1] ∝ x_ij[t]·exp(−c_ij / α)          then keep x_ij ≥ ρ, Σ_j x_ij = 1     (D_x = KL)
    s_i       = Σ_j c_ij·x_ij[t+1] + α·KL(x_i[t+1] ‖ x_i[t])
    B_i[t+1]  = clip(B_i[t] − s_i / α, 0, B_max)                                      (D_B = ½(B − B[t])²)

A leaf has one child, a dummy sink (`x = 1`, `Y = B`). Two queues per node, not to be confused:

* **physical queue** `Q_i`: the FIFO of real jobs waiting at the node (Python, in `env.py`);
* **virtual queue** `Z_i`: the paper's auxiliary variable above (in `hold.py`).

Decisions (user): serving satellite (root only) + 4 targets `sat1`, `sat2`, `sat3`, `tn`; each
target has 2 AMFs, each AMF its own 3 SMFs, each SMF its own 2 UPFs (80 NFs, 85 tree nodes); no
slicing; `B_max` chosen; `D_x` = KL, `D_B` = squared distance; `B`, `Y`, `Z` and the queues are
counted **in jobs** (session tasks at SMF and UPF); `λ = s` only at AMF → SMF; the cost gradient
is sent straight to the root every slot (no relay).

## How the environment is built (`env.py`)

The environment is what Algorithm 1 acts on. It holds the jobs, the physical queues and the
CPU capacities, serves the jobs as the allocation says, and reports the cost gradient.

### One slot, in order

1. **Arrivals.** `γ_0[t]` jobs, uniform in 16–48 (mean 32), are appended to the root's queue.
2. **Capacity.** Every node's CPU budget for this slot is computed (below).
3. **Service, top-down.** Every node serves its FIFO queue as far as its allocation `B_i[t]`
   and its CPU budget allow, and splits what it forwards among its children by `x_i[t]`.
   Forwarded jobs reach the child in the **next** slot. A job served at a UPF leaves the system
   (in `testbed.py` this is when its real handover runs).
4. **`λ`.** Each AMF reports `λ[t]` = session tasks created per job it forwarded in this slot.
5. **Feedback.** Each node reports its cost and the gradient `∂L_t/∂Y_ij (Y[t])`, `λ[t]`, `γ[t]`.
6. **Algorithm 1** (`hold.py`) computes `x[t+1]`, `B[t+1]` and `Z[t+1]` from that feedback only.

A slot has no real duration; it is one decision step.

### Jobs

| Field | How it is drawn, per job |
|---|---|
| `s` | 1, 2 or 3 PDU sessions, each with probability 1/3 |
| `p_disp` | uniform 0.15–0.25 Mc (Mc = 10⁶ cycles); dispatchers are Python only, so this is chosen |
| `p_amf` | CPU measured on Open5GS for one handover with `s` sessions, `(174 + 62·s)` µs, times uniform 0.75–1.25 |
| `p_smf` | per session task: `(118 + 80·s)/s` µs, times uniform 0.75–1.25 |
| `p_upf` | per session task: `(74 + 45·s)/s` µs, times uniform 0.75–1.25 |

Microseconds are turned into cycles at an assumed 3.2 GHz (the VM has no cycle counter; the
factor scales `p` and `C` together, so it changes no result). Mean `p`: dispatcher 0.20 Mc per
job, AMF 0.95 Mc per job, SMF 0.45 Mc and UPF 0.26 Mc per session task.

**Units.** Root, dispatchers and AMFs count jobs. At AMF → SMF a job unpacks into `s` session
tasks, so SMF and UPF queues and allocations count tasks. The `s` tasks of a job stay together
on one SMF and one UPF (one job = one UE with `s` sessions).

### CPU capacity

    budget_i[t] = C_i · TN · (1 − background_i[t]) · (1 − busy_i[t])          (Mc in slot t)
    cap_i[t]    = budget_i[t] / (mean p of the tasks waiting at i)            (jobs or tasks per slot)

| Part | Value |
|---|---|
| `C_i` (Mc per slot) | root 12.8, dispatcher 3.2, AMF 7.6, SMF 2.4, UPF 0.7 |
| `TN` | 1.10 for every TN node (dispatcher, AMFs, SMFs, UPFs), 1 for satellites |
| `background_i[t]` | random share of the capacity taken by other work: starts uniform in 0–0.5, then moves each slot by a normal step of standard deviation 0.02, reflected at 0 and 0.5. Independent for every node, from its own random stream (so the jobs do not change when it changes) |
| `busy_i[t]` | scripted busy periods (`EVENTS`), one per kind of node: AMF `sat1.a0` −80 % in slots 400–800, SMF `sat2.a0.s1` −70 % in 1100–1500, the whole satellite `sat3` (all 20 NFs and its dispatcher) −50 % in 1800–2200, the whole TN −50 % in 2500–2900 |

So capacity is not static: TN and satellite capacities vary over ranges that overlap (AMF:
satellites 3.8–7.6 Mc per slot, TN 4.2–8.4), and congestion also appears on its own. The
capacities give 50 % utilisation at mean load without background; with the background the
nodes use 61–72 % of their CPU on average.

### How a node serves (the job-level realisation of `B` and `x`)

* **Allocation.** A node may serve up to `B_i[t]` jobs (tasks) in slot `t`; fractions add up
  from slot to slot (at most 3 units are carried over).
* **CPU.** It may spend at most `budget_i[t]` cycles. A job too large for the rest of the slot
  is started and finished in the next slot; unused capacity is lost, never saved.
* **Order.** FIFO.
* **Split.** Every served job is sent to the child that is owed most by the shares `x_i[t]`
  (largest-remainder rounding), so the real split follows `x` (checked: within 0.02).

`B` and `Y` are allocations, not physical departures (the paper). An allocation above the
capacity is allowed; the node simply cannot do more than its CPU.

### What is measured for every job at every node

* **Queueing time:** slots the job waited before the slot in which the node finished it.
* **Processing time:** the share of a slot the node needed for it, `cycles / budget`.

A job moves one stage per slot, so time in system = 4 slots + its queueing at the 5 nodes.
This is the model's time, not an Open5GS measurement.

## The cost that is fed back to the root

**What the paper fixes.** `L_t = Σ_i L_t^i(Y_i)` is a convex, differentiable, "delay related"
penalty of node `i`'s forwarding allocation, with "sharp penalty increases as an allocation
approaches a service-capacity threshold". It is unknown in advance; after slot `t` the gradient
`∂L_t/∂Y_ij (Y[t])` is reported, and Algorithm 1 uses nothing else from the environment (plus
`λ[t]` and `γ[t]`). The paper gives no formula, so the cost is ours to choose.

**The cost in use** (`COST` in [env.py](env.py) = `own / power4 / quad / plain`):

    L_t(Y) =  Σ_i  cap_i·φ( B_i / cap_i )       what every node processes, against its capacity
            + Σ_i  cap_i·ψ( R_i / cap_i )       backlog every node is left with

    R_i   = [ Q_i[t] + g_i^t(Y) ]+              physical queue + what is sent in − B_i
    φ(u)  = u⁴/4          almost free below the capacity (u = 1), sharp near and above it
    ψ(r)  = r²/2          r = backlog measured in slots of work

    ∂L_t/∂Y_ij = (B_i / cap_i)³  −  R_i / cap_i  +  λ_ij·R_j / cap_j
                 node i gets close   node i drains   child j's backlog
                 to its capacity     its backlog     grows

| Part | From | Role |
|---|---|---|
| `Y_ij`, `B_i`, `λ_ij`, `γ`, `g_i^t(Y)` | the paper | the variables of the cost |
| `cap_i = C_i / p_i` | the job definition (`p`, cycles) and the node capacity (`C`, cycles per slot) | turns cycles into "jobs per slot"; a heavy job or a busy node gives a small `cap_i` |
| first sum, `φ` | of the paper's form `Σ_i L_t^i(Y_i)`: each term depends on node `i`'s own allocation only | keeps `B_i` at what node `i` can do: when a node gets busy its `B` falls with its capacity |
| second sum, `ψ`, `Q_i` | new | brings the physical queue into the cost. Its gradient is a difference of physical queues, the same shape as the paper's virtual-queue weight `λ_ij·Z_j − Z_i`; it tells a parent which child is congested |

`own` against the paper's wording `child` (each edge pays for the load it offers to its child):
with `child`, a congested AMF or SMF keeps its `B` (4.5 before, 4.7 during, 4.6 after, with a
capacity of 1.1); with `own` its `B` follows its capacity (4.6 → 1.7 → 4.8), the time in system
is 5 % shorter and the shifts of `x` and of the queues are the same. Switched to `own` on
15 Nov 2026 (user decision); `child` stays available.

**The options** (all kept in `env.py`; `COST` picks one of each):

| Choice | Options |
|---|---|
| `phi` (processing) | `quad` u²/2 · `quartic` u²/2 + u⁴/4 · `steep` u²/2 + u⁸/8 · `mm1` u/(1−u) (the classical delay formula, extended past u = 0.9) · `power4` u⁴/4 · `power8` u⁸/8 |
| `psi` (queueing) | `none` · `linear` r · `quad` r²/2 · `cubic` r³/3 |
| `load` (what `φ` is applied to) | `own`: what each node processes itself, `B_i / cap_i` · `child`: what each edge offers to its child, `λ_ij·Y_ij / cap_j` (the paper's wording) |
| `drain` | `plain`: `R_i` counts `B_i` as served · `capped`: counts `min(B_i, cap_i)`, so allocating beyond the capacity drains nothing (has a kink at `B_i = cap_i`) |

## Experiments: commands and expected output

All commands run from `~/HOLD/HOLD_FINAL`. No command-line options: each file has its
constants at the top. Run one heavy command at a time (the machine has 4 cores and 17 GB disk).

### 0. After every reboot (once)

```bash
sudo ./setup.sh           # stops the stock AMF/SMF/UPF, creates the 48 UPF tun devices
```

Expected: `stock AMF/SMF/UPF stopped, 48 UPF devices ready (owner amirndr)`.

### 1. Topology — `python3 topology.py` (no core needed)

Starlink orbits (SGP4) over Montreal for 90 min, the serving chain, the handover events, 3
candidate satellites + TN per event, checked against skyfield.
Output: `results/topology/events.csv`, `timeline.csv`, `serving.png`.
Expected: 39 serving satellites, 38 handover events (one every ~139 s), altitude 527–577 km,
elevation within 0.03° of skyfield, all checks `OK`.

### 2. Tree — `python3 tree.py` (instant)

Prints the 85-node tree. Expected: `total 85 nodes, 48 leaves` and `every non-root node has
exactly one parent`.

### 3. Open5GS core — `python3 core.py start | status | stop` (start: about 1 min)

Writes the 80 configs, provisions 240 subscribers, starts the 48 UPFs, 24 SMFs and 8 AMFs as
user processes. Expected after `start`: a table with `2/2 6/6 12/12` per platform and
`total 80/80 processes, ~1.5 GB`. `stop` prints `stopped (0 left)`.

### 4. Path check — `python3 check_paths.py` (a few minutes, core must be up)

For each AMF: 6 UEs with 3 sessions, checks from the NFs' own counters that the sessions land on
exactly the addressed AMF/SMF/UPF, hands every UE over and back.
Output: `results/paths/paths.csv`, `handovers.csv`, `handover.png`.
Expected last line: `OK   48/48 leaves verified, 96 handovers completed`.

### 5. Algorithm on the environment — `python3 sim.py` (about 35 s, no core needed)

Runs Algorithm 1 on `env.py` for 3200 slots (seed 1) and prints the checks.
Output: `results/sim/slots.csv`, `nodes.csv` and four figures:

| Figure | Shows |
|---|---|
| `overview.png` | routing shares of the root, of `sat1` and of `sat2.a0`; `B` and queue of the four busy nodes; total queue per stage; flow violation; cost |
| `events.png` | one row per busy period: capacity, `x`, `B`, queue and cost of the busy node and its siblings |
| `dynamics.png` | the whole run for the 4 platforms, the AMFs of `sat1` and the SMFs of `sat3.a0`: capacity, `x`, `B`, queue, queueing time, processing time, cost |
| `response.png` | the average congestion episode at each level of the tree (see "How HOLD reacts to congestion") |

Expected printout (the numbers of the current run):

    slots 3200   jobs in 101810   jobs done 101627 (99.8%)   ...   -> jobs conserved: yes
    mean cost per slot 305.40   mean backlog 326   mean time in system 6.44 slots = 4 + 2.44 of queueing
    capacity ... max 1.00 at every stage                      (CPU spent never above the budget)
    routing  largest gap between a child's real share of the jobs and its share x: 0.019
    flow constraint [sum_t g_i]+ / T at T: 0.000-0.008        (must go to 0)
    traffic shift during each busy period:
      sat1.a0    share at sat1    0.51 -> 0.22 -> 0.53
      sat2.a0.s1 share at sat2.a0 0.34 -> 0.19 -> 0.29
      sat3       share at root    0.26 -> 0.13 -> 0.25
      tn         share at root    0.26 -> 0.13 -> 0.27

### 6. Parameter sweep — `python3 study.py sweep` (about 15 min, no core needed)

`V`, `α`, `ρ`, `B_max` and the load, one at a time, 3 seeds each (93 runs on 4 cores).
Output: `results/study/sweep.csv` (one row per run), `sweep_<parameter>.png` and
`sweep_<parameter>_series.csv`. Each figure: top row = `x`, `B` and queue of the busy AMF and
the backlog of all nodes over time, one curve per value; two rows below = eight metrics
(completion, time in system, backlog, busy-node queue peak, cost, `Z`, flow violation, routing
change) against the parameter. Expected numbers: see "How the key parameters change the
behaviour".

### 7. Cost comparison — `python3 study.py costs` (about 15 min, no core needed)

All 84 combinations of the cost options, same run (seed 1).
Output: `results/study/costs.csv`, `costs.png` (time in system against busy-node queue peak,
one dot per option). Expected: the option in use is first by time in system (6.44 slots); see
"Cost comparison".

### 8. Real execution on Open5GS — `python3 testbed.py` (about 17 min, core must be up)

Attaches 144 UEs (one per leaf and per `s`), measures the idle CPU of the 80 NFs for 30 s,
then runs the same 3200 slots as `sim.py`; every job that leaves its UPF is executed as a real
Xn handover of that UE on its AMF → SMF → UPF chain. Log files are emptied every 200 slots.
Output in `results/testbed/`: `handovers.csv` (one row per handover), `cpu.csv` (CPU of every
NF in every slot), `testbed.png`, and the same `slots.csv`, `nodes.csv`, `overview.png` as
`sim.py`. Expected printout: the `sim.py` printout, then

    after the attach: every AMF holds 18 UEs, every SMF 12 sessions, every UPF 6 sessions
    101627 handovers in ~820 s
    after the run: every AMF holds 18 UEs, every SMF 12 sessions, every UPF 6 sessions
    jobs finished by the algorithm 101627   real handovers completed 101627   with a wrong number of sessions 0
    handover time: median ~18 ms   95% ~41 ms
    measured CPU per handover   AMF ~176 + 66·s   SMF ~112 + 82·s   UPF ~72 + 22·s   (µs)
    real CPU of the busy node among its siblings
      sat1.a0 0.51 -> 0.25 -> 0.54,  sat2.a0.s1 0.34 -> 0.20 -> 0.29,  sat3 0.26 -> 0.14 -> 0.25,  tn 0.26 -> 0.15 -> 0.27

When finished: `python3 core.py stop`.

## Results

### How HOLD reacts to congestion (`results/sim/`)

What `events.png` shows (seed 1, 3200 slots), one row per busy period:

| Busy node | Routing share: before → during → after | `B`: before → during → after (its capacity during) | Its queue |
|---|---|---|---|
| AMF `sat1.a0` (−80 %) | 0.51 → 0.22 → 0.53 | 4.6 → 1.7 → 4.8 (1.1) | 4 before, 6–10 during with peaks of 14, 4 after |
| SMF `sat2.a0.s1` (−70 %) | 0.34 → 0.19 → 0.29 | 2.9 → 1.5 → 2.7 (1.5) | peak 11 at the start, then below its siblings' |
| satellite `sat3` (−50 %) | 0.26 → 0.13 → 0.25 | 9.1 → 4.3 → 8.7 (6.1) | peak 22 at the start, then about 5; a sibling that takes the traffic while its own capacity is low peaks at 50 |
| TN (−50 %) | 0.26 → 0.13 → 0.27 | 9.2 → 4.5 → 9.5 (6.3) | peak 34 at the start, then about 5 |

`response.png` is an average, not one node: for each level of the tree it takes every moment in
the run when some child's queue rose into the top 5 % above its siblings' (scripted busy periods
left out; hundreds of such moments, at all the nodes of that level), lines them up at that
moment, and averages its queue, share, `B`, queueing time and cost against the mean of its
siblings, from 20 slots before to 60 after. It shows what happens, typically, around a
congestion that appears on its own:

| Parent → children | Episodes | Queue above siblings: at the start, +5, +10 slots | Share of the congested child: 20 slots before, at the start, lowest after, +40 slots |
|---|---|---|---|
| root → platforms | 16 | +3.0, +1.9, +2.1 | 0.271, 0.278, 0.268 (after 25 slots), 0.268 |
| dispatcher → AMFs | 51 | +7.2, +6.8, +3.9 | 0.495, 0.543, 0.475 (after 9 slots), 0.500 |
| AMF → SMFs | 705 | +5.7, +1.0, 0.0 | 0.348, 0.352, 0.300 (after 5 slots), 0.336 |
| SMF → UPFs | 1377 | +5.4, +0.9, +0.2 | 0.540, 0.568, 0.525 (after 9 slots), 0.532 |

The sequence is the same at every level: the child's share drifts up, its queue, queueing time
and cost rise, the algorithm cuts its share below the normal value, the queue comes back towards
the siblings' level, and the share returns. Reading note: episodes are selected at a queue peak,
so part of the queue's fall would happen anyway; the share going below its normal value is the
algorithm's reaction.

Time per stage (per job): queueing 0.26 / 0.27 / 0.43 / 0.58 / 0.90 slots at root / dispatcher /
AMF / SMF / UPF; processing 0.02 / 0.09 / 0.18 / 0.51 / 1.02 of a slot.

### How the key parameters change the behaviour (`results/study/sweep_*.png`)

Mean of 3 seeds; the default value in bold.

| Parameter | Values | What happens | Expected from the paper |
|---|---|---|---|
| `V` | 0.5, 1, 2, **4**, 8, 16, 32 | cost per slot 1712, 753, 412, **349**, 414, 589, 1017; busy-node queue peak 126, 60, 44, **34**, 35, 47, 56; time in system 8.19, 6.15, 6.24, **6.76**, 8.43, 11.5, 16.7; routing change per slot 0.005 → 0.139 | drift-plus-penalty trade-off: too small a `V` and the cost signal is too weak, queues grow at busy nodes; too large and backlog and jitter grow. `V = 4` gives the lowest cost and the lowest busy-node queue; `V = 1–2` gives a 10 % shorter time in system with higher peaks |
| `α` | 10, 30, **100**, 300, 1000 | routing change per slot 0.082, 0.020, **0.008**, 0.005, 0.004; busy-node peak 33, 26, **34**, 56, 107; time in system 7.64, 6.75, **6.76**, 8.07, 17.0 | step size of the proximal update: small = fast but jittery, large = smooth but slow to leave a busy node |
| `ρ` | 0.005, **0.02**, 0.05, 0.1, 0.2, 0.25 | no effect up to 0.1 (time 6.73–6.81); at 0.2 the share of a busy node cannot go below 0.20: busy-node peak 485, time 10.1; at 0.25 (= 1/4) the root cannot move at all: peak 1050, time 21.6 | `ρ` only keeps the shares away from zero; if it is large it blocks the shift |
| `B_max` | 16, 24, 32, 48, **64**, 96 | completion 49.8, 74.6, 99.3, 99.8, **99.8**, 99.8 %; time 843, 431, 21.3, 6.77, **6.76**, 6.76 | `B_max` must exceed the offered load (32 jobs per slot at the root); above it, it has no effect |
| load | 0.5, 0.75, **1**, 1.1, 1.2, 1.3, 1.5 × (16–48 jobs per slot) | completion 99.9, 99.9, **99.8**, 99.6, 98.6, 95.6, 85.4 %; time in system 4.57, 5.01, **6.76**, 11.7, 27.7, 80.5, 256; busy-node peak 7, 14, **34**, 80, 208, 494, 1607 | stable inside the capacity region; with the random background load (25 % of the capacity on average) and the busy periods, the region ends between 1.1× and 1.2× |

Flow violation `max_i [Σg_i]+ / T` stays below 0.03 in every stable run.

### Cost comparison (`results/study/costs.csv`, `costs.png`)

All 84 combinations, `V = 4`, seed 1, 3200 slots. "Slots to half of the shift" is for the 4
busy periods.

| `load` | `phi` | `psi` | `drain` | Time in system (slots) | Largest queue anywhere | Largest queue of a busy node | Slots to half of the shift | Routing change per slot | Slots with `B > cap` | Jobs completed |
|---|---|---|---|---|---|---|---|---|---|---|
| **own** | **power4** | **quad** | **plain** | 6.44 | 74 | 34 | 2, 2, 19, 2 | 0.0090 | 33.9 % | 99.8 % |
| child | power4 | quad | plain | 6.78 | 87 | 35 | 0, 0, 16, 0 | 0.0089 | 33.6 % | 99.8 % |
| own | quad | cubic | plain | 6.82 | 85 | 21 | 0, 1, 15, 1 | 0.0245 | 53.9 % | 99.8 % |
| own | quad | quad | plain | 6.92 | 79 | 46 | 2, 2, 21, 5 | 0.0108 | 43.8 % | 99.8 % |
| own | power4 | cubic | plain | 6.93 | 103 | 22 | 0, 0, 13, 1 | 0.0253 | 42.2 % | 99.8 % |
| own | power4 | quad | capped | 7.40 | 97 | 35 | 1, 2, 17, 2 | 0.0095 | 24.2 % | 99.8 % |
| own | power8 | quad | plain | 7.66 | 96 | 34 | 1, 1, 15, 2 | 0.0105 | 26.9 % | 99.8 % |
| own | quartic | quad | plain | 9.51 | 112 | 44 | 3, 1, 19, 3 | 0.0143 | 20.0 % | 99.7 % |
| own | steep | quad | plain | 9.59 | 116 | 28 | 2, 1, 13, 2 | 0.0115 | 19.1 % | 99.7 % |
| own | power4 | linear | plain | 10.73 | 152 | 152 | 7, 7, 30, 9 | 0.0115 | 30.2 % | 99.8 % |
| own | power4 | none | plain | 14.62 | 143 | 143 | 5, 7, 30, 10 | 0.0056 | 18.7 % | 99.6 % |
| own | quad | none | plain | 57.57 | 565 | 558 | 0, 15, 49, 24 | 0.0223 | 13.0 % | 96.9 % |
| own | mm1 | quad | plain | 93.80 | 518 | 276 | 0, 0, 0, 0 | 0.0836 | 9.9 % | 95.2 % |
| child | mm1 | quad | plain | 222.87 | 1255 | 679 | 0, 0, 0, 0 | 1.1799 | 11.2 % | 89.8 % |

Reading:

* **The cost in use is the best of the 84 by time in system** (6.44 slots).
* **`φ = u⁴/4` is the best processing cost.** Almost free below the capacity, it lets `B` follow
  the arrivals; `u²/2 + u⁴/4` charges for every unit of `B`, so a node leaves work waiting rather
  than pay for it (9.51 against 6.44 slots).
* **`ψ` is needed and must grow faster than linearly.** With none or a linear one the largest
  queue of a busy node is four times higher (143–152 against 34) and the time in system 1.7–2.3
  times longer. `ψ = r³/3` gives lower busy-node peaks (21–22) but 3 times more routing jitter.
* **`load: child` against `own`:** `child` reacts to a busy child in the same slot (0 slots
  against 2), but its `B` does not follow a busy node's capacity and its time is 5 % longer.
* **`mm1` (the classical `u/(1−u)`) is too cautious** at this `V`: it holds the utilisation down
  and lets the backlog grow.
* **Slots with `B > cap`:** an allocation above the capacity is not an error (the paper: `B` is
  not a physical departure), and the node still never spends more CPU than it has.
* Earlier, with constant capacity, the cost in use was also the least sensitive to `V` (time in
  system 4.32, 4.25, 4.25, 4.57, 5.48 for `V` = 1, 2, 4, 8, 16); the plain `u²/2` was unstable
  for `V ≤ 2`.

### What Open5GS adds (`results/testbed/`)

How Open5GS is used is explained in "What the code does, in simple words". The curves of
`testbed.py` (cost, queues, `x`, `B`, `Z`) are the same as those of `sim.py`; the numbers below
are what Open5GS adds:

1. **The decisions are executable on a real 5G core.** Every job the algorithm finishes is one
   real Xn handover of a UE with `s` PDU sessions, through exactly the AMF, SMF and UPF the
   routing vectors chose. Five full runs: 83 284, 82 827, 82 827, 82 858 jobs (2600 slots,
   14 Nov) and 101 627 jobs (3200 slots, 15 Nov), the same number of completed handovers, none
   with a wrong number of sessions.
2. **The traffic shift is visible in the real NFs.** The CPU time of the real `open5gs-amfd` /
   `smfd` / `upfd` processes follows the routing (`testbed.png`, top row): the busy node's share
   of the real CPU of its siblings is 0.51 → 0.25 → 0.54 (AMF `sat1.a0`), 0.34 → 0.20 → 0.29
   (SMF `sat2.a0.s1`), 0.26 → 0.14 → 0.25 (`sat3`), 0.26 → 0.15 → 0.27 (TN).
3. **The real `p` per stage.** CPU time of one handover with `s` sessions, in µs:

   | Stage | Run 1 (used for the calibration) | Run 2 | Run 3 | Run 4 | Run 5 (15 Nov) | Run 5 at `s` = 1, 2, 3 |
   |---|---|---|---|---|---|---|
   | AMF | `174 + 62·s` | `181 + 67·s` | `188 + 67·s` | `174 + 62·s` | `176 + 66·s` | 241, 307, 373 |
   | SMF | `118 + 80·s` | `105 + 84·s` | `118 + 84·s` | `114 + 76·s` | `112 + 82·s` | 194, 275, 357 |
   | UPF | `74 + 45·s` | `76 + 38·s` | `82 + 41·s` | `76 + 37·s` | `72 + 22·s` | 94, 115, 137 |

   AMF and SMF agree within 12 % across runs; the UPF, whose handover CPU is the smallest, is
   the noisiest (R² 0.32 in run 5, against 0.89 for the AMF and 0.84 for the SMF).

   How it is measured: the CPU time of every NF process is read from
   `/proc/<pid>/task/*/schedstat` (nanosecond resolution) after every slot. The NF's idle CPU
   (median of 15 windows of 2 s before the run) is subtracted. Then, per stage, a least squares
   fit `cpu = a·handovers + b·sessions` over all (slot, NF) pairs; pairs further than 3 robust
   standard deviations from the fit (a timer of the NF, or the VM being slow for a moment) are
   left out of the fit (13–16 %) and cut off in the CPU curves.

**What is taken from Open5GS.** The run-1 measurement is the source of `p` in `env.py` (see
"Jobs"). The real SMF and UPF work is not proportional to `s` (a 3-session job costs the SMF
1.8 times a 1-session job, not 3 times); this fixed part per handover is inside `p`, while
`λ = s` stays exact because it counts session tasks. Real congestion cannot come from Open5GS in
this setup: with no CPU quota a handover takes ~0.3 ms of CPU on an otherwise idle NF, so the
real NFs never queue. Capacity, busy periods and physical queues therefore stay in Python.

## Is the environment right? (checks printed by `sim.py`)

Three errors were found at step 5 and fixed in `env.py`:

| What was wrong | Effect | Now | Check |
|---|---|---|---|
| Unused CPU was saved without limit while a queue was not empty | a node could spend up to 15 times its capacity in one slot | at most the budget of the slot; only a job already started carries over | CPU spent / budget: max 1.00 at every stage |
| Jobs went to the child with most unused allocation | when `B` was larger than the jobs available, the real split was more extreme than `x` | jobs are split by the shares `x_ij` | largest gap between a child's real share and `x`: 0.019 |
| `λ` was measured per AMF → SMF edge | very noisy, the routing followed the noise | one `λ` per AMF and slot | flow violation → 0 |

Also printed: jobs in = jobs done + jobs still waiting (conserved); mean CPU use per stage
(0.61–0.72); CPU capacity ranges of satellites and TN per stage; queueing and processing time
per stage; and for each busy period the share of its remaining capacity the busy node really
uses (100 %, 69 %, 68 %, 71 %).

## Points for the paper (to settle with the co-author)

1. **Leaves need a cost on `B`.** The paper defines the penalty for non-leaf nodes. Without a leaf
   cost nothing stops `B_leaf` from growing, and because `Z ≥ −g`, the leaf's virtual queue
   becomes `≈ B_max − arrivals`: a false congestion signal (measured at step 3: `B_leaf` 64
   against 1.3 arriving, SMF routing collapsed to 0.02 / 0.98).
2. **The queue term couples parent and child.** `R_i` depends on the node's own `B_i` and on its
   parent's `Y`, so `Σ_i cap_i·ψ(R_i/cap_i)` is not of the form `Σ_i L_t^i(Y_i)`; the `φ` part is.
   Algorithm 1 only uses `∇L_t(Y[t])`, so the algorithm is unchanged. In the paper's form each
   `L_t^i` can treat the other node's current allocation as a time-varying parameter. Without the
   queue term the cost is exactly separable and still works, with much larger queues at busy
   nodes (see the cost comparison).
3. **The virtual queue uses `g^t(Y[t+1])`, the requirement is on `g^t(Y[t])`.** If `λ[t]` is noisy
   and the allocation reacts to it, the two differ systematically. With `λ` measured per edge
   and `α = 30` the violation `[Σg]+/T` stayed at 0.18 jobs per slot; with `λ` measured per AMF
   it goes to 0.

## Plan and progress

- [x] **Step 1 — topology.** 39 serving satellites and 38 handover events in 90 min
      (one every ~139 s); altitude 527–577 km, speed 7.6 km/s, period ~96 min;
      elevation matches skyfield within 0.03°; every event has 3 candidates above 25°
      with ISL line of sight.
- [x] **Step 2 — tree, instances, gNBs.** 80/80 NFs up as user processes (1.5 GB), every
      SMF associated with its 2 UPFs, all AMFs and SMFs registered. `check_paths.py`:
      48/48 leaves verified, 96 handovers completed, sessions stay on their leaf across the
      handover, everything released afterwards. Lesson: deregister UEs with `disable-5g`;
      after `normal` the UE re-registers at once and its new sessions leak.
- [x] **Step 3 — algorithm (on the Python environment).** `hold.py` + `env.py` + `sim.py`.
      Current numbers: 3200 slots, 101 810 jobs, 99.8 % completed, 6.44 slots in the system
      (4 + 2.44 of queueing), mean backlog 326 jobs over 85 nodes; flow violation → 0.
- [x] **Step 4 — execution on Open5GS.** `testbed.py`; last run 15 Nov: 101 627 jobs =
      101 627 real handovers, 0 failed, 0 with a wrong number of sessions, median 18 ms,
      95 % under 41 ms, 14 min for the 3200 slots; every NF holds exactly its UEs and sessions
      before and after.
- [ ] **Step 5 — cost function, tuning, results.**
      - [x] `p` calibrated from the Open5GS measurements.
      - [x] Environment audited: capacity, job split and `λ` fixed.
      - [x] All cost options kept in `env.py` and compared (84 combinations); new processing
            cost `u⁴/4`; cost in use `own / power4 / quad / plain`.
      - [x] Random capacity (TN +10 %, overlapping ranges), queueing and processing time per
            stage, a busy period for each kind of node, congestion figures.
      - [x] Sweeps of `V`, `α`, `ρ`, `B_max`, load; cost comparison; testbed run, all on the
            current environment (15 Nov).
      - [ ] Choose and style the final paper figures (with the user).

## Open points

1. **Parameters (current).** `V = 4`, `α = 100`, `ρ = 0.02`, `B_max = 64`, `T = 3200` slots,
   seed 1; the environment numbers are in "How the environment is built". `B` starts at 0, so
   the first ~100 slots are a cold start (cut from the metrics: `WARM = 150`).
2. **A testbed run stopped once** (15 Nov, 07:01): all UEs lost their radio link in the same
   second and 25 handovers never completed. No out-of-memory or disk problem was found and the
   re-run passed. Cause unknown; if it happens again, the gNB and UE logs in `logs/ran/` (they
   keep the last 200 slots) show it.
3. **Disk.** The machine has 17 GB (1.4 GB free on 15 Nov). A testbed run empties its own log
   files every 200 slots (`TRIM` in `testbed.py`); before that, the NF logs grew to 1.2 GB.
   Not deleted, waiting for the user: `~/HOLD/Hold_newarch/results` (2.7 GB, old testbed) and
   `~/5g-lab/xn_ho_latency.csv` (330 MB; the modified gNB appends one line to it per handover).
   **Memory** (4 cores, 7.7 GB): the 80 NFs take 1.5–1.9 GB, the 144 UEs 6.5 MB each, the
   16 gNBs 10 MB each.
4. **One job = one UE.** All `s` sessions of a job stay together on one leaf (one UE with
   `s ≤ 3` PDU sessions). The fluid model would let them split across SMFs; the paper leaves
   job-level rounding outside the analysis.
5. **A slot has no real duration.** The testbed executes each slot's handovers and goes on
   (about 0.25 s per slot). The handover time (ms) is logged as a sanity value only.

## Reference

### Components

| File | What it does | Run |
|---|---|---|
| [topology.py](topology.py) | Starlink orbits (SGP4), UE at Montreal, TN node, serving chain, handover events, 3 candidate satellites + TN per event | `python3 topology.py` |
| [tree.py](tree.py) | The paper's tree: 85 nodes, ids like `sat1.a0.s2.u1`. Used by everything else | `python3 tree.py` |
| [setup.sh](setup.sh) | System setup, **after every reboot**: stops the stock AMF/SMF/UPF, creates the 48 UPF devices | `sudo ./setup.sh` |
| [core.py](core.py) | Open5GS: writes the 80 configs, adds 240 subscribers, starts/stops the NFs as plain user processes | `python3 core.py start` / `status` / `stop` |
| [ran.py](ran.py) | UERANSIM_XN: gNB and UE configs, processes, and the Xn handover of one chosen UE (`trigger`, `completed`, `finish`) | (library) |
| [check_paths.py](check_paths.py) | Attaches UEs to all 48 leaves, verifies where the sessions landed, hands every UE over and back | `python3 check_paths.py` |
| [hold.py](hold.py) | **Algorithm 1**, exactly: `x` update (KL, floor `ρ`), `B` update, virtual queues `Z`. No environment knowledge | (library) |
| [env.py](env.py) | The environment: jobs, physical FIFO queues, capacities, background load, busy periods, service, `λ`, and the cost gradient fed back to the root. All cost options live here (`COST` picks one) | (library) |
| [sim.py](sim.py) | Runs `hold.py` on `env.py`; prints the checks; writes CSVs and the four figures | `python3 sim.py` |
| [study.py](study.py) | Parameter sweeps (`sweep`) and the cost comparison (`costs`) | `python3 study.py sweep` / `costs` |
| [testbed.py](testbed.py) | The `sim.py` run with every finished job executed as a real Xn handover; measures the CPU of the 80 NFs | `python3 testbed.py` |

### How the tree is mapped onto Open5GS and UERANSIM

| Tree node | Real thing | How a job is steered to it |
|---|---|---|
| root, target dispatcher | Python only (no 3GPP function) | the algorithm's `x` |
| AMF `satN.aA` | `open5gs-amfd` at `127.0.(20+N).(10+A)`, own tracking area `10N+A` | the UE attaches through that AMF's gNBs |
| SMF `…sK` | `open5gs-smfd` at `127.0.(20+N).(20+3A+K)` | registers for its parent AMF's tracking area and its own UPFs' DNNs |
| UPF `…uU` | `open5gs-upfd` at `127.0.(20+N).(40+6A+2K+U)`, device `hfNAKU` | the DNN `d<K><U>a/b/c` (3 DNNs = up to 3 PDU sessions of one UE on one leaf) |

* No slicing: one S-NSSAI (sst 1) everywhere.
* The stock NRF, SCP, UDM, UDR, AUSF, PCF, NSSF, BSF are reused. The installed Open5GS is 2.7.7.
* gNBs: the modified UERANSIM sends the Path Switch to a gNB's first connected AMF, so each AMF
  gets two gNB **processes** of its own (16 in total). A handover moves a UE from one of the two
  to the other, so the same UE can be handed over again and again.
* A gNB numbers the UEs in the order it first hears them and keeps the number. Both gNBs are
  started first and the UEs one at a time, so each UE has a known id on each gNB; that is how
  one chosen UE is handed over.
* Testbed pool: one UE per leaf and per `s` ∈ {1,2,3} = 144 UEs, 288 PDU sessions. The UE for
  AMF number `g`, SMF `k`, UPF `u`, `s` sessions is subscriber `6·(3g + s − 1) + 2k + u`.
* Open5GS allows 4 DNNs per subscriber slice, so a subscriber is tied to one leaf position:
  subscriber `n` reaches SMF `(n % 6) // 2`, UPF `n % 2` under whichever AMF it attaches to.
* UERANSIM_XN is built in `/home/amirndr/UERANSIM_XN/build`. Its gNB appends one line per
  handover to `/home/amirndr/5g-lab/xn_ho_latency.csv` (hard-coded in the fork); we read the
  gNB log instead.
* Logs: `logs/core/<node>.log`, `logs/ran/<gnb or ue>.log`.

### topology.py constants

| Constant | Value | Meaning |
|---|---|---|
| `UE_LAT/LON` | Montreal | UE position |
| `TN_LAT/LON` | ~5 km from the UE | terrestrial base station |
| `SHELL_KM` | 500–600 | Starlink shell used |
| `WINDOW_S`, `STEP_S` | 90 min, 5 s | simulated time and sampling |
| `ELEV_TRIGGER` | 25° | serving satellite below this → handover event |
| `ELEV_MIN` | 10° | a satellite is unusable below this |
| `N_CAND` | 3 | candidate target satellites per event |
