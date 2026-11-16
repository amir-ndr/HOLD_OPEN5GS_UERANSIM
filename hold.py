#!/usr/bin/env python3
"""
Algorithm 1 of the paper ("Multi-Stage Handover"), nothing else.

State per node i:  routing vector x_i, allocation B_i, virtual queue Z_i.
Every slot, from the feedback of slot t (cost gradient, lambda, arrivals):

    c_ij      = V * dL_t/dY_ij (Y[t]) + lambda_ij[t] * Z_j[t] - Z_i[t]

    x_i[t+1]  = argmin over X_i of  sum_j c_ij x_ij + alpha * KL(x_i || x_i[t])
              = x_ij[t] * exp(-c_ij / alpha), normalised, with x_ij >= rho

    B_i[t+1]  = argmin over [0, B_max] of
                B * ( sum_j c_ij x_ij[t+1] + alpha * KL(x_i[t+1] || x_i[t]) )
                + alpha/2 * (B - B_i[t])^2
              = clip( B_i[t] - (...) / alpha, 0, B_max )

    Z_i[t+1]  = max{ -g_i^t(Y[t+1]),  Z_i[t] + g_i^t(Y[t+1]) }
    g_i^t(Y)  = gamma_i[t] + sum_parents lambda_ji[t] Y_ji - sum_children Y_ij

A leaf has one child, the dummy sink: x = 1, Y = B, lambda = 1, Z_sink = 0.
B, Y, Z and gamma are counted in jobs.
"""
import math

from tree import NODES, ROOT

SINK = "sink"


def children(i: str) -> list[str]:
    return NODES[i].children or [SINK]


def kl(x: list[float], y: list[float]) -> float:
    return sum(a * math.log(a / b) for a, b in zip(x, y))


def x_update(x: list[float], c: list[float], alpha: float, rho: float) -> list[float]:
    """Exact minimiser of <c, x> + alpha * KL(x || x_old) over {sum x = 1, x >= rho}."""
    low = min(c)
    w = [a * math.exp(-(b - low) / alpha) for a, b in zip(x, c)]
    floor = set()
    while True:                     # KKT: x_j = max(rho, k * w_j), k fixed by sum x = 1
        k = (1 - rho * len(floor)) / sum(v for j, v in enumerate(w) if j not in floor)
        new = {j for j, v in enumerate(w) if j not in floor and k * v < rho}
        if not new:
            return [rho if j in floor else k * v for j, v in enumerate(w)]
        floor |= new


class Hold:
    def __init__(self, V: float, alpha: float, rho: float, b_max: float):
        self.V, self.alpha, self.rho, self.b_max = V, alpha, rho, b_max
        self.x = {i: [1 / len(children(i))] * len(children(i)) for i in NODES}
        self.B = {i: 0.0 for i in NODES}
        self.Z = {i: 0.0 for i in NODES}          # Z_i[1] = 0

    def Y(self) -> dict[tuple[str, str], float]:
        return {(i, j): self.B[i] * xj for i in NODES for j, xj in zip(children(i), self.x[i])}

    def step(self, grad: dict, lam: dict, gamma: float) -> None:
        """grad[(i, j)] = dL_t/dY_ij at Y[t]; lam[(i, j)] = lambda_ij[t]; gamma = gamma_0[t]."""
        Z, a = self.Z, self.alpha
        x_new, B_new = {}, {}
        for i in NODES:
            c = [self.V * grad[i, j] + lam.get((i, j), 1.0) * Z.get(j, 0.0) - Z[i]
                 for j in children(i)]
            x_new[i] = x_update(self.x[i], c, a, self.rho)
            slope = sum(cj * xj for cj, xj in zip(c, x_new[i])) + a * kl(x_new[i], self.x[i])
            B_new[i] = min(max(self.B[i] - slope / a, 0.0), self.b_max)
        self.x, self.B = x_new, B_new

        # virtual queues: the NEW allocation on the flow function of slot t
        for i, node in NODES.items():
            g = (gamma if i == ROOT else 0.0) - B_new[i]
            if node.parent is not None:
                p = node.parent
                g += lam.get((p, i), 1.0) * B_new[p] * x_new[p][NODES[p].children.index(i)]
            self.Z[i] = max(-g, Z[i] + g)
