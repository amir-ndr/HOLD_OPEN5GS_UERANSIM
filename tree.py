#!/usr/bin/env python3
"""
The processing tree of the paper: G = (I, E), one parent per node.

    root (source dispatcher on the serving satellite, i = 0)
      -> 4 target dispatchers   sat1, sat2, sat3, tn
      -> 2 AMFs per target
      -> 3 SMFs per AMF
      -> 2 UPFs per SMF          (leaves)

Node ids read as the path: "sat1.a0.s2.u1" is UPF 1 of SMF 2 of AMF 0 on sat1.

    python3 tree.py        # print the tree and its counts
"""
from dataclasses import dataclass, field

PLATFORMS = ["sat1", "sat2", "sat3", "tn"]
N_AMF, N_SMF, N_UPF = 2, 3, 2
ROOT = "root"


@dataclass
class Node:
    id: str
    kind: str                      # root | disp | amf | smf | upf
    parent: str | None
    children: list = field(default_factory=list)
    n: int | None = None           # platform number 1..4
    a: int | None = None           # AMF index inside the platform
    k: int | None = None           # SMF index inside the AMF
    u: int | None = None           # UPF index inside the SMF


def build() -> dict[str, Node]:
    nodes = {ROOT: Node(ROOT, "root", None)}

    def add(id_, kind, parent, **ix):
        nodes[id_] = Node(id_, kind, parent, **ix)
        nodes[parent].children.append(id_)

    for n, p in enumerate(PLATFORMS, 1):
        add(p, "disp", ROOT, n=n)
        for a in range(N_AMF):
            amf = f"{p}.a{a}"
            add(amf, "amf", p, n=n, a=a)
            for k in range(N_SMF):
                smf = f"{amf}.s{k}"
                add(smf, "smf", amf, n=n, a=a, k=k)
                for u in range(N_UPF):
                    add(f"{smf}.u{u}", "upf", smf, n=n, a=a, k=k, u=u)
    return nodes


NODES = build()


def of_kind(kind: str) -> list[Node]:
    return [x for x in NODES.values() if x.kind == kind]


def path(node_id: str) -> list[str]:
    """Ids from the root down to this node."""
    out = []
    while node_id is not None:
        out.append(node_id)
        node_id = NODES[node_id].parent
    return out[::-1]


if __name__ == "__main__":
    for kind in ("root", "disp", "amf", "smf", "upf"):
        print(f"{kind:5} {len(of_kind(kind)):3}")
    print(f"total {len(NODES)} nodes, {len(of_kind('upf'))} leaves")
    assert all(len([p for p in NODES.values() if x.id in p.children]) == 1
               for x in NODES.values() if x.id != ROOT), "a node has more than one parent"
    print("every non-root node has exactly one parent\n")

    def show(i, depth=0):
        x = NODES[i]
        if x.kind == "smf":          # print the leaves on the SMF line
            print("  " * depth + f"{i}  ->  " + ", ".join(c.rsplit('.', 1)[1] for c in x.children))
            return
        print("  " * depth + i)
        for c in x.children:
            show(c, depth + 1)
    show(ROOT)
