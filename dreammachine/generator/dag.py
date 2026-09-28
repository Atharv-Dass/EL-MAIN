"""Computation DAG for word problems, evaluated in exact rational arithmetic.

Every generated problem carries the DAG that produced its answer. This is what
makes the diagnosis "construct-valid": the skills an item requires (which
operations, how many steps, which numbers) are known by construction rather
than tagged after the fact by an LLM or a human.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction


class Op(str, Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"

    def apply(self, a: Fraction, b: Fraction) -> Fraction:
        if self is Op.ADD:
            return a + b
        if self is Op.SUB:
            return a - b
        if self is Op.MUL:
            return a * b
        if b == 0:
            raise ZeroDivisionError("division by zero in DAG")
        return a / b


@dataclass(frozen=True)
class Node:
    id: int
    kind: str  # "leaf" or "op"
    value: Fraction
    op: Op | None = None
    inputs: tuple[int, ...] = ()
    role: str = ""  # semantic tag, e.g. "start", "operand", "combine"

    @property
    def is_leaf(self) -> bool:
        return self.kind == "leaf"


class DAG:
    """Append-only DAG. Insertion order is a valid topological order."""

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.root: int | None = None

    def leaf(self, value: int | Fraction, role: str = "operand") -> int:
        node = Node(id=len(self.nodes), kind="leaf", value=Fraction(value), role=role)
        self.nodes.append(node)
        return node.id

    def op(self, op: Op, a: int, b: int, role: str = "step") -> int:
        for i in (a, b):
            if not 0 <= i < len(self.nodes):
                raise IndexError(f"input node {i} does not exist")
        value = op.apply(self.nodes[a].value, self.nodes[b].value)
        node = Node(id=len(self.nodes), kind="op", value=value, op=op, inputs=(a, b), role=role)
        self.nodes.append(node)
        self.root = node.id
        return node.id

    # ------------------------------------------------------------------ views
    def __getitem__(self, i: int) -> Node:
        return self.nodes[i]

    @property
    def answer(self) -> Fraction:
        if self.root is None:
            raise ValueError("DAG has no operation nodes")
        return self.nodes[self.root].value

    def leaves(self) -> list[Node]:
        return [n for n in self.nodes if n.is_leaf]

    def op_nodes(self) -> list[Node]:
        return [n for n in self.nodes if not n.is_leaf]

    @property
    def n_steps(self) -> int:
        return len(self.op_nodes())

    def depth(self) -> int:
        """Longest chain of operations from any leaf to the root."""
        d: dict[int, int] = {}
        for n in self.nodes:
            d[n.id] = 0 if n.is_leaf else 1 + max(d[i] for i in n.inputs)
        return d[self.root] if self.root is not None else 0

    def evaluate(self) -> Fraction:
        """Recompute the root from the leaves, independently of stored op values."""
        vals: dict[int, Fraction] = {}
        for n in self.nodes:
            if n.is_leaf:
                vals[n.id] = n.value
            else:
                a, b = n.inputs
                vals[n.id] = n.op.apply(vals[a], vals[b])  # type: ignore[union-attr]
        if self.root is None:
            raise ValueError("DAG has no operation nodes")
        return vals[self.root]

    # -------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "nodes": [
                {
                    "id": n.id,
                    "kind": n.kind,
                    "value": str(n.value),
                    "op": n.op.value if n.op else None,
                    "inputs": list(n.inputs),
                    "role": n.role,
                }
                for n in self.nodes
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DAG":
        dag = cls()
        for nd in d["nodes"]:
            dag.nodes.append(
                Node(
                    id=nd["id"],
                    kind=nd["kind"],
                    value=Fraction(nd["value"]),
                    op=Op(nd["op"]) if nd["op"] else None,
                    inputs=tuple(nd["inputs"]),
                    role=nd.get("role", ""),
                )
            )
        dag.root = d["root"]
        return dag
