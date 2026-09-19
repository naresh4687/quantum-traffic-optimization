"""Road network: intersections as nodes, directed roads as edges.

Traffic model conventions
-------------------------
* Vehicles go straight through every intersection (no turning movements yet).
* An *approach* is the set of vehicles arriving at an intersection while
  travelling in one heading (E, W, N or S). It owns one queue. Its upstream is the
  road feeding it from a neighbouring intersection, or an external entry when the
  intersection is on the network boundary in that direction.
* Released vehicles continue in the same heading to the next intersection, or leave
  the network if there is none.
* Capacity is the maximum number of vehicles an approach's road/queue can hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import networkx as nx

from .signals import Phase


class TopologyError(ValueError):
    """Raised when a network definition is structurally invalid."""


class Heading(Enum):
    E = "E"
    W = "W"
    N = "N"
    S = "S"

    @property
    def phase(self) -> Phase:
        return Phase.EW if self in (Heading.E, Heading.W) else Phase.NS

    @property
    def step(self) -> tuple[int, int]:
        """(d_row, d_col) of one step in this heading; row 0 is the north edge."""
        return {"E": (0, 1), "W": (0, -1), "N": (-1, 0), "S": (1, 0)}[self.value]


@dataclass(frozen=True)
class Approach:
    node: str
    heading: Heading

    @property
    def phase(self) -> Phase:
        return self.heading.phase

    def __str__(self) -> str:
        return f"{self.node}:{self.heading.value}"


class Network:
    """A validated directed road network.

    ``graph`` edges must carry ``heading`` (Heading) and ``capacity`` (> 0).
    Nodes may carry ``pos=(row, col)``; if present it is checked against headings.
    ``entries`` maps boundary approaches (fed from outside) to their capacity.
    """

    def __init__(self, graph: nx.DiGraph, entries: Mapping[Approach, float]):
        self._graph = nx.freeze(graph.copy())
        self._entries: dict[Approach, float] = dict(entries)
        self._validate_graph()
        self._incoming: dict[Approach, tuple[str, str]] = {}
        for u, v, data in self._graph.edges(data=True):
            self._incoming[Approach(v, data["heading"])] = (u, v)
        self._validate_entries()
        self._approaches = tuple(
            Approach(node, h)
            for node in self._graph.nodes
            for h in Heading
            if Approach(node, h) in self._incoming or Approach(node, h) in self._entries
        )
        self._service_order = self._build_service_order()

    # -- validation ---------------------------------------------------------
    def _validate_graph(self) -> None:
        g = self._graph
        if g.number_of_nodes() == 0:
            raise TopologyError("network has no intersections")
        seen_out: set[tuple[str, Heading]] = set()
        seen_in: set[tuple[str, Heading]] = set()
        for u, v, data in g.edges(data=True):
            if u == v:
                raise TopologyError(f"self-loop road at {u}")
            heading = data.get("heading")
            if not isinstance(heading, Heading):
                raise TopologyError(f"road {u}->{v} has no valid heading")
            capacity = data.get("capacity")
            if not isinstance(capacity, (int, float)) or not capacity > 0:
                raise TopologyError(f"road {u}->{v} needs a positive capacity")
            if (u, heading) in seen_out:
                raise TopologyError(f"{u} has two outgoing roads with heading {heading.value}")
            if (v, heading) in seen_in:
                raise TopologyError(f"{v} has two incoming roads with heading {heading.value}")
            seen_out.add((u, heading))
            seen_in.add((v, heading))
            pu, pv = g.nodes[u].get("pos"), g.nodes[v].get("pos")
            if pu is not None and pv is not None:
                dr, dc = heading.step
                if (pu[0] + dr, pu[1] + dc) != tuple(pv):
                    raise TopologyError(
                        f"road {u}->{v} heading {heading.value} contradicts node positions"
                    )
        if not nx.is_strongly_connected(g):
            raise TopologyError("network is not strongly connected")

    def _validate_entries(self) -> None:
        for approach, capacity in self._entries.items():
            if approach.node not in self._graph:
                raise TopologyError(f"entry {approach} is at an unknown intersection")
            if approach in self._incoming:
                raise TopologyError(f"entry {approach} also has an incoming road")
            if not capacity > 0:
                raise TopologyError(f"entry {approach} needs a positive capacity")

    def _build_service_order(self) -> tuple[Approach, ...]:
        chain = nx.DiGraph()
        chain.add_nodes_from(self._approaches)
        for a in self._approaches:
            down = self.downstream_approach(a)
            if down is not None:
                chain.add_edge(a, down)
        if not nx.is_directed_acyclic_graph(chain):
            raise TopologyError("approach flow graph contains a cycle")
        # Downstream-first, so a queue's post-service length is known before its
        # upstream neighbour decides how much it may release into it.
        return tuple(reversed(list(nx.topological_sort(chain))))

    # -- basic accessors ----------------------------------------------------
    @property
    def graph(self) -> nx.DiGraph:
        """Read-only view of the intersection graph."""
        return self._graph

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(self._graph.nodes)

    @property
    def approaches(self) -> tuple[Approach, ...]:
        return self._approaches

    @property
    def entry_approaches(self) -> tuple[Approach, ...]:
        return tuple(a for a in self._approaches if a in self._entries)

    @property
    def service_order(self) -> tuple[Approach, ...]:
        """Approaches ordered downstream-first."""
        return self._service_order

    def roads(self) -> list[tuple[str, str, Heading, float]]:
        return [(u, v, d["heading"], d["capacity"]) for u, v, d in self._graph.edges(data=True)]

    def capacity(self, approach: Approach) -> float:
        if approach in self._entries:
            return self._entries[approach]
        u, v = self._incoming[approach]
        return self._graph[u][v]["capacity"]

    # -- upstream / downstream relationships --------------------------------
    def upstream_nodes(self, node: str) -> tuple[str, ...]:
        return tuple(self._graph.predecessors(node))

    def downstream_nodes(self, node: str) -> tuple[str, ...]:
        return tuple(self._graph.successors(node))

    def is_entry(self, approach: Approach) -> bool:
        return approach in self._entries

    def downstream_approach(self, approach: Approach) -> Approach | None:
        """Approach that released vehicles join next, or None if they exit."""
        for _, v, data in self._graph.out_edges(approach.node, data=True):
            if data["heading"] is approach.heading:
                return Approach(v, approach.heading)
        return None

    def upstream_approach(self, approach: Approach) -> Approach | None:
        """Approach feeding this one, or None if it is fed externally / not fed."""
        road = self._incoming.get(approach)
        if road is None:
            return None
        feeder = Approach(road[0], approach.heading)
        return feeder if feeder in self._approaches else None


def grid_network(
    rows: int = 2,
    cols: int = 3,
    road_capacity: float = 40.0,
    entry_capacity: float | None = None,
) -> Network:
    """Build a rows x cols grid; intersections are named I1..I<n> row by row.

    The default 2x3 grid is::

        I1 --- I2 --- I3
        |      |      |
        I4 --- I5 --- I6

    Every adjacent pair is joined by a road in each direction. ``entry_capacity``
    defaults to ``road_capacity``.
    """
    if rows < 1 or cols < 1 or rows * cols < 2:
        raise TopologyError("grid needs at least two intersections")
    entry_capacity = road_capacity if entry_capacity is None else entry_capacity

    def name(r: int, c: int) -> str:
        return f"I{r * cols + c + 1}"

    g = nx.DiGraph()
    for r in range(rows):
        for c in range(cols):
            g.add_node(name(r, c), pos=(r, c))
    for r in range(rows):
        for c in range(cols):
            for h in Heading:
                dr, dc = h.step
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    g.add_edge(name(r, c), name(nr, nc), heading=h, capacity=road_capacity)
    entries = {
        Approach(name(r, c), h): entry_capacity
        for r in range(rows)
        for c in range(cols)
        for h in Heading
        if not (0 <= r - h.step[0] < rows and 0 <= c - h.step[1] < cols)
    }
    return Network(g, entries)
