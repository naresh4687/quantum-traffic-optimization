import networkx as nx
import pytest

from qtraffic import Approach, Heading, Network, TopologyError, grid_network

E, W, N, S = Heading.E, Heading.W, Heading.N, Heading.S


def test_default_network_has_six_intersections():
    net = grid_network()
    assert net.nodes == ("I1", "I2", "I3", "I4", "I5", "I6")


def test_2x3_grid_topology_matches_diagram():
    net = grid_network()
    horizontal = [("I1", "I2"), ("I2", "I3"), ("I4", "I5"), ("I5", "I6")]
    vertical = [("I1", "I4"), ("I2", "I5"), ("I3", "I6")]
    expected = set()
    for u, v in horizontal + vertical:
        expected |= {(u, v), (v, u)}
    assert set(net.graph.edges) == expected
    assert net.graph.number_of_edges() == 14
    # no diagonal / wrap-around roads
    assert not net.graph.has_edge("I1", "I5")
    assert not net.graph.has_edge("I3", "I4")
    assert not net.graph.has_edge("I1", "I3")


def test_road_headings_are_consistent_with_geometry():
    net = grid_network()
    g = net.graph
    assert g["I1"]["I2"]["heading"] is E
    assert g["I2"]["I1"]["heading"] is W
    assert g["I1"]["I4"]["heading"] is S
    assert g["I4"]["I1"]["heading"] is N
    assert g["I5"]["I6"]["heading"] is E


def test_directed_upstream_downstream_relationships():
    net = grid_network()
    assert set(net.downstream_nodes("I2")) == {"I1", "I3", "I5"}
    assert set(net.upstream_nodes("I2")) == {"I1", "I3", "I5"}
    assert set(net.downstream_nodes("I1")) == {"I2", "I4"}
    # eastbound traffic flows I1 -> I2 -> I3 -> out
    assert net.downstream_approach(Approach("I1", E)) == Approach("I2", E)
    assert net.downstream_approach(Approach("I2", E)) == Approach("I3", E)
    assert net.downstream_approach(Approach("I3", E)) is None
    # southbound traffic flows I2 -> I5 -> out
    assert net.downstream_approach(Approach("I2", S)) == Approach("I5", S)
    assert net.downstream_approach(Approach("I5", S)) is None
    # and the inverse relationship
    assert net.upstream_approach(Approach("I2", E)) == Approach("I1", E)
    assert net.upstream_approach(Approach("I1", E)) is None


def test_entry_approaches_are_exactly_the_boundary_inflows():
    net = grid_network()
    entries = set(net.entry_approaches)
    assert len(entries) == 10
    assert entries == {
        Approach("I1", E), Approach("I4", E), Approach("I3", W), Approach("I6", W),
        Approach("I1", S), Approach("I2", S), Approach("I3", S),
        Approach("I4", N), Approach("I5", N), Approach("I6", N),
    }
    # every approach is either an entry or has an upstream road; never both
    for a in net.approaches:
        assert net.is_entry(a) != (net.upstream_approach(a) is not None)


def test_capacity_is_configurable():
    net = grid_network(road_capacity=25.0, entry_capacity=60.0)
    assert net.capacity(Approach("I2", E)) == 25.0
    assert net.capacity(Approach("I1", E)) == 60.0
    assert all(cap == 25.0 for *_, cap in net.roads())


def test_service_order_puts_downstream_before_upstream():
    net = grid_network()
    order = {a: i for i, a in enumerate(net.service_order)}
    assert set(order) == set(net.approaches)
    for a in net.approaches:
        down = net.downstream_approach(a)
        if down is not None:
            assert order[down] < order[a]


def test_topology_is_extendable_to_eight_intersections():
    net = grid_network(2, 4)
    assert len(net.nodes) == 8
    assert net.graph.number_of_edges() == 20
    assert net.downstream_approach(Approach("I3", E)) == Approach("I4", E)
    assert len(net.entry_approaches) == 12


def _line(headings_and_caps, **kwargs):
    g = nx.DiGraph()
    for u, v, h, cap in headings_and_caps:
        g.add_edge(u, v, heading=h, capacity=cap)
    return g


def test_validation_rejects_self_loop():
    g = _line([("A", "A", E, 5), ("A", "B", E, 5), ("B", "A", W, 5)])
    with pytest.raises(TopologyError, match="self-loop"):
        Network(g, {})


def test_validation_rejects_non_positive_capacity():
    g = _line([("A", "B", E, 0), ("B", "A", W, 5)])
    with pytest.raises(TopologyError, match="capacity"):
        Network(g, {})


def test_validation_rejects_missing_heading():
    g = nx.DiGraph()
    g.add_edge("A", "B", capacity=5)
    g.add_edge("B", "A", capacity=5)
    with pytest.raises(TopologyError, match="heading"):
        Network(g, {})


def test_validation_rejects_ambiguous_outgoing_heading():
    g = _line([("A", "B", E, 5), ("A", "C", E, 5), ("B", "A", W, 5), ("C", "A", W, 5)])
    with pytest.raises(TopologyError, match="two outgoing"):
        Network(g, {})


def test_validation_rejects_heading_contradicting_positions():
    g = nx.DiGraph()
    g.add_node("A", pos=(0, 0))
    g.add_node("B", pos=(0, 1))
    g.add_edge("A", "B", heading=S, capacity=5)  # B is east of A, not south
    g.add_edge("B", "A", heading=W, capacity=5)
    with pytest.raises(TopologyError, match="positions"):
        Network(g, {})


def test_validation_rejects_disconnected_network():
    g = _line([("A", "B", E, 5), ("B", "A", W, 5), ("C", "D", E, 5), ("D", "C", W, 5)])
    with pytest.raises(TopologyError, match="strongly connected"):
        Network(g, {})


def test_validation_rejects_entry_on_fed_approach():
    g = _line([("A", "B", E, 5), ("B", "A", W, 5)])
    with pytest.raises(TopologyError, match="incoming road"):
        Network(g, {Approach("B", E): 5})


def test_validation_rejects_flow_cycle():
    # a one-way ring: E from A to B, then E from B back to A
    g = _line([("A", "B", E, 5), ("B", "A", E, 5)])
    with pytest.raises(TopologyError, match="cycle"):
        Network(g, {})


def test_graph_view_is_read_only():
    net = grid_network()
    with pytest.raises(nx.NetworkXError):
        net.graph.add_edge("I1", "I6", heading=E, capacity=1)
