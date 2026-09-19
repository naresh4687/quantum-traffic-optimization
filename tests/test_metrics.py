import pytest
from helpers import ScriptedDemand

from qtraffic import (
    CYCLE_LENGTH, Approach, FixedTimeController, Heading, Simulator, compute_metrics, grid_network,
)


@pytest.fixture(scope="module")
def result():
    """Hand-checkable scenario: 20 veh/cycle enters I1 eastbound for 3 cycles.

    Green capacity is 15 veh/cycle (30 s * 0.5 veh/s) so the I1 queue builds
    5 -> 10 -> 15, and 15 veh/cycle flow on to I2 (from cycle 1) and I3 (cycle 2).
    """
    net = grid_network(1, 3)  # 3 intersections x 4 approaches = 12 approaches
    demand = ScriptedDemand(constant={Approach("I1", Heading.E): 20.0})
    return Simulator(net, demand).run(FixedTimeController(), 3)


def test_total_waiting_time_matches_hand_calculation(result):
    # I1:E  (q0,q1) = (20,5) (25,10) (30,15)  -> (12.5+17.5+22.5)*60 = 3150
    # I2:E  (15,0) in cycles 1 and 2          -> 7.5*60*2           =  900
    # I3:E  (15,0) in cycle 2                 -> 7.5*60             =  450
    assert result.metrics.total_waiting_time == pytest.approx(4500.0)


def test_average_queue_is_time_averaged_per_approach(result):
    m = result.metrics
    assert m.average_queue_length == pytest.approx(4500.0 / (CYCLE_LENGTH * 3 * 12))


def test_max_queue_is_the_peak_after_arrivals(result):
    assert result.metrics.max_queue_length == pytest.approx(30.0)


def test_throughput_and_vehicles_served(result):
    m = result.metrics
    assert m.vehicles_entered == pytest.approx(60.0)
    assert m.vehicles_served == pytest.approx(45.0 + 30.0 + 15.0)  # stop-line discharges
    assert m.vehicles_exited == pytest.approx(15.0)  # only the first batch has left
    assert m.throughput_per_hour == pytest.approx(15.0 / (3 * 60 / 3600))


def test_final_queue_per_intersection_and_in_transit(result):
    m = result.metrics
    assert m.final_queue == pytest.approx({"I1": 15.0, "I2": 0.0, "I3": 0.0})
    assert m.final_queue_total == pytest.approx(15.0)
    assert m.in_transit_final == pytest.approx(30.0)
    # 60 entered = 15 exited + 15 queued + 30 in transit
    assert m.vehicles_entered == pytest.approx(m.vehicles_exited + m.final_queue_total + m.in_transit_final)


def test_metrics_are_recomputable_from_history_and_exportable(result):
    assert compute_metrics(result.history) == result.metrics
    d = result.metrics.as_dict()
    assert d["cycles"] == 3 and d["total_waiting_time"] == pytest.approx(4500.0)


def test_empty_network_has_all_zero_metrics():
    net = grid_network()
    m = Simulator(net, ScriptedDemand()).run(FixedTimeController(), 5).metrics
    assert m.total_waiting_time == 0 and m.average_queue_length == 0 and m.max_queue_length == 0
    assert m.vehicles_entered == m.vehicles_served == m.throughput_per_hour == 0
    assert set(m.final_queue) == set(net.nodes) and m.final_queue_total == 0


def test_cannot_compute_metrics_without_history():
    with pytest.raises(ValueError):
        compute_metrics([])
