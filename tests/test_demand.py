import pytest

from qtraffic import DEMAND_LEVELS, Approach, DemandConfig, DemandModel, Heading, grid_network


@pytest.fixture
def net():
    return grid_network()


def test_same_seed_and_config_give_identical_demand(net):
    a = DemandModel(net, DemandConfig(seed=7))
    b = DemandModel(net, DemandConfig(seed=7))
    assert [a.arrivals(c) for c in range(30)] == [b.arrivals(c) for c in range(30)]


def test_different_seed_gives_different_demand(net):
    a = DemandModel(net, DemandConfig(seed=1)).arrivals(0)
    b = DemandModel(net, DemandConfig(seed=2)).arrivals(0)
    assert a != b


def test_arrivals_are_independent_of_query_order(net):
    m = DemandModel(net, DemandConfig(seed=3))
    forward = {c: m.arrivals(c) for c in range(10)}
    fresh = DemandModel(net, DemandConfig(seed=3))
    assert fresh.arrivals(9) == forward[9]
    assert fresh.arrivals(4) == forward[4]


def test_demand_only_at_entry_approaches_and_never_negative(net):
    arrivals = DemandModel(net, DemandConfig(seed=0, variation=1.0)).arrivals(0)
    assert set(arrivals) == set(net.entry_approaches)
    assert all(v >= 0 for v in arrivals.values())


def test_arrivals_stay_within_variation_band_and_mean_matches_base(net):
    cfg = DemandConfig(base_rate=10.0, seed=5, variation=0.3)
    m = DemandModel(net, cfg)
    samples = [v for c in range(1000) for v in m.arrivals(c).values()]
    assert min(samples) >= 10.0 * 0.7 - 1e-9
    assert max(samples) <= 10.0 * 1.3 + 1e-9
    assert sum(samples) / len(samples) == pytest.approx(10.0, rel=0.01)


def test_zero_variation_is_constant(net):
    m = DemandModel(net, DemandConfig(base_rate=9.0, variation=0.0))
    assert set(m.arrivals(0).values()) == {9.0} == set(m.arrivals(17).values())


def test_multipliers_scale_individual_entries(net):
    hot = Approach("I1", Heading.E)
    cfg = DemandConfig(base_rate=10.0, variation=0.0, multipliers={hot: 2.0})
    arrivals = DemandModel(net, cfg).arrivals(0)
    assert arrivals[hot] == 20.0
    assert arrivals[Approach("I4", Heading.E)] == 10.0


def test_demand_levels_are_ordered(net):
    rates = [DemandConfig.from_level(lvl).base_rate for lvl in ("low", "medium", "high")]
    assert rates == sorted(rates) and len(set(rates)) == 3
    assert rates == [DEMAND_LEVELS[k] for k in ("low", "medium", "high")]


@pytest.mark.parametrize(
    "kwargs",
    [{"base_rate": -1.0}, {"seed": -1}, {"seed": 1.5}, {"variation": 1.5}, {"variation": -0.1}],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DemandConfig(**kwargs)


def test_unknown_level_and_non_entry_multiplier_rejected(net):
    with pytest.raises(ValueError):
        DemandConfig.from_level("extreme")
    with pytest.raises(ValueError, match="non-entry"):
        DemandModel(net, DemandConfig(multipliers={Approach("I2", Heading.E): 2.0}))
