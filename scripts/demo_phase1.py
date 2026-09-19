"""Phase 1 demo: 6-intersection grid, fixed 30/30 control, 20 cycles.

Run:  python scripts/demo_phase1.py [--level low|medium|high] [--seed N] [--cycles N]

Everything printed is computed by the simulator; the checks at the end fail loudly
(non-zero exit) if determinism, conservation, propagation or non-negativity break.
"""

from __future__ import annotations

import argparse
import sys

from qtraffic import (
    Approach, DemandConfig, DemandModel, FixedTimeController, Heading, Simulator, grid_network,
)

EPS = 1e-6


def build(level: str, seed: int) -> Simulator:
    net = grid_network(2, 3)
    return Simulator(net, DemandModel(net, DemandConfig.from_level(level, seed=seed)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cycles", type=int, default=20)
    args = ap.parse_args()

    sim = build(args.level, args.seed)
    net = sim.network
    result = sim.run(FixedTimeController(), args.cycles)
    m, hist = result.metrics, result.history

    print("=" * 66)
    print(f"Network: 2x3 grid, {len(net.nodes)} intersections, {net.graph.number_of_edges()} directed roads")
    print("    I1 --- I2 --- I3")
    print("    |      |      |")
    print("    I4 --- I5 --- I6")
    print(f"Controller: {result.controller} (NS=30 / EW=30, 60 s cycle)")
    print(f"Demand: level={args.level} ({sim.demand.config.base_rate:g} veh/cycle/entry), "
          f"seed={args.seed}, cycles={m.cycles}")
    print("=" * 66)
    print(f"Vehicles demanded (external)  : {m.demand_generated:10.2f}")
    print(f"Vehicles entering network     : {m.vehicles_entered:10.2f}  (rejected: {m.vehicles_rejected:.2f})")
    print(f"Vehicles served (stop-line)   : {m.vehicles_served:10.2f}")
    print(f"Vehicles exited network       : {m.vehicles_exited:10.2f}")
    print(f"Throughput                    : {m.throughput_per_hour:10.1f} veh/h")
    print(f"Total waiting time            : {m.total_waiting_time:10.1f} veh*s")
    print(f"Average queue (per approach)  : {m.average_queue_length:10.3f} veh")
    print(f"Maximum queue (any approach)  : {m.max_queue_length:10.2f} veh")
    print(f"Blocked by downstream capacity: {m.total_blocked:10.2f} veh")
    print("Final queue per intersection  : " + ", ".join(f"{n}={q:.2f}" for n, q in m.final_queue.items()))
    print(f"Final queue total / in transit: {m.final_queue_total:.2f} / {m.in_transit_final:.2f}")

    # -- propagation proof --------------------------------------------------
    print("\nPropagation along the top row, eastbound (I1 -> I2 -> I3 -> exit):")
    print("cycle | I1:E served -> I2:E arrivals(next) | I2:E served -> I3:E arrivals(next)")
    chain = [Approach(n, Heading.E) for n in ("I1", "I2", "I3")]
    for rec, nxt in zip(hist[:6], hist[1:7]):
        print(f"{rec.cycle:5d} | {rec.served[chain[0]]:10.2f} -> {nxt.transit_arrivals[chain[1]]:10.2f}"
              f"             | {rec.served[chain[1]]:10.2f} -> {nxt.transit_arrivals[chain[2]]:10.2f}")

    link_released = link_received = 0.0
    for rec, nxt in zip(hist, hist[1:]):
        for a in net.approaches:
            down = net.downstream_approach(a)
            if down is not None:
                link_released += rec.served[a]
                link_received += nxt.transit_arrivals[down]
    print(f"Inter-intersection flow, cycles 0..{len(hist) - 2}: released {link_released:.2f}, "
          f"received next cycle {link_received:.2f}")
    print("Vehicles received from upstream per intersection (total):")
    for node in net.nodes:
        got = sum(rec.transit_arrivals[a] for rec in hist for a in net.approaches if a.node == node)
        print(f"  {node}: {got:8.2f}")

    # -- validation ---------------------------------------------------------
    print("\nChecks:")
    again = build(args.level, args.seed).run(FixedTimeController(), args.cycles)
    deterministic = again.history == result.history and again.metrics == m
    print(f"  deterministic (two identical runs give identical results): {deterministic}")

    balance = m.vehicles_entered - (m.vehicles_exited + m.final_queue_total + m.in_transit_final)
    demand_balance = m.demand_generated - (m.vehicles_entered + m.vehicles_rejected)
    conserved = abs(balance) < EPS and abs(demand_balance) < EPS
    print(f"  conservation: entered - (exited + queued + in transit) = {balance:.2e}; "
          f"demand - (entered + rejected) = {demand_balance:.2e} -> {conserved}")

    min_queue = min(min(r.queue_end.values()) for r in hist)
    non_negative = min_queue >= 0 and all(v >= 0 for r in hist for v in r.queue_after_arrivals.values())
    print(f"  no negative queues (min queue observed = {min_queue:.4f}): {non_negative}")

    propagates = (
        abs(link_released - link_received) < EPS
        and link_received > 0
        and all(any(rec.transit_arrivals[a] > 0 for rec in hist for a in net.approaches if a.node == n)
                for n in net.nodes if any(net.upstream_approach(a) for a in net.approaches if a.node == n))
    )
    print(f"  downstream intersections receive upstream flow: {propagates}")

    ok = deterministic and conserved and non_negative and propagates
    print("\nALL CHECKS PASSED" if ok else "\nCHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
