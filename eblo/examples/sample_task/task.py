"""Non-trivial demonstration task: Traveling Salesperson Problem with Time Windows (TSPTW)."""

from __future__ import annotations

import random

from eblo.core.domain import TaskSpec, TestCase


def _reference_solve_tsptw(
    dist_matrix: list[list[int]],
    time_windows: list[tuple[int, int]],
    service_times: list[int],
) -> int:
    """Reference exact solver using bitmask DP over Pareto-minimal arrival times for generating ground truth."""
    n = len(dist_matrix)
    if n <= 1:
        return 0

    full_mask = (1 << n) - 1
    # dp[(mask, u)] = (min_cost, min_time) or list of Pareto-undominated (cost, time) pairs
    dp: dict[tuple[int, int], list[tuple[int, int]]] = {(1, 0): [(0, max(time_windows[0][0], 0) + service_times[0])]}

    for mask in range(1, full_mask + 1):
        if not (mask & 1):
            continue
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            states = dp.get((mask, u))
            if not states:
                continue
            for v in range(1, n):
                if mask & (1 << v):
                    continue
                travel = dist_matrix[u][v]
                earliest, latest = time_windows[v]
                svc = service_times[v]
                next_mask = mask | (1 << v)
                key = (next_mask, v)
                existing = dp.setdefault(key, [])

                for cost, cur_time in states:
                    arrival = cur_time + travel
                    if arrival > latest:
                        continue
                    start_service = earliest if arrival < earliest else arrival
                    finish_time = start_service + svc
                    new_cost = cost + travel

                    # Check dominance in existing
                    dominated = False
                    filtered: list[tuple[int, int]] = []
                    for ec, et in existing:
                        if ec <= new_cost and et <= finish_time:
                            dominated = True
                            break
                        if not (new_cost <= ec and finish_time <= et):
                            filtered.append((ec, et))
                    if not dominated:
                        filtered.append((new_cost, finish_time))
                        dp[key] = filtered

    best_total = -1
    depot_earliest, depot_latest = time_windows[0]
    for u in range(1, n):
        states = dp.get((full_mask, u))
        if not states:
            continue
        ret_cost = dist_matrix[u][0]
        for cost, cur_time in states:
            return_arrival = cur_time + ret_cost
            if depot_earliest <= return_arrival <= depot_latest:
                total = cost + ret_cost
                if best_total == -1 or total < best_total:
                    best_total = total
    return best_total


def _generate_random_tsptw_instance(
    n: int,
    seed: int,
    feasible: bool = True,
) -> tuple[list[list[int]], list[tuple[int, int]], list[int], int]:
    rng = random.Random(seed)
    coords = [(rng.randint(0, 60), rng.randint(0, 60)) for _ in range(n)]
    dist_matrix: list[list[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dist_matrix[i][j] = int((dx * dx + dy * dy) ** 0.5) + 1

    service_times = [0] + [rng.randint(2, 8) for _ in range(n - 1)]

    if not feasible:
        # Force an impossible deadline on node 1
        time_windows = [(0, 1000)] + [(0, 0)] + [(0, 500) for _ in range(n - 2)]
        expected = _reference_solve_tsptw(dist_matrix, time_windows, service_times)
        return dist_matrix, time_windows, service_times, expected

    # Construct a guaranteed feasible hidden tour and build time windows around arrival times
    perm = list(range(1, n))
    rng.shuffle(perm)
    cur_node = 0
    cur_time = 0
    windows: dict[int, tuple[int, int]] = {}
    for nxt in perm:
        cur_time += dist_matrix[cur_node][nxt]
        slack_before = rng.randint(10, 45)
        slack_after = rng.randint(25, 90)
        windows[nxt] = (max(0, cur_time - slack_before), cur_time + slack_after)
        cur_time += service_times[nxt]
        cur_node = nxt

    cur_time += dist_matrix[cur_node][0]
    windows[0] = (0, cur_time + 200)
    time_windows = [windows[i] for i in range(n)]
    expected = _reference_solve_tsptw(dist_matrix, time_windows, service_times)
    return dist_matrix, time_windows, service_times, expected


def get_task_spec() -> TaskSpec:
    """Build the TSPTW TaskSpec with smoke tests and full benchmark tests."""
    description = """
Solve the **Traveling Salesperson Problem with Time Windows (TSPTW)**.

Write a top-level Python function with the exact signature:
```python
def solve(
    dist_matrix: list[list[int]],
    time_windows: list[tuple[int, int]],
    service_times: list[int],
) -> int:
```

### Problem Rules:
1. You are given `N` locations indexed `0` to `N - 1`, where `0` is the starting and ending depot.
2. `dist_matrix[u][v]` is the non-negative integer travel time (and travel cost) from location `u` to location `v`.
3. Every location `i` has a time window `(earliest_i, latest_i) = time_windows[i]` and service duration `service_times[i]`.
4. The tour starts at node `0` at time `max(0, time_windows[0][0]) + service_times[0]`.
5. When moving from `u` to `v` (departing `u` at `t_depart`):
   - You arrive at `v` at `t_arrive = t_depart + dist_matrix[u][v]`.
   - If `t_arrive > time_windows[v][1]`, this transition violates the deadline and is **infeasible**.
   - If `t_arrive < time_windows[v][0]`, you wait until `t_start = time_windows[v][0]`; otherwise `t_start = t_arrive`.
   - You depart `v` at `t_depart_v = t_start + service_times[v]`.
6. After visiting all nodes `1 .. N - 1` exactly once, you must return from the last node `u` to depot `0` such that the return arrival time `t_depart_u + dist_matrix[u][0]` is within `[time_windows[0][0], time_windows[0][1]]`.
7. **Objective**: Minimize the **total travel distance** `sum(dist_matrix[u][v])` along the tour (waiting times and service times affect schedule feasibility, NOT travel distance cost).
8. If no feasible tour visits all nodes within their time windows, return `-1`.
""".strip()

    smoke_tests: list[TestCase] = [
        TestCase(
            name="smoke_small_3_nodes",
            args=(
                [[0, 10, 15], [10, 0, 20], [15, 20, 0]],
                [(0, 100), (5, 25), (25, 60)],
                [0, 5, 5],
            ),
            expected=45,
        ),
        TestCase(
            name="smoke_infeasible_window",
            args=(
                [[0, 20, 20], [20, 0, 15], [20, 15, 0]],
                [(0, 100), (0, 10), (0, 100)],
                [0, 2, 2],
            ),
            expected=-1,
        ),
        TestCase(
            name="smoke_waiting_required",
            args=(
                [[0, 5, 50], [5, 0, 10], [50, 10, 0]],
                [(0, 200), (20, 30), (35, 50)],
                [0, 5, 5],
            ),
            expected=65,
        ),
    ]

    benchmark_tests: list[TestCase] = list(smoke_tests)
    # Add 9 non-trivial benchmark instances up to N=11 nodes
    configs = [
        (5, 101, True),
        (6, 202, True),
        (7, 303, True),
        (8, 404, True),
        (8, 505, False),
        (9, 606, True),
        (9, 707, True),
        (10, 808, True),
        (10, 909, True),
    ]
    for idx, (n_nodes, seed, is_feas) in enumerate(configs, start=1):
        dm, tw, st, ans = _generate_random_tsptw_instance(n_nodes, seed=seed, feasible=is_feas)
        benchmark_tests.append(
            TestCase(
                name=f"bench_{idx}_n{n_nodes}_seed{seed}",
                args=(dm, tw, st),
                expected=ans,
            )
        )

    return TaskSpec(
        task_id="tsptw_exact_optimization",
        description=description,
        smoke_tests=smoke_tests,
        benchmark_tests=benchmark_tests,
        entrypoint="solve",
        language="python",
    )


TASK_SPEC = get_task_spec()
