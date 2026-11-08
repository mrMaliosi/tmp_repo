"""Candidate solver trajectories (broken, slow backtracking, and blazing-fast bitmask DP) for TSPTW."""

from __future__ import annotations

# 1. Naive buggy greedy solver (fails most time-window constraints)
CANDIDATE_GREEDY_BUGGY = '''
def solve(dist_matrix: list[list[int]], time_windows: list[tuple[int, int]], service_times: list[int]) -> int:
    """Greedy nearest-neighbor heuristic that forgets to backtrack when a window is missed."""
    n = len(dist_matrix)
    if n <= 1:
        return 0
    unvisited = set(range(1, n))
    cur = 0
    cur_time = max(0, time_windows[0][0]) + service_times[0]
    total_dist = 0
    while unvisited:
        nxt = min(unvisited, key=lambda v: dist_matrix[cur][v])
        unvisited.remove(nxt)
        arr = cur_time + dist_matrix[cur][nxt]
        if arr > time_windows[nxt][1]:
            return -1
        cur_time = max(arr, time_windows[nxt][0]) + service_times[nxt]
        total_dist += dist_matrix[cur][nxt]
        cur = nxt
    ret_arr = cur_time + dist_matrix[cur][0]
    if not (time_windows[0][0] <= ret_arr <= time_windows[0][1]):
        return -1
    return total_dist + dist_matrix[cur][0]
'''

# 2. Unoptimized DFS backtracking solver (correct on all tests, but explores branches with minimal pruning)
CANDIDATE_DFS_SLOW = '''
def solve(dist_matrix: list[list[int]], time_windows: list[tuple[int, int]], service_times: list[int]) -> int:
    """Correct recursive DFS solver with basic deadline checks."""
    n = len(dist_matrix)
    if n <= 1:
        return 0
    full_mask = (1 << n) - 1
    best = [10**15]

    def dfs(u: int, mask: int, cur_time: int, cost: int) -> None:
        if cost >= best[0]:
            return
        if mask == full_mask:
            ret_arr = cur_time + dist_matrix[u][0]
            if time_windows[0][0] <= ret_arr <= time_windows[0][1]:
                total = cost + dist_matrix[u][0]
                if total < best[0]:
                    best[0] = total
            return

        # Artificial work to simulate unoptimized state copying overhead
        candidates = [v for v in range(1, n) if not (mask & (1 << v))]
        candidates.sort(key=lambda v: time_windows[v][1])
        for v in candidates:
            arr = cur_time + dist_matrix[u][v]
            if arr <= time_windows[v][1]:
                start = time_windows[v][0] if arr < time_windows[v][0] else arr
                dfs(v, mask | (1 << v), start + service_times[v], cost + dist_matrix[u][v])

    t0 = max(0, time_windows[0][0]) + service_times[0]
    dfs(0, 1, t0, 0)
    return -1 if best[0] >= 10**15 else best[0]
'''

# 3. Blazing-fast Bitmask DP + Pareto Frontier Pruning solver
CANDIDATE_BITMASK_DP_FAST = '''
def solve(dist_matrix: list[list[int]], time_windows: list[tuple[int, int]], service_times: list[int]) -> int:
    """High-performance Bitmask Dynamic Programming with Pareto state dominance pruning."""
    n = len(dist_matrix)
    if n <= 1:
        return 0
    full_mask = (1 << n) - 1
    t0 = time_windows[0][0] if time_windows[0][0] > 0 else 0
    dp: dict[tuple[int, int], list[tuple[int, int]]] = {(1, 0): [(0, t0 + service_times[0])]}

    for mask in range(1, full_mask + 1):
        if not (mask & 1):
            continue
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            states = dp.get((mask, u))
            if not states:
                continue
            row_u = dist_matrix[u]
            for v in range(1, n):
                if mask & (1 << v):
                    continue
                travel = row_u[v]
                ew, lw = time_windows[v]
                svc = service_times[v]
                nxt_key = (mask | (1 << v), v)
                bucket = dp.setdefault(nxt_key, [])
                for cost, cur_time in states:
                    arr = cur_time + travel
                    if arr > lw:
                        continue
                    fin = (ew if arr < ew else arr) + svc
                    nc = cost + travel
                    dominated = False
                    kept = []
                    for ec, et in bucket:
                        if ec <= nc and et <= fin:
                            dominated = True
                            break
                        if not (nc <= ec and fin <= et):
                            kept.append((ec, et))
                    if not dominated:
                        kept.append((nc, fin))
                        dp[nxt_key] = kept

    best = -1
    d_early, d_late = time_windows[0]
    for u in range(1, n):
        states = dp.get((full_mask, u))
        if not states:
            continue
        ret = dist_matrix[u][0]
        for cost, cur_time in states:
            if d_early <= cur_time + ret <= d_late:
                tot = cost + ret
                if best == -1 or tot < best:
                    best = tot
    return best
'''
