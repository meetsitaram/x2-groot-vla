"""Grid path planner for the kitchen (nav_grid.npz: walkable mask is already
eroded by the robot footprint; esdf = clearance in m).

A* (8-connected) with a clearance penalty, then line-of-sight shortcutting
on esdf >= LOS_CLEAR. Start/goal may sit outside the eroded mask (counter
stands do); they are connected to the nearest walkable cell.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

GRID = "/home/stickbot/projects/x2-kitchen-sim/assets/nav_grid.npz"
LOS_CLEAR = 0.26


class NavPlanner:
    def __init__(self, path=GRID, stride=2):
        g = np.load(path)
        self.W = g["walkable"][::stride, ::stride]
        self.E = g["esdf"][::stride, ::stride]
        self.o = np.asarray(g["origin"], float)
        self.r = float(g["res"]) * stride

    def cell(self, x, y):
        return (int(round((x - self.o[0]) / self.r)), int(round((y - self.o[1]) / self.r)))

    def xy(self, c):
        return (self.o[0] + c[0] * self.r, self.o[1] + c[1] * self.r)

    def esdf(self, x, y):
        i, j = self.cell(x, y)
        if 0 <= i < self.E.shape[0] and 0 <= j < self.E.shape[1]:
            return float(self.E[i, j])
        return -1.0

    def _nearest_walkable(self, c):
        ii, jj = np.nonzero(self.W)
        k = np.argmin((ii - c[0]) ** 2 + (jj - c[1]) ** 2)
        return (int(ii[k]), int(jj[k]))

    def _los(self, a, b, clear=LOS_CLEAR):
        (x0, y0), (x1, y1) = a, b
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / (self.r * 0.5)))
        for t in np.linspace(0, 1, n):
            if self.esdf(x0 + t * (x1 - x0), y0 + t * (y1 - y0)) < clear:
                return False
        return True

    def plan(self, start_xy, goal_xy):
        s = self.cell(*start_xy)
        g = self.cell(*goal_xy)
        s_w = s if self.W[s] else self._nearest_walkable(s)
        g_w = g if self.W[g] else self._nearest_walkable(g)
        H, Wd = self.W.shape
        dist = {s_w: 0.0}
        prev = {}
        pq = [(0.0, s_w)]
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        while pq:
            f, c = heapq.heappop(pq)
            if c == g_w:
                break
            for di, dj in nbrs:
                n = (c[0] + di, c[1] + dj)
                if not (0 <= n[0] < H and 0 <= n[1] < Wd) or not self.W[n]:
                    continue
                step = math.hypot(di, dj) * self.r
                pen = 1.0 + 2.0 * max(0.0, 0.45 - float(self.E[n]))
                nd = dist[c] + step * pen
                if nd < dist.get(n, 1e9):
                    dist[n] = nd
                    prev[n] = c
                    h = math.hypot(n[0] - g_w[0], n[1] - g_w[1]) * self.r
                    heapq.heappush(pq, (nd + h, n))
        if g_w not in dist:
            return None
        cells = [g_w]
        while cells[-1] != s_w:
            cells.append(prev[cells[-1]])
        pts = [self.xy(c) for c in reversed(cells)]
        pts = [tuple(start_xy)] + pts + [tuple(goal_xy)]
        # line-of-sight shortcutting (endpoints exempt: they may be tight)
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            def clear(i, j):
                short = math.hypot(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]) < 0.6
                return 0.18 if (i == 0 or j == len(pts) - 1) and short else LOS_CLEAR
            while j > i + 1 and not self._los(pts[i], pts[j], clear(i, j)):
                j -= 1
            out.append(pts[j])
            i = j
        return out


if __name__ == "__main__":
    import json
    p = NavPlanner()
    cat = json.load(open("/home/stickbot/projects/x2-groot-vla/gemma_ctl/kitchen_catalog.json"))
    for a, b in [((0, 0), cat["stove_counter"]["stand"]["xy"]),
                 (cat["stove_counter"]["stand"]["xy"], cat["sink_counter"]["stand"]["xy"]),
                 (cat["sink_counter"]["stand"]["xy"], cat["side_table"]["stand"]["xy"])]:
        path = p.plan(a, b)
        print([tuple(round(v, 2) for v in q) for q in path] if path else None)


def stand_for(planner: "NavPlanner", target_xy, robot_xy=None, reach=(0.24, 0.36), esdf_min=0.20):
    """Online standing spot for reaching target_xy (replaces the offline
    catalog): free cell with clearance at reach distance, nearest the robot."""
    best = None
    tx, ty = target_xy
    for a in np.radians(np.arange(0, 360, 10)):
        for d in np.arange(reach[0], reach[1] + 1e-6, 0.02):
            qx, qy = tx - d * math.cos(a), ty - d * math.sin(a)
            e = planner.esdf(qx, qy)
            if e < esdf_min:
                continue
            cost = -min(e, 0.35) * 4 + d
            if robot_xy is not None:
                cost += 0.1 * math.hypot(qx - robot_xy[0], qy - robot_xy[1])
            if best is None or cost < best[0]:
                best = (cost, (qx, qy), a)
            break
    return None if best is None else {"xy": best[1], "yaw": best[2]}
