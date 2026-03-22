import numpy as np
import math
from dataclasses import dataclass

@dataclass
class OrbitResult:
    t0: float
    residual: float
    P0: np.ndarray
    points: np.ndarray  # P0..PN
    v_end: np.ndarray

def P_of_t(t: float, a: float, b: float) -> np.ndarray:
    """Point on ellipse boundary."""
    return np.array([a * math.cos(t), b * math.sin(t)], dtype=float)

def next_intersection_on_ellipse(p: np.ndarray, v: np.ndarray, a: float, b: float):
    """
    Given p on ellipse and direction v, find next intersection.
    Includes a projection step to prevent floating-point drift.
    """
    x, y = p
    vx, vy = v

    A = (vx * vx) / (a * a) + (vy * vy) / (b * b)
    B = (x * vx) / (a * a) + (y * vy) / (b * b)

    # 這裡原本假設常數項 C = x^2/a^2 + y^2/b^2 - 1 = 0
    # s = -2B / A 只有在 C 嚴格為 0 時才成立。
    s = -2.0 * B / A
    p_next = p + s * v

    # 【修正1】將點強制投影回橢圓上，消除浮點數誤差累積
    r = math.sqrt((p_next[0] / a)**2 + (p_next[1] / b)**2)
    p_next = p_next / r

    return p_next, s

def reflect_on_ellipse(p: np.ndarray, v: np.ndarray, a: float, b: float) -> np.ndarray:
    """Specular reflection on ellipse boundary."""
    x, y = p
    n = np.array([x / (a * a), y / (b * b)], dtype=float)
    nn = np.linalg.norm(n)
    if nn == 0:
        raise ValueError("Normal became zero.")
    n = n / nn
    return v - 2.0 * np.dot(v, n) * n

def iterate_orbit(t0: float, v0: np.ndarray, N: int, a: float, b: float):
    """Start at P(t0) with initial direction v0, bounce N times."""
    p = P_of_t(t0, a, b)
    v = v0.copy()

    pts = [p.copy()]
    steps = []

    for _ in range(N):
        p, s = next_intersection_on_ellipse(p, v, a, b)
        pts.append(p.copy())
        steps.append(s)
        v = reflect_on_ellipse(p, v, a, b)

    return np.array(pts), np.array(steps), v

def residual_periodic(t0: float, v0: np.ndarray, N: int, a: float, b: float,
                      min_step: float = 1e-3) -> float:
    """Residual = ||P_N - P_0|| + ||v_end - v0||"""
    pts, steps, v_end = iterate_orbit(t0, v0, N, a, b)
    if np.any(steps < min_step):
        return 1e9  # penalty for grazing/degenerate
    p0 = pts[0]
    return np.linalg.norm(pts[-1] - p0) + np.linalg.norm(v_end - v0)


def refine_t0(t0: float, v0: np.ndarray, N: int, a: float, b: float,
              min_step: float, iters: int = 200) -> tuple[float, float]:
    """Local refinement around t0."""
    t = t0 % (2.0 * math.pi)
    step = 0.1
    best = residual_periodic(t, v0, N, a, b, min_step=min_step)

    for _ in range(iters):
        improved = False
        for dt in (0.0, -step, step):
            tt = (t + dt) % (2.0 * math.pi)
            r = residual_periodic(tt, v0, N, a, b, min_step=min_step)
            if r < best:
                best = r
                t = tt
                improved = True
        if not improved:
            step *= 0.6
        if step < 1e-12:
            break
    return t, best

def cluster_angles(ts: list[float], eps: float = 1e-6) -> list[float]:
    """Cluster angles on circle [0,2pi)."""
    if not ts:
        return []
    ts = [t % (2.0 * math.pi) for t in ts]
    ts.sort()

    clusters = []
    cur = [ts[0]]
    for t in ts[1:]:
        if abs(t - cur[-1]) <= eps:
            cur.append(t)
        else:
            clusters.append(cur)
            cur = [t]
    clusters.append(cur)

    if len(clusters) >= 2:
        if (clusters[0][0] + 2.0 * math.pi) - clusters[-1][-1] <= eps:
            merged = clusters[-1] + [x + 2.0 * math.pi for x in clusters[0]]
            clusters = [merged] + clusters[1:-1]

    reps = []
    for cl in clusters:
        reps.append(float(np.mean(cl) % (2.0 * math.pi)))
    reps.sort()
    return reps

def enumerate_solutions(a: float, b: float, m: float, N: int,
                        grid: int = 8000,
                        candidate_factor: float = 5.0,
                        tol: float = 1e-6,
                        min_step: float = 1e-3,
                        refine_iters: int = 250) -> list[OrbitResult]:

    # 【修正2】由斜率產生兩個可能的方向向量 (向右與向左)
    v_right = np.array([1.0, m])
    v_right /= np.linalg.norm(v_right)
    v_left = np.array([-1.0, -m])
    v_left /= np.linalg.norm(v_left)

    directions = [v_right, v_left]
    out = []

    ts = np.linspace(0.0, 2.0 * math.pi, grid, endpoint=False)

    for v0 in directions:
        rs = np.array([residual_periodic(float(t), v0, N, a, b, min_step=min_step) for t in ts])

        finite_rs = rs[np.isfinite(rs) & (rs < 1e8)]
        if finite_rs.size == 0:
            continue

        rmin = float(np.min(finite_rs))
        loose = max(candidate_factor * rmin, 10.0 * tol)

        cand = []
        for i in range(grid):
            r0 = rs[i]
            if r0 > loose:
                continue
            rL = rs[(i - 1) % grid]
            rR = rs[(i + 1) % grid]
            if r0 <= rL and r0 <= rR:
                cand.append(float(ts[i]))

        refined = []
        for t0 in cand:
            tr, rr = refine_t0(t0, v0, N, a, b, min_step=min_step, iters=refine_iters)
            if rr <= tol:
                refined.append(tr)

        reps = cluster_angles(refined, eps=max(1e-7, 10.0 * tol))

        for t0 in reps:
            pts, steps, v_end = iterate_orbit(t0, v0, N, a, b)
            r = np.linalg.norm(pts[-1] - pts[0]) + np.linalg.norm(v_end - v0)
            out.append(OrbitResult(t0=t0, residual=float(r), P0=pts[0], points=pts, v_end=v_end))

    out.sort(key=lambda z: (z.residual, z.t0))
    return out


def main():
    print("==============================================")
    print(" Ellipse billiard: enumerate ALL N-periodic")
    print("==============================================\n")

    a = float(input("Input a (semi-major axis, e.g. 5): ").strip())
    b = float(input("Input b (semi-minor axis, e.g. 4): ").strip())
    if a <= 0 or b <= 0:
        raise ValueError("a,b must be positive.")

    # 【修正3】移除了 a, b 強制交換的邏輯

    m = float(input("Input initial slope m (dy/dx, e.g. -3.325): ").strip())
    N = int(input("Input number of reflections N (e.g. 5): ").strip())
    if N <= 0:
        raise ValueError("N must be positive.")

    grid = 8000
    tol = 1e-6
    min_step = 1e-3

    print("\nRunning enumeration...")
    sols = enumerate_solutions(a, b, m, N, grid=grid, tol=tol, min_step=min_step)

    print("\n==================== RESULTS ====================")
    print(f"Ellipse: x^2/{a*a:g} + y^2/{b*b:g} = 1")
    print(f"Slope m = {m}")
    print(f"N = {N}")
    print(f"Found {len(sols)} solution(s).\n")

    for k, sol in enumerate(sols, 1):
        x0, y0 = sol.P0
        print(f"[Solution {k}]")
        print(f"  t0 = {sol.t0:.15f} rad")
        print(f"  P0 = ({x0:.10f}, {y0:.10f})")
        print(f"  residual = {sol.residual:.3e}")
        print("  Points P0..PN:")
        for i, p in enumerate(sol.points):
            print(f"    P{i} = ({p[0]: .10f}, {p[1]: .10f})")
        print()

if __name__ == "__main__":
    main()