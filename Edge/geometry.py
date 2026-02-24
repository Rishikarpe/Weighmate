"""Point-in-polygon check, matrix helpers, Kalman filter, and trilateration calculations."""

import math
import time

from config import (
    ANCHORS, WAREHOUSE_BOUNDARY,
    NLOS_RESIDUAL_THRESHOLD, NLOS_MAX_ITERATIONS,
    OUTLIER_DISTANCE_THRESHOLD,
    KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE
)


# ============== POINT-IN-POLYGON CHECK ==============
def point_in_polygon(x, y, polygon):
    """Ray-casting algorithm to check if point (x,y) is inside a polygon.
    polygon is a list of (x,y) tuples defining the boundary vertices."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ============== MATRIX HELPERS (pure Python, no numpy) ==============
def _mat_mul(A, B):
    """Multiply matrices A (m x n) and B (n x p)."""
    m, n, p = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(m)]


def _mat_transpose(A):
    """Transpose matrix A."""
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def _mat_add(A, B):
    """Element-wise add matrices A and B."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _inv2x2(M):
    """Invert a 2x2 matrix. Returns None if singular."""
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    if abs(det) < 1e-10:
        return None
    return [[M[1][1] / det, -M[0][1] / det],
            [-M[1][0] / det, M[0][0] / det]]


# ============== KALMAN FILTER ==============
class KalmanFilter2D:
    """2D Kalman filter with constant-velocity model for position smoothing.

    State vector: [x, y, vx, vy]
    Measurement:  [x, y] from trilateration
    """

    def __init__(self, process_noise, measurement_noise):
        self.state = [0.0, 0.0, 0.0, 0.0]
        # High initial uncertainty
        self.P = [[100, 0, 0, 0],
                   [0, 100, 0, 0],
                   [0, 0, 10, 0],
                   [0, 0, 0, 10]]
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initialized = False
        self.last_time = None

    def update(self, meas_x, meas_y):
        """Feed a new trilateration measurement. Returns filtered (x, y)."""
        now = time.time()

        if not self.initialized:
            self.state = [meas_x, meas_y, 0.0, 0.0]
            self.initialized = True
            self.last_time = now
            return meas_x, meas_y

        dt = now - self.last_time
        if dt <= 0:
            dt = 0.1
        self.last_time = now

        # --- PREDICT ---
        F = [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1,  0],
             [0, 0, 0,  1]]

        q = self.process_noise
        Q = [[q * dt**2, 0, 0, 0],
             [0, q * dt**2, 0, 0],
             [0, 0, q, 0],
             [0, 0, 0, q]]

        # x_pred = F @ state
        x_pred = [sum(F[i][k] * self.state[k] for k in range(4)) for i in range(4)]

        # P_pred = F @ P @ F^T + Q
        P_pred = _mat_add(_mat_mul(_mat_mul(F, self.P), _mat_transpose(F)), Q)

        # --- UPDATE ---
        # H = [[1,0,0,0],[0,1,0,0]], so H @ x = x[:2], H @ P @ H^T = P[:2,:2]
        r = self.measurement_noise
        R = [[r, 0], [0, r]]

        # Innovation y = z - H @ x_pred
        y = [meas_x - x_pred[0], meas_y - x_pred[1]]

        # S = P_pred[:2,:2] + R
        S = [[P_pred[0][0] + R[0][0], P_pred[0][1] + R[0][1]],
             [P_pred[1][0] + R[1][0], P_pred[1][1] + R[1][1]]]

        S_inv = _inv2x2(S)
        if S_inv is None:
            self.state = x_pred
            self.P = P_pred
            return x_pred[0], x_pred[1]

        # K = P_pred @ H^T @ S^-1  (H^T selects first 2 cols of P_pred)
        PH_T = [[P_pred[i][0], P_pred[i][1]] for i in range(4)]
        K = _mat_mul(PH_T, S_inv)

        # State update: x = x_pred + K @ y
        self.state = [x_pred[i] + K[i][0] * y[0] + K[i][1] * y[1] for i in range(4)]

        # Covariance update: P = (I - K @ H) @ P_pred
        KH = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            KH[i][0] = K[i][0]
            KH[i][1] = K[i][1]
        I_KH = [[(1.0 if i == j else 0.0) - KH[i][j] for j in range(4)] for i in range(4)]
        self.P = _mat_mul(I_KH, P_pred)

        return self.state[0], self.state[1]


# ============== TRILATERATION (Weighted LS + Outlier Rejection + NLOS) ==============
def _solve_triplet(a1, r1, a2, r2, a3, r3):
    """Solve position from a single triplet of anchors. Returns (x, y) or None."""
    p1, p2, p3 = ANCHORS[a1], ANCHORS[a2], ANCHORS[a3]

    Av = 2 * (p2['x'] - p1['x'])
    Bv = 2 * (p2['y'] - p1['y'])
    Cv = r1**2 - r2**2 - p1['x']**2 + p2['x']**2 - p1['y']**2 + p2['y']**2
    Dv = 2 * (p3['x'] - p2['x'])
    Ev = 2 * (p3['y'] - p2['y'])
    Fv = r2**2 - r3**2 - p2['x']**2 + p3['x']**2 - p2['y']**2 + p3['y']**2

    denom = Av * Ev - Bv * Dv
    if abs(denom) < 0.001:
        return None

    x = (Cv * Ev - Fv * Bv) / denom
    y = (Av * Fv - Dv * Cv) / denom

    if not point_in_polygon(x, y, WAREHOUSE_BOUNDARY):
        return None

    return (x, y)


def _weighted_trilaterate(valid):
    """Compute weighted trilateration with outlier rejection.

    Args:
        valid: dict {anchor_id: distance} of anchors to use.
    Returns:
        {'x': float, 'y': float} or None.
    """
    anchors = list(valid.items())
    if len(anchors) < 3:
        return None

    # Step 1: Solve all C(n,3) triplets with inverse-distance weighting
    results = []  # (x, y, weight)
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            for k in range(j + 1, len(anchors)):
                a1, r1 = anchors[i]
                a2, r2 = anchors[j]
                a3, r3 = anchors[k]

                pos = _solve_triplet(a1, r1, a2, r2, a3, r3)
                if pos:
                    avg_dist = (r1 + r2 + r3) / 3.0
                    weight = 1.0 / (avg_dist ** 2) if avg_dist > 0.1 else 1.0
                    results.append((pos[0], pos[1], weight))

    if not results:
        return None

    # Step 2: Outlier rejection — discard results far from median
    if len(results) >= 3:
        xs = sorted(r[0] for r in results)
        ys = sorted(r[1] for r in results)
        median_x = xs[len(xs) // 2]
        median_y = ys[len(ys) // 2]

        filtered = []
        for x, y, w in results:
            dist_from_median = math.sqrt((x - median_x)**2 + (y - median_y)**2)
            if dist_from_median <= OUTLIER_DISTANCE_THRESHOLD:
                filtered.append((x, y, w))
            else:
                print(f"  [OUTLIER] Rejected ({x:.2f}, {y:.2f}), {dist_from_median:.2f}m from median")

        if filtered:
            results = filtered

    # Step 3: Weighted average (closer anchors contribute more)
    total_weight = sum(w for _, _, w in results)
    if total_weight < 1e-10:
        return None

    avg_x = sum(x * w for x, y, w in results) / total_weight
    avg_y = sum(y * w for x, y, w in results) / total_weight

    return {'x': round(avg_x, 3), 'y': round(avg_y, 3)}


def trilaterate(distances):
    """Calculate position using weighted least squares, NLOS detection, and outlier rejection.

    Pipeline:
        1. Weighted trilateration from all valid anchor triplets
        2. Median-based outlier rejection
        3. NLOS residual check — remove anchors with large positive residuals
        4. Recompute without NLOS anchors (up to NLOS_MAX_ITERATIONS passes)

    Kalman filtering is applied separately after this function returns.
    """
    valid = {k: v for k, v in distances.items() if v is not None and k in ANCHORS}
    if len(valid) < 3:
        print(f"  [TRI] Only {len(valid)} valid anchors, need 3+")
        return None

    pos = None
    for iteration in range(NLOS_MAX_ITERATIONS + 1):
        pos = _weighted_trilaterate(valid)
        if pos is None:
            print(f"  [TRI] No valid results from {len(valid)} anchors")
            return None

        # On last allowed iteration, skip NLOS check and return current estimate
        if iteration == NLOS_MAX_ITERATIONS:
            break

        # NLOS residual check: flag anchors whose reported distance is much
        # longer than expected (signal bounced off obstacles)
        nlos_anchors = []
        for anchor_id, reported_dist in valid.items():
            ax = ANCHORS[anchor_id]['x']
            ay = ANCHORS[anchor_id]['y']
            expected_dist = math.sqrt((pos['x'] - ax)**2 + (pos['y'] - ay)**2)
            residual = reported_dist - expected_dist
            if residual > NLOS_RESIDUAL_THRESHOLD:
                nlos_anchors.append(anchor_id)
                print(f"  [NLOS] Anchor {anchor_id}: residual=+{residual:.2f}m "
                      f"(reported={reported_dist:.2f}, expected={expected_dist:.2f})")

        if not nlos_anchors:
            break  # All anchors are clean

        # Remove NLOS anchors and recompute
        for a in nlos_anchors:
            del valid[a]

        if len(valid) < 3:
            print(f"  [NLOS] Only {len(valid)} anchors remain after NLOS removal, keeping previous estimate")
            break

        print(f"  [NLOS] Recomputing without {nlos_anchors} (pass {iteration + 1})")

    return pos
