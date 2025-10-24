"""

Vector Space Architect — Step-by-step implementation  

This file implements a full, well-documented pipeline to:
 1. Build an 8-dimensional traffic flow model (4 approaches × 2 directions).
 2. Construct transition (turning) matrix P and form the linear system (I - P)x = b.
 3. Solve the linear system using:
    - a demonstrative pure-Python Gaussian elimination (with partial pivoting)
    - numpy's solver (for robustness)
 4. Perform eigenvalue analysis on P to check stability (spectral radius < 1).
 5. Validate results and provide simple checks / residuals.



Design notes (high level):
 - We treat flows as volumes on 8 links (x has length 8). Some fraction of vehicles exit
   the modeled network (sink) so the columns of P sum to less than 1 (this ensures existence
   of a unique steady-state solution in normal cases).
 - The steady-state equation is: x = b + P @ x  =>  (I - P) x = b
 - We build b from exogenous arrival rates per approach (split into the two directions
   according to given turning ratios).

This code is intentionally verbose (lots of comments and explanatory prints) so you can
follow it step-by-step for reports/demos.

"""

from typing import Dict, List, Tuple, Optional
import numpy as np


# ----------------------------- Utility / Indexing -----------------------------
# We'll index variables as follows:
# For approach i in [0,1,2,3] (clockwise: North, East, South, West):
#   idx = 2*i     => the "straight"/primary outbound link from approach i
#   idx = 2*i + 1 => the "right-turn" outbound link from approach i
# (This is a convention used in the matrix construction below.)

APPROACH_NAMES = ["North", "East", "South", "West"]
DIRECTION_NAMES = ["straight", "right_turn"]


# ----------------------------- Gaussian elimination -----------------------------

def gaussian_elimination(A: np.ndarray, b: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    Returns (x, steps) where steps is a list of human-readable operations performed.
    This function is educational — for production use numpy.linalg.solve is preferable.
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = A.shape[0]
    aug = np.hstack([A, b.reshape(-1, 1)])
    steps = []

    # Forward elimination
    for k in range(n):
        # Partial pivoting: find the row with max abs value in column k at or below k
        pivot_row = np.argmax(np.abs(aug[k:, k])) + k
        if abs(aug[pivot_row, k]) < 1e-12:
            steps.append(f"Column {k}: pivot nearly zero -> system may be singular or underdetermined.")
            continue

        if pivot_row != k:
            aug[[k, pivot_row], :] = aug[[pivot_row, k], :]
            steps.append(f"Swap row {k} with row {pivot_row} (partial pivoting).")

        pivot = aug[k, k]
        aug[k, :] = aug[k, :] / pivot
        steps.append(f"Normalize row {k} by pivot {pivot:.6g}.")

        for i in range(k + 1, n):
            factor = aug[i, k]
            if abs(factor) < 1e-15:
                continue
            aug[i, :] = aug[i, :] - factor * aug[k, :]
            steps.append(f"Eliminate row {i}, using factor {factor:.6g} from row {k}.")

    # Back substitution
    x = np.zeros(n)
    # If a row is all zeros in coefficients but has nonzero RHS => inconsistent;
    # if all zeros including RHS, it is underdetermined (infinitely many solutions). We'll handle simply.
    # Back substitute from bottom up
    for i in range(n - 1, -1, -1):
        row = aug[i, :n]
        rhs = aug[i, n]
        # If near-zero row
        if np.all(np.isclose(row, 0)):
            if abs(rhs) > 1e-9:
                steps.append(f"Row {i} reduced to 0 but RHS {rhs:.6g} != 0 -> inconsistent system.")
                raise np.linalg.LinAlgError("Inconsistent system detected during Gaussian elimination.")
            else:
                # Free variable: choose zero
                x[i] = 0.0
                steps.append(f"Row {i} all zeros -> free variable set to 0.")
                continue

        # assume row has 1 on diagonal because we normalized; compute x[i]
        x[i] = rhs - np.dot(row[:i], x[:i]) - np.dot(row[i + 1 :], x[i + 1 :])
        # if diagonal not 1 due to numerical issues adjust
        if abs(row[i] - 1.0) > 1e-8:
            x[i] = x[i] / (row[i] if abs(row[i]) > 1e-12 else 1.0)
        steps.append(f"Back-substitution: x[{i}] = {x[i]:.6g}.")

    if verbose:
        for s in steps:
            print(s)

    return x, steps


# ----------------------------- Model Construction -----------------------------

def build_b_from_arrivals(arrival_rates: List[float], turning_ratios: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
    """
    Convert arrival_rates (length 4) into an 8-vector b by splitting arrivals per approach
    into the two directions according to turning_ratios.

    turning_ratios: list of 4 tuples (p_straight, p_right). They should sum to <= 1.
    The remaining fraction is considered as immediate exits (not entering any modeled link).
    """
    if turning_ratios is None:
        # default: 70% straight, 30% right for each approach
        turning_ratios = [(0.7, 0.3) for _ in range(4)]

    assert len(arrival_rates) == 4, "arrival_rates must be length 4 (one per approach)."
    b = np.zeros(8)
    for i, arr in enumerate(arrival_rates):
        p_straight, p_right = turning_ratios[i]
        b[2 * i] = arr * p_straight
        b[2 * i + 1] = arr * p_right
    return b


def build_transition_matrix(turning_ratios: Optional[List[Tuple[float, float]]] = None, exit_fraction: float = 0.2) -> np.ndarray:
    """
    Build an 8x8 transition matrix P where P[d, s] is the fraction of vehicles leaving link s that
    enter link d.

    exit_fraction is the fraction of vehicles that leave the modeled subnet (i.e., sink).
    The remaining (1 - exit_fraction) is distributed according to turning fractions.

    We use a simple topology:
      - straight from approach i goes to the straight link of the opposite approach (i+2)%4
      - right_turn from approach i goes to the right_turn link of the next approach (i+1)%4

    Column s sums to (1 - exit_fraction) for each s.
    """
    if turning_ratios is None:
        turning_ratios = [(0.6, 0.2) for _ in range(4)]  # remaining exit_fraction will be 0.2 by default

    P = np.zeros((8, 8))
    for i in range(4):
        src_straight = 2 * i
        src_right = 2 * i + 1
        p_straight, p_right = turning_ratios[i]

        # normalize so that p_straight + p_right <= (1 - exit_fraction). If not, scale down.
        total_wanted = p_straight + p_right
        available = 1.0 - exit_fraction
        if total_wanted > available and total_wanted > 0:
            scale = available / total_wanted
            p_straight *= scale
            p_right *= scale

        # destinations
        dest_straight = 2 * ((i + 2) % 4)       # straight goes to opposite approach's straight link
        dest_right = 2 * ((i + 1) % 4) + 1      # right-turn goes to next approach's right_turn link

        # fill columns: fraction arriving to dest from src
        P[dest_straight, src_straight] = p_straight
        P[dest_right, src_straight] = 0.0  # example: straight won't send to right-turn dest directly in this model

        P[dest_right, src_right] = p_right
        P[dest_straight, src_right] = 0.0  # example: right-turn won't send to straight dest directly

        # Note: columns may sum to < available because we left some >>0 to exits.
    return P


# ----------------------------- Solvers & Analysis -----------------------------

def form_system_from_intersection(intersection_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given intersection_data (expected keys: 'arrival_rates', optionally 'turning_ratios', 'exit_fraction'),
    return (A, b, P) where A = I - P and b is the exogenous arrivals vector.
    """
    arrival_rates = intersection_data.get("arrival_rates", [20, 15, 18, 12])
    turning_ratios = intersection_data.get("turning_ratios", None)
    exit_fraction = intersection_data.get("exit_fraction", 0.2)

    b = build_b_from_arrivals(arrival_rates, turning_ratios)
    P = build_transition_matrix(turning_ratios, exit_fraction)
    A = np.eye(8) - P
    return A, b, P


def solve_with_numpy(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Solve Ax = b using numpy; gracefully handle singular/ill-conditioned cases.
    Returns (x, method_description)
    """
    try:
        cond = np.linalg.cond(A)
        if cond > 1e12:
            # warn high condition number
            method = f"numpy.linalg.solve (warning: ill-conditioned matrix, cond={cond:.3e}). Using lstsq fallback."
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
        else:
            x = np.linalg.solve(A, b)
            method = f"numpy.linalg.solve (cond={cond:.3e})"
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        method = "numpy.linalg.lstsq fallback (matrix singular or nearly singular)"
    return x, method


def eigenvalue_analysis(P: np.ndarray) -> Dict[str, object]:
    """
    Analyze eigenvalues of the transition matrix P and return a small report.
    For the discrete-time link model x = b + P x, stability (no unbounded growth) requires spectral radius < 1.
    """
    eigvals = np.linalg.eigvals(P)
    spectral_radius = max(abs(eigvals))
    stable = spectral_radius < 1.0 - 1e-9  # margin
    return {
        "eigenvalues": eigvals,
        "spectral_radius": spectral_radius,
        "stable": stable,
        "interpretation": (
            "Stable (spectral radius < 1)" if stable else "Potentially unstable (spectral radius >= 1)"
        ),
    }


def validate_solution(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    Return basic validation metrics: residual norm and relative error estimate.
    """
    residual = A.dot(x) - b
    res_norm = np.linalg.norm(residual)
    b_norm = np.linalg.norm(b)
    rel_error = res_norm / (b_norm + 1e-12)
    return {"residual_norm": float(res_norm), "relative_error": float(rel_error)}


# ----------------------------- High-level API (for integration_main.py) -----------------------------

def solve_traffic_system(intersection_data: Dict, verbose: bool = False) -> Dict:
    """
    High-level function to be imported and used by integration_main.py

    Returns a dictionary with keys:
        'x' -> solution vector (length 8)
        'A', 'b', 'P' -> matrices used
        'numpy_method', 'gauss_steps' -> solver info
        'eigen_report', 'validation' -> analysis and checks
    """
    A, b, P = form_system_from_intersection(intersection_data)

    # Solve with numpy
    x_numpy, numpy_method = solve_with_numpy(A, b)

    # Solve with our educational Gaussian elimination (catch exceptions)
    gauss_steps = []
    try:
        x_gauss, gauss_steps = gaussian_elimination(A, b, verbose=False)
    except Exception as e:
        # In case of issues (singular/inconsistent) we fall back to numpy's solution
        x_gauss = x_numpy
        gauss_steps = [f"Gaussian elimination failed with: {e}. Using numpy result as fallback."]

    # Choose primary solution (numpy preferred for numerical stability)
    x = x_numpy

    eigen_report = eigenvalue_analysis(P)
    validation = validate_solution(A, x, b)

    if verbose:
        print("---- System (I - P) = A ----")
        print(A)
        print("---- b (exogenous arrivals split by direction) ----")
        print(b)
        print("---- Solution x (flows on 8 links) ----")
        for i in range(4):
            print(f"Approach {i} ({APPROACH_NAMES[i]}): straight = {x[2*i]:.3f}, right_turn = {x[2*i+1]:.3f}")
        print("---- Eigen report (P) ----")
        print(eigen_report)
        print("---- Validation ----")
        print(validation)

    return {
        "x": x,
        "A": A,
        "b": b,
        "P": P,
        "numpy_method": numpy_method,
        "gauss_steps": gauss_steps,
        "eigen_report": eigen_report,
        "validation": validation,
    }


# ----------------------------- Example / Demo -----------------------------

if __name__ == "__main__":
    # Try to import SAMPLE_INTERSECTION from shared_data.py if present
    try:
        import shared_data as sd

        SAMPLE = sd.SAMPLE_INTERSECTION
        print("Loaded SAMPLE_INTERSECTION from shared_data.py")
    except Exception:
        SAMPLE = {
            "points": [(-1 - 1j), (1 - 1j), (1 + 1j), (-1 + 1j)],
            "arrival_rates": [20, 15, 18, 12],
            "constraints": {"min_green": 20, "max_green": 60, "cycle_time": 120},
        }
        print("shared_data.py not importable — using built-in SAMPLE for demo.")

    # Add optional turning ratios and exit fraction to demonstrate customization
    SAMPLE["turning_ratios"] = [(0.6, 0.25), (0.65, 0.25), (0.7, 0.2), (0.6, 0.3)]
    SAMPLE["exit_fraction"] = 0.2

    print("Running solve_traffic_system(...) with demo data — verbose prints follow:\n")
    results = solve_traffic_system(SAMPLE, verbose=True)

    print("\nQuick summary:")
    for i in range(4):
        print(f"Approach {i} ({APPROACH_NAMES[i]}): straight = {results['x'][2*i]:.3f}, right = {results['x'][2*i+1]:.3f}")
    print("Spectral radius:", results["eigen_report"]["spectral_radius"]) 

    print("\nIf you'd like, I can:")
    print(" - adapt the turning matrix to match your actual intersection geometry")
    print(" - enforce bounds or linear constraints (e.g., capacity limits)")
    print(" - produce a LaTeX-ready step-by-step elimination trace for your report")

