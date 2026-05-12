"""Internal Bessel function utilities.

Pure torch implementation — no scipy. Used by SO2onDisk for disk harmonic
computations. Not part of the public API.

Provides:
  bessel_jn       — J_n(x) for integer order n >= 0 (torch tensor)
  bessel_jn_zeros — first k positive roots of J_n(x) = 0
  compute_all_bessel_roots — roots for all orders 0..n_max in a single pass
"""

import math

import torch


def bessel_jn(n: int, x: torch.Tensor) -> torch.Tensor:
    """Compute J_n(x) for integer order n >= 0 via forward recurrence.

    Uses torch.special.bessel_j0 and bessel_j1 as base cases and the
    standard recurrence J_{k+1}(x) = (2k/x)*J_k(x) - J_{k-1}(x).

    Forward recurrence is stable when x >= n, which holds for our use
    case (evaluating at Bessel root * r where r in [0, 1]).

    Args:
        n: Non-negative integer order.
        x: Argument tensor (any shape).

    Returns:
        J_n(x) with same shape and dtype as x.
    """
    if n < 0:
        raise ValueError(f'Order n must be >= 0, got {n}')

    if n == 0:
        return torch.special.bessel_j0(x)
    if n == 1:
        return torch.special.bessel_j1(x)

    j_prev = torch.special.bessel_j0(x)
    j_curr = torch.special.bessel_j1(x)

    for k in range(1, n):
        safe_x = torch.where(x == 0, torch.ones_like(x), x)
        j_next = (2.0 * k / safe_x) * j_curr - j_prev
        j_next = torch.where(x == 0, torch.zeros_like(j_next), j_next)
        j_prev = j_curr
        j_curr = j_next

    return j_curr


def _jn_scalar(n: int, x: float) -> float:
    """Fast scalar evaluation of J_n(x) using raw math."""
    if n == 0:
        return torch.special.bessel_j0(torch.tensor(x, dtype=torch.float64)).item()
    if n == 1:
        return torch.special.bessel_j1(torch.tensor(x, dtype=torch.float64)).item()

    xt = torch.tensor(x, dtype=torch.float64)
    j_prev = torch.special.bessel_j0(xt).item()
    j_curr = torch.special.bessel_j1(xt).item()

    if x == 0:
        return 0.0

    for k in range(1, n):
        j_next = (2.0 * k / x) * j_curr - j_prev
        j_prev = j_curr
        j_curr = j_next

    return j_curr


def _djn_scalar(n: int, x: float) -> float:
    """Scalar J_n'(x) = (J_{n-1}(x) - J_{n+1}(x)) / 2."""
    if n == 0:
        return -_jn_scalar(1, x)
    return (_jn_scalar(n - 1, x) - _jn_scalar(n + 1, x)) / 2.0


def _bisect_newton(n: int, a: float, b: float) -> float:
    """Find root of J_n in bracket [a, b] using Newton + bisection."""
    fa = _jn_scalar(n, a)
    fb = _jn_scalar(n, b)

    if abs(fa) < 1e-15:
        return a
    if abs(fb) < 1e-15:
        return b
    if fa * fb > 0:
        return (a + b) / 2.0

    x = (a + b) / 2.0
    for _ in range(80):
        fx = _jn_scalar(n, x)
        if abs(fx) < 1e-15:
            return x

        dfx = _djn_scalar(n, x)

        if abs(dfx) > 1e-30:
            x_new = x - fx / dfx
        else:
            x_new = x

        if a < x_new < b:
            x = x_new
        else:
            x = (a + b) / 2.0

        fx = _jn_scalar(n, x)
        if fa * fx < 0:
            b = x
            fb = fx
        else:
            a = x
            fa = fx

        if (b - a) < 1e-14 * max(abs(a), 1.0):
            return (a + b) / 2.0

    return (a + b) / 2.0


def _mcmahon_zeros_j0(num_zeros: int) -> list[float]:
    """McMahon expansion for J_0 roots — highly accurate for all k."""
    if num_zeros <= 0:
        return []
    s = torch.arange(1, num_zeros + 1, dtype=torch.float64)
    beta = math.pi * (s - 0.25)
    z = beta - 1.0 / (8.0 * beta)
    for _ in range(10):
        fz = bessel_jn(0, z)
        dfz = -bessel_jn(1, z)
        safe_dfz = torch.where(dfz.abs() < 1e-30, torch.ones_like(dfz), dfz)
        dz = fz / safe_dfz
        dz = torch.where(dfz.abs() < 1e-30, torch.zeros_like(dz), dz)
        z = z - dz
        if (dz.abs() / z.abs().clamp(min=1.0)).max() < 1e-14:
            break
    return z.tolist()


def _bisect_newton_batch(n: int, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized root-finding for J_n in brackets [a, b].

    Args:
        n: Bessel order.
        a: Lower bracket endpoints, shape (num_roots,).
        b: Upper bracket endpoints, shape (num_roots,).

    Returns:
        Roots tensor, shape (num_roots,).
    """
    fa = bessel_jn(n, a)
    fb = bessel_jn(n, b)

    exact_a = fa.abs() < 1e-15
    exact_b = fb.abs() < 1e-15
    no_sign_change = fa * fb > 0

    x = (a + b) / 2.0

    for _ in range(80):
        fx = bessel_jn(n, x)
        if n == 0:
            dfx = -bessel_jn(1, x)
        else:
            dfx = (bessel_jn(n - 1, x) - bessel_jn(n + 1, x)) / 2.0

        newton_ok = dfx.abs() > 1e-30
        x_newton = torch.where(newton_ok, x - fx / dfx.clamp_min(1e-30).copysign(dfx), x)
        in_bracket = (a < x_newton) & (x_newton < b)
        x = torch.where(in_bracket & newton_ok, x_newton, (a + b) / 2.0)

        fx = bessel_jn(n, x)
        go_left = fa * fx < 0
        b = torch.where(go_left, x, b)
        fb = torch.where(go_left, fx, fb)
        a = torch.where(~go_left, x, a)
        fa = torch.where(~go_left, fx, fa)

        converged = (b - a) < 1e-14 * a.abs().clamp(min=1.0)
        if converged.all():
            break

    result = (a + b) / 2.0
    result = torch.where(exact_a, a, result)
    result = torch.where(exact_b, b, result)
    result = torch.where(no_sign_change & ~exact_a & ~exact_b, (a + b) / 2.0, result)
    return result


def compute_all_bessel_roots(n_max: int, k_max: int) -> dict[int, list[float]]:
    """Compute Bessel roots for all orders 0..n_max using interlacing.

    Uses the interlacing property j_{n-1,k} < j_{n,k} < j_{n-1,k+1}
    to bracket each root, then Newton-bisection within the bracket.
    All orders are computed in a single pass from J_0 upward, sharing
    intermediate results.

    Args:
        n_max: Maximum Bessel order.
        k_max: Maximum number of roots per order.

    Returns:
        Dict mapping order n -> list of first k_max roots (or fewer if
        not enough brackets exist).
    """
    total_j0 = k_max + n_max + 5
    prev_roots_list = _mcmahon_zeros_j0(total_j0)

    all_roots: dict[int, list[float]] = {0: prev_roots_list[:k_max]}

    prev_roots = torch.tensor(prev_roots_list, dtype=torch.float64)

    for order in range(1, n_max + 1):
        num_needed = k_max + (n_max - order) + 3
        num_brackets = min(num_needed, len(prev_roots) - 1)
        if num_brackets <= 0:
            all_roots[order] = []
            prev_roots = torch.tensor([], dtype=torch.float64)
            continue
        a = prev_roots[:num_brackets]
        b = prev_roots[1 : num_brackets + 1]
        curr_roots = _bisect_newton_batch(order, a, b)
        prev_roots = curr_roots
        all_roots[order] = curr_roots[:k_max].tolist()

    return all_roots


def bessel_jn_zeros(n: int, num_zeros: int) -> torch.Tensor:
    """Compute the first `num_zeros` positive roots of J_n(x) = 0.

    For single-order queries. For multi-order queries, use
    compute_all_bessel_roots() which is more efficient.

    Args:
        n: Non-negative integer order.
        num_zeros: Number of positive roots to compute.

    Returns:
        1D float64 tensor of shape (num_zeros,) with roots in ascending order.
    """
    if num_zeros <= 0:
        return torch.zeros(0, dtype=torch.float64)

    all_roots = compute_all_bessel_roots(n, num_zeros)
    roots = all_roots.get(n, [])
    return torch.tensor(roots[:num_zeros], dtype=torch.float64)
