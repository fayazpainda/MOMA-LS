"""
Wei2021 Reproduction -- structured reference-point generation

Two modes, selected by ``generate_reference_points(mode=...)``:

    mode="uniform"     Das-Dennis structured directions over the 3-objective
                       simplex (Cmax, TWET, NEC). This is the validated
                       U-NSGA-III baseline and is byte-for-byte identical to
                       the grid produced by the historical
                       ``unsga3.generate_reference_points_3d(divisions=p)``.

    mode="ahp_guided"  The same Das-Dennis grid, then FILTERED to keep only
                       directions within ``radius`` of an AHP-derived
                       weight vector, concentrating the search near a
                       manager-specified trade-off region.

Only the reference-point *set* differs between the two modes. Population size,
generations, crossover, mutation, and the (corrected) objective functions are
untouched -- that isolation is the whole point of the experiment.

Why filtering is safe for U-NSGA-III association
------------------------------------------------
``nsga3_selection.associate_population`` consumes each reference point purely
as a *direction*: ``perpendicular_distance`` normalises it to a unit vector
before use. So (a) keeping a subset of the grid changes only which directions
solutions may associate with, and (b) whether the retained points sum to 1 is
cosmetic. We renormalise the retained set to the simplex anyway, for clarity.
"""

import warnings

import numpy as np


# Default Das-Dennis division number. For M = 3 objectives, p = 12 gives
# H = C(p+2, 2) = 91 directions, matching the validated baseline (N = 100 > H).
DEFAULT_DIVISIONS = 12

# Default filtering radius for ahp_guided mode, as a Euclidean distance on the
# 3-objective simplex (all points, and the weight vector, sum to 1).
#
# Reasoning: the simplex triangle has vertices (1,0,0), (0,1,0), (0,0,1); its
# vertex-to-centroid distance is sqrt(6)/3 ~ 0.816 and its max side length is
# sqrt(2) ~ 1.414. The p=12 Das-Dennis grid places 91 points over that
# triangle (~105 points per unit area), with nearest-neighbour spacing
# ~ sqrt(2)/12 ~ 0.118. A radius of 0.30 therefore retains on the order of a
# few tens of directions for an interior weight vector, and still comfortably
# more than the >=10 diversity floor for a weight vector near a simplex edge
# (e.g. the default AHP weights ~ (0.62, 0.27, 0.11)). It is a parameter so it
# can be swept later to probe the diversity/convergence trade-off.
DEFAULT_RADIUS = 0.30

# Below this many retained directions, warn: too small a radius can starve the
# algorithm of diversity, which is one of the effects we want to detect.
MIN_RECOMMENDED_POINTS = 10


# ============================================================
# Das-Dennis grid (baseline)
# ============================================================

def das_dennis_grid(divisions):
    """
    Das-Dennis structured reference directions for M = 3 objectives.

    Every direction satisfies z1 + z2 + z3 = 1 with z_i >= 0. The iteration
    order (i, then j, with k = divisions - i - j) is kept identical to the
    historical ``unsga3.generate_reference_points_3d`` so ``mode="uniform"``
    reproduces the validated baseline exactly.
    """

    if divisions <= 0:
        raise ValueError("divisions must be greater than zero")

    reference_points = []

    for i in range(divisions + 1):
        for j in range(divisions - i + 1):
            k = divisions - i - j
            reference_points.append([
                i / divisions,
                j / divisions,
                k / divisions,
            ])

    return reference_points


# ============================================================
# Filtering helpers
# ============================================================

def _as_simplex_weight_vector(weight_vector):
    """Coerce and validate an M=3 weight vector, renormalised to sum to 1."""

    w = np.asarray(weight_vector, dtype=float)

    if w.shape != (3,):
        raise ValueError(
            f"weight_vector must have length 3 (Cmax, TWET, NEC); got shape "
            f"{w.shape}."
        )

    if not np.all(np.isfinite(w)):
        raise ValueError("weight_vector contains non-finite entries.")

    if np.any(w < 0.0):
        raise ValueError("weight_vector must be non-negative.")

    total = w.sum()
    if total <= 0.0:
        raise ValueError("weight_vector must have a positive sum.")

    return w / total


def reference_target_from_weights(weights):
    """
    Map AHP preference weights to the reference-simplex point the guided search
    should cluster around.

    Why this inversion is necessary
    -------------------------------
    U-NSGA-III normalises objectives with the IDEAL point at the origin, so a
    solution is *good* on objective i exactly when its normalised i-th
    coordinate is *small*. A reference direction attracts solutions along its
    ray; a direction with a LARGE i-th coordinate therefore attracts solutions
    that are BAD (large value) on objective i.

    A manager who weights objective i highly wants to MINIMISE it, i.e. wants
    solutions with a small i-th coordinate. So the reference points must be
    concentrated where the important objective's coordinate is *small* -- the
    opposite end of the simplex from the weight vector itself. Clustering the
    reference points on the raw weight vector steers the front the wrong way
    (empirically: the prioritised objective's mean gets worse).

    We use the normalised reciprocal, target_i proportional to 1 / w_i, which:
      * gives the most important objective the smallest reference coordinate,
      * leaves a balanced weight (1/3, 1/3, 1/3) unchanged, and
      * preserves the preference ordering.
    """

    w = _as_simplex_weight_vector(weights)
    inverse = 1.0 / w
    return (inverse / inverse.sum()).tolist()


def _distance(points, target, metric):
    """
    Distance from each row of ``points`` (n x 3) to ``target`` (3,).

        metric="euclidean" -> Euclidean distance on the simplex
        metric="angular"   -> angle in radians between direction vectors
    """

    pts = np.asarray(points, dtype=float)

    if metric == "euclidean":
        return np.linalg.norm(pts - target, axis=1)

    if metric == "angular":
        target_norm = target / np.linalg.norm(target)
        point_norms = np.linalg.norm(pts, axis=1)
        # Guard the (unused in practice) all-zero point; grid points are all
        # non-negative and sum to 1, so norms are strictly positive.
        point_norms = np.where(point_norms <= 1e-12, 1.0, point_norms)
        unit_points = pts / point_norms[:, None]
        cosines = np.clip(unit_points @ target_norm, -1.0, 1.0)
        return np.arccos(cosines)

    raise ValueError(f"Unknown metric {metric!r}; use 'euclidean' or "
                     f"'angular'.")


def _renormalise_to_simplex(points):
    """Renormalise each point to sum to 1 (defensive; grid points already do)."""

    pts = np.asarray(points, dtype=float)
    row_sums = pts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 1e-12, 1.0, row_sums)
    return (pts / row_sums).tolist()


# ============================================================
# Public entry point
# ============================================================

def generate_reference_points(
        mode="uniform",
        weight_vector=None,
        p=DEFAULT_DIVISIONS,
        radius=None,
        metric="euclidean",
):
    """
    Generate structured reference directions for U-NSGA-III.

    Parameters
    ----------
    mode:
        "uniform"     -> full Das-Dennis grid (validated baseline).
        "ahp_guided"  -> Das-Dennis grid filtered near ``weight_vector``.
    weight_vector:
        Length-3 AHP weight vector (Cmax, TWET, NEC). Required for
        "ahp_guided"; ignored for "uniform".
    p:
        Das-Dennis division number (H = C(p+2, 2) for M = 3).
    radius:
        Filtering radius for "ahp_guided". Interpreted per ``metric``. If None,
        DEFAULT_RADIUS is used (Euclidean) -- for the angular metric a radius
        must be supplied explicitly, since the default is a Euclidean value.
    metric:
        "euclidean" (simplex distance, default) or "angular" (radians).

    Returns
    -------
    list of [z1, z2, z3]
        Reference directions summing to 1.
    """

    grid = das_dennis_grid(p)

    if mode == "uniform":
        return grid

    if mode != "ahp_guided":
        raise ValueError(
            f"Unknown mode {mode!r}; use 'uniform' or 'ahp_guided'."
        )

    # --- ahp_guided ---------------------------------------------------------

    if weight_vector is None:
        raise ValueError(
            "mode='ahp_guided' requires a weight_vector (e.g. from "
            "ahp.derive_weights)."
        )

    target = _as_simplex_weight_vector(weight_vector)

    if radius is None:
        if metric != "euclidean":
            raise ValueError(
                "A radius must be supplied explicitly for metric="
                f"{metric!r}; the default radius is a Euclidean value."
            )
        radius = DEFAULT_RADIUS

    if radius <= 0.0:
        raise ValueError("radius must be greater than zero.")

    grid_array = np.asarray(grid, dtype=float)
    distances = _distance(grid_array, target, metric)

    keep_mask = distances <= radius
    retained = grid_array[keep_mask]

    if len(retained) == 0:
        # A radius too small to catch any grid point would leave the algorithm
        # with no reference directions at all. Fall back to the single closest
        # direction so association still works, and warn loudly.
        nearest_index = int(np.argmin(distances))
        retained = grid_array[nearest_index:nearest_index + 1]
        warnings.warn(
            f"ahp_guided reference filtering with radius={radius} ({metric}) "
            f"retained no grid directions; falling back to the single closest "
            f"direction {retained[0].tolist()}. Increase the radius.",
            stacklevel=2,
        )
    elif len(retained) < MIN_RECOMMENDED_POINTS:
        warnings.warn(
            f"ahp_guided reference filtering retained only {len(retained)} "
            f"direction(s) (radius={radius}, {metric}); this is below the "
            f"recommended diversity floor of {MIN_RECOMMENDED_POINTS} and may "
            f"starve U-NSGA-III of diversity. Consider a larger radius.",
            stacklevel=2,
        )

    return _renormalise_to_simplex(retained)
