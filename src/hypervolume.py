"""
Wei2021 Reproduction -- hypervolume convergence indicator

The codebase had no convergence indicator, so this adds hypervolume (HV) for
the 3-objective *minimisation* problem (Cmax, TWET, NEC), used to draw
per-generation convergence curves and compare final front quality.

Comparability across configurations
-----------------------------------
HV depends on the reference point, so for the vanilla-vs-AHP comparison to be
meaningful every configuration on a given instance must use the SAME reference
point and the SAME normalisation bounds. The intended workflow (see
ahp_comparison / the analysis script):

    1. Collect all objective vectors seen on an instance across BOTH configs,
       all seeds and all generations.
    2. ideal  = componentwise min, nadir = componentwise max of that union.
    3. Normalise objectives to [0, 1] with (ideal, nadir).
    4. Reference point = nadir * ref_factor in normalised space, i.e.
       (ref_factor, ref_factor, ref_factor) with ref_factor = 1.1.
    5. Report HV as a fraction of the reference box (in [0, 1]).

``normalized_hypervolume`` implements steps 3-5 given precomputed
(ideal, nadir). ``hypervolume`` is the raw 3-D primitive.
"""


# ============================================================
# Raw 3-D hypervolume (minimisation)
# ============================================================

def _pareto_max(points):
    """Keep only the non-dominated points under *maximisation* (all coords)."""

    kept = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if j == i:
                continue
            # q dominates p (maximisation) if q >= p everywhere and > somewhere.
            if (q[0] >= p[0] and q[1] >= p[1] and q[2] >= p[2] and
                    (q[0] > p[0] or q[1] > p[1] or q[2] > p[2])):
                dominated = True
                break
        if not dominated:
            kept.append(p)
    return kept


def _area_2d_union(yz_points):
    """
    Area of the union of origin-anchored rectangles [0, y] x [0, z]
    (maximisation). ``yz_points`` are (y, z) with y, z >= 0.
    """

    # Sort by y descending; accumulate as the z threshold rises.
    ordered = sorted(yz_points, key=lambda t: t[0], reverse=True)
    area = 0.0
    last_z = 0.0
    for y, z in ordered:
        if z > last_z:
            area += y * (z - last_z)
            last_z = z
    return area


def hypervolume(points, reference):
    """
    Hypervolume dominated by ``points`` (minimisation) with respect to the
    upper-bound ``reference`` point, for exactly 3 objectives.

    Points that are not strictly below the reference in every objective
    contribute nothing (they lie outside the reference box).

    Parameters
    ----------
    points:
        Iterable of 3-tuples/lists (Cmax, TWET, NEC), minimisation.
    reference:
        3-tuple upper-bound (worse than every point of interest).

    Returns
    -------
    float
        The dominated hypervolume (raw units).
    """

    r0, r1, r2 = (float(reference[0]), float(reference[1]), float(reference[2]))

    # Convert minimisation -> maximisation boxes anchored at the origin:
    # q = reference - p, keep only points strictly inside the reference box.
    boxes = []
    for p in points:
        q0 = r0 - float(p[0])
        q1 = r1 - float(p[1])
        q2 = r2 - float(p[2])
        if q0 > 0.0 and q1 > 0.0 and q2 > 0.0:
            boxes.append((q0, q1, q2))

    if not boxes:
        return 0.0

    boxes = _pareto_max(boxes)

    # Sweep the x (first) dimension from largest to smallest. Between the
    # current x and the next-smaller x, the cross-section is the 2-D union of
    # (y, z) rectangles of all boxes with x >= current x.
    boxes.sort(key=lambda t: t[0], reverse=True)

    volume = 0.0
    for i in range(len(boxes)):
        upper_x = boxes[i][0]
        lower_x = boxes[i + 1][0] if i + 1 < len(boxes) else 0.0
        slab_height = upper_x - lower_x
        if slab_height <= 0.0:
            continue
        area = _area_2d_union([(b[1], b[2]) for b in boxes[: i + 1]])
        volume += area * slab_height

    return volume


# ============================================================
# Normalised hypervolume (comparable across configs)
# ============================================================

def normalized_hypervolume(points, ideal, nadir, ref_factor=1.1):
    """
    HV as a fraction of the reference box, in [0, 1], after normalising each
    objective to [0, 1] with (ideal, nadir) and placing the reference point at
    (ref_factor, ref_factor, ref_factor).

    A degenerate objective (nadir == ideal) is treated as already normalised
    (its normalised value is 0 for every point), so it neither helps nor
    breaks the computation.
    """

    ideal = [float(v) for v in ideal]
    nadir = [float(v) for v in nadir]

    spans = [nadir[k] - ideal[k] for k in range(3)]

    def _norm(p):
        out = []
        for k in range(3):
            if spans[k] <= 0.0:
                out.append(0.0)
            else:
                out.append((float(p[k]) - ideal[k]) / spans[k])
        return out

    normalised = [_norm(p) for p in points]
    reference = (ref_factor, ref_factor, ref_factor)

    raw = hypervolume(normalised, reference)

    box_volume = ref_factor ** 3
    if box_volume <= 0.0:
        return 0.0

    return raw / box_volume
