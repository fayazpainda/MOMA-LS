"""
Wei2021 Reproduction

NSGA-III Environmental Selection

Implements Wei Algorithm 1:

Step 5:
    Non-dominated sorting of Rt.

Steps 6-9:
    Construct St by adding fronts until |St| >= N.

Steps 10-14:
    If |St| > N, identify splitting front Fl and
    calculate K individuals still required.

Step 15:
    Normalize objectives using St.

Step 16:
    Associate every member of St with its closest
    reference direction.

Step 17:
    Calculate niche count using St / Fl.

Step 18:
    Select K members from Fl using niching.

The same normalization/association functions are also
used to prepare metadata for U-NSGA-III mating selection.
"""


import math
import random

import numpy as np


from nondominated_sort import fast_non_dominated_sort



# ============================================================
# Objectives
# ============================================================

OBJECTIVE_NAMES = [

    "Cmax",

    "TWET",

    "NEC"

]



# ============================================================
# Objective Vector
# ============================================================

def objective_vector(
        solution
):

    """
    Return objective vector:

        f(s) =
        [
            Cmax,
            TWET,
            NEC
        ]
    """


    return np.array(

        [

            float(
                solution["Cmax"]
            ),

            float(
                solution["TWET"]
            ),

            float(
                solution["NEC"]
            )

        ],

        dtype=float

    )



# ============================================================
# NSGA-III Normalization
# ============================================================

def normalize_population(
        population
):

    """
    NSGA-III normalization.

    Procedure:

    1. Find ideal point.
    2. Translate objectives.
    3. Identify extreme points using ASF.
    4. Construct intercept hyperplane.
    5. Normalize objectives.

    IMPORTANT:
    During Wei Algorithm 1 environmental selection,
    this function must receive St, NOT the entire Rt.
    """


    if not population:

        return {}


    objective_matrix = np.array(

        [

            objective_vector(
                solution
            )

            for solution
            in population

        ],

        dtype=float

    )



    number_of_objectives = (

        objective_matrix.shape[1]

    )



    # ========================================================
    # Ideal Point
    # ========================================================

    ideal_point = np.min(

        objective_matrix,

        axis=0

    )



    # ========================================================
    # Translate Objectives
    # ========================================================

    translated = (

        objective_matrix

        -

        ideal_point

    )



    # ========================================================
    # Extreme Points Using ASF
    # ========================================================

    extreme_points = []


    for objective_index in range(

        number_of_objectives

    ):


        weights = np.full(

            number_of_objectives,

            1e-6,

            dtype=float

        )


        weights[
            objective_index
        ] = 1.0



        asf_values = np.max(

            translated

            /

            weights,

            axis=1

        )



        extreme_index = int(

            np.argmin(

                asf_values

            )

        )



        extreme_points.append(

            translated[
                extreme_index
            ]

        )



    extreme_points = np.array(

        extreme_points,

        dtype=float

    )



    # ========================================================
    # Hyperplane Intercepts
    # ========================================================

    use_fallback = False



    try:


        coefficients = np.linalg.solve(

            extreme_points,

            np.ones(

                number_of_objectives

            )

        )



        # A zero coefficient makes this divide yield a non-finite intercept,
        # which is explicitly detected and handled by the degenerate-fallback
        # check just below. The numpy divide-by-zero warning is therefore
        # benign; silence it locally (not globally) to keep run output clean.
        with np.errstate(divide="ignore", invalid="ignore"):

            intercepts = (

                1.0

                /

                coefficients

            )



        # ----------------------------------------------------
        # Degenerate / invalid intercepts
        # ----------------------------------------------------

        if (

            np.any(

                ~np.isfinite(
                    intercepts
                )

            )

            or

            np.any(

                intercepts <= 1e-12

            )

        ):

            use_fallback = True



    except np.linalg.LinAlgError:

        use_fallback = True



    # ========================================================
    # Degenerate Hyperplane Fallback
    # ========================================================

    if use_fallback:


        intercepts = np.max(

            translated,

            axis=0

        )



    # Avoid division by zero

    intercepts = np.where(

        intercepts > 1e-12,

        intercepts,

        1.0

    )



    # ========================================================
    # Normalize
    # ========================================================

    normalized_matrix = (

        translated

        /

        intercepts

    )



    normalized = {}



    for solution, vector in zip(

        population,

        normalized_matrix

    ):


        # Python object identity is used because
        # IDs assigned during separate parent and
        # offspring evaluations may overlap.

        normalized[
            id(solution)
        ] = vector



    return normalized



# ============================================================
# Perpendicular Distance
# ============================================================

def perpendicular_distance(
        point,
        reference_direction
):

    """
    Distance d(s) between normalized solution s
    and a reference direction.
    """


    direction = np.array(

        reference_direction,

        dtype=float

    )



    direction_norm = np.linalg.norm(

        direction

    )



    if direction_norm <= 1e-12:

        return math.inf



    unit_direction = (

        direction

        /

        direction_norm

    )



    projection_length = np.dot(

        point,

        unit_direction

    )



    projected_point = (

        projection_length

        *

        unit_direction

    )



    distance = np.linalg.norm(

        point

        -

        projected_point

    )



    return float(

        distance

    )



# ============================================================
# Reference Association
# Wei Algorithm 1 Step 16
# ============================================================

def associate_population(
        population,
        reference_points
):

    """
    Associate every solution with the closest
    reference direction.

    Stores:

        solution["reference_id"] = pi(s)

        solution["perpendicular_distance"] = d(s)
    """


    if not population:

        return


    if not reference_points:

        raise ValueError(

            "Reference point list cannot be empty."

        )



    normalized = normalize_population(

        population

    )



    for solution in population:


        point = normalized[

            id(solution)

        ]



        closest_reference = None

        closest_distance = math.inf



        for reference_id, reference in enumerate(

            reference_points

        ):


            distance = perpendicular_distance(

                point,

                reference

            )



            if distance < closest_distance:


                closest_distance = (

                    distance

                )


                closest_reference = (

                    reference_id

                )



        solution["reference_id"] = (

            closest_reference

        )


        solution[
            "perpendicular_distance"
        ] = (

            closest_distance

        )



# ============================================================
# Non-Dominated Rank
# ============================================================

def assign_ranks(
        population
):

    """
    Perform non-dominated sorting.

    F1:
        rank = 0

    F2:
        rank = 1

    etc.
    """


    fronts = fast_non_dominated_sort(

        population

    )



    for rank, front in enumerate(

        fronts

    ):


        for solution in front:


            solution["rank"] = (

                rank

            )



    return fronts



# ============================================================
# Metadata for Niching-Based Mating Selection
# ============================================================

def prepare_reference_metadata(
        population,
        reference_points
):

    """
    Prepare:

        rank
        reference_id
        perpendicular_distance

    for U-NSGA-III niching-based tournament selection.

    Here the complete parent population Pt is used,
    because these metadata are needed to compare
    members of Pt during mating selection.
    """


    fronts = assign_ranks(

        population

    )



    associate_population(

        population,

        reference_points

    )



    return fronts



# ============================================================
# Wei / NSGA-III Niching
# Algorithm 1 Step 18
# ============================================================

def niching_selection(
        already_selected,
        splitting_front,
        remaining,
        reference_points
):

    """
    Select K = remaining individuals from Fl.

    Niche count rho_j is initially calculated only from:

        St / Fl

    which is the population already selected from
    complete fronts.

    Selection:

    If rho_j == 0:
        select the member of Fl associated with j
        having the smallest perpendicular distance.

    Otherwise:
        select a random member of Fl associated
        with reference direction j.
    """


    if remaining <= 0:

        return []



    chosen = []


    number_of_references = len(

        reference_points

    )



    # ========================================================
    # Algorithm 1 Step 17
    #
    # rho_j =
    #
    # number of members in St / Fl associated
    # with reference point j
    # ========================================================

    niche_count = [

        0

        for _ in range(

            number_of_references

        )

    ]



    for solution in already_selected:


        reference_id = (

            solution[
                "reference_id"
            ]

        )


        niche_count[
            reference_id
        ] += 1



    # ========================================================
    # Candidates from Fl grouped by reference direction
    # ========================================================

    candidates = {

        reference_id: []

        for reference_id in range(

            number_of_references

        )

    }



    for solution in splitting_front:


        reference_id = (

            solution[
                "reference_id"
            ]

        )


        candidates[
            reference_id
        ].append(

            solution

        )



    # ========================================================
    # Algorithm 1 Step 18
    #
    # Select K members one at a time
    # ========================================================

    while len(chosen) < remaining:


        # ----------------------------------------------------
        # Reference directions that still have an
        # available Fl candidate.
        # ----------------------------------------------------

        active_references = [

            reference_id

            for reference_id
            in range(

                number_of_references

            )

            if candidates[
                reference_id
            ]

        ]



        if not active_references:


            # This should not occur when:
            #
            # remaining <= len(splitting_front)
            #
            # and all Fl members have valid associations.

            raise RuntimeError(

                "Niching selection exhausted candidates "
                "before selecting the required K members."

            )



        # ====================================================
        # Find minimum rho_j
        # ====================================================

        minimum_niche_count = min(

            niche_count[
                reference_id
            ]

            for reference_id
            in active_references

        )



        least_crowded_references = [

            reference_id

            for reference_id
            in active_references

            if niche_count[
                reference_id
            ]
            ==
            minimum_niche_count

        ]



        # ----------------------------------------------------
        # If multiple reference points have the same
        # minimum niche count, choose one randomly.
        # ----------------------------------------------------

        reference_id = random.choice(

            least_crowded_references

        )



        associated_candidates = (

            candidates[
                reference_id
            ]

        )



        # ====================================================
        # rho_j == 0
        #
        # Choose candidate having minimum d(s).
        # ====================================================

        if niche_count[
            reference_id
        ] == 0:


            selected_solution = min(

                associated_candidates,

                key=lambda solution:

                    solution[
                        "perpendicular_distance"
                    ]

            )



        # ====================================================
        # rho_j > 0
        #
        # Choose an associated candidate randomly.
        # ====================================================

        else:


            selected_solution = random.choice(

                associated_candidates

            )



        # ----------------------------------------------------
        # Add selected member
        # ----------------------------------------------------

        chosen.append(

            selected_solution

        )



        # ----------------------------------------------------
        # Remove it from Fl candidates
        # ----------------------------------------------------

        candidates[
            reference_id
        ].remove(

            selected_solution

        )



        # ----------------------------------------------------
        # Increase rho_j
        # ----------------------------------------------------

        niche_count[
            reference_id
        ] += 1



    return chosen



# ============================================================
# Environmental Selection
# Wei Algorithm 1 Steps 5-18
# ============================================================

def nsga3_environmental_selection(
        population,
        population_size,
        reference_points
):

    """
    Environmental selection from:

        Rt = Pt union Qt

    following Wei Algorithm 1.
    """


    if not population:

        raise ValueError(

            "Population cannot be empty."

        )


    if population_size <= 0:

        raise ValueError(

            "Population size must be greater than zero."

        )


    if len(population) < population_size:

        raise ValueError(

            "Combined population is smaller than "
            "the requested next population size."

        )



    # ========================================================
    # Algorithm 1 Step 5
    #
    # (F1, F2, ..., Fn) =
    # Non-dominated-sort(Rt)
    # ========================================================

    fronts = assign_ranks(

        population

    )



    # ========================================================
    # Algorithm 1 Steps 6-9
    #
    # St = F1 union F2 ... union Fl
    #
    # Stop once |St| >= N.
    # ========================================================

    st = []


    complete_fronts = []


    splitting_front = None



    for front in fronts:


        # ----------------------------------------------------
        # Adding this front still leaves |St| < N
        # ----------------------------------------------------

        if (

            len(st)
            +
            len(front)

            <
            population_size

        ):


            st.extend(

                front

            )


            complete_fronts.extend(

                front

            )



        # ----------------------------------------------------
        # Adding this front gives exactly N.
        #
        # Algorithm 1 Steps 10-11.
        # ----------------------------------------------------

        elif (

            len(st)
            +
            len(front)

            ==
            population_size

        ):


            st.extend(

                front

            )


            # Pt+1 = St

            return st



        # ----------------------------------------------------
        # This is Fl:
        #
        # adding it makes |St| > N.
        # ----------------------------------------------------

        else:


            splitting_front = (

                front

            )


            st.extend(

                front

            )


            break



    # ========================================================
    # Splitting front must exist here
    # ========================================================

    if splitting_front is None:

        raise RuntimeError(

            "No splitting front was found although "
            "environmental selection requires one."

        )



    # ========================================================
    # Algorithm 1 Step 13
    #
    # Pt+1 =
    # F1 union ... union F(l-1)
    # ========================================================

    next_population = (

        complete_fronts.copy()

    )



    # ========================================================
    # Algorithm 1 Step 14
    #
    # K = N - |Pt+1|
    # ========================================================

    remaining = (

        population_size

        -

        len(next_population)

    )



    # ========================================================
    # Algorithm 1 Step 15
    #
    # Normalize objectives using St.
    #
    # IMPORTANT:
    # This is the major correction from the previous code.
    #
    # We do NOT normalize the entire Rt.
    # ========================================================

    associate_population(

        st,

        reference_points

    )



    # ========================================================
    # Algorithm 1 Steps 17-18
    #
    # Niche counts come from:
    #
    # St / Fl
    #
    # which equals complete_fronts.
    # ========================================================

    selected_from_last_front = niching_selection(

        complete_fronts,

        splitting_front,

        remaining,

        reference_points

    )



    next_population.extend(

        selected_from_last_front

    )



    # ========================================================
    # Validation
    # ========================================================

    if len(next_population) != population_size:


        raise RuntimeError(

            "NSGA-III environmental selection produced "
            f"{len(next_population)} individuals instead "
            f"of {population_size}."

        )



    return next_population